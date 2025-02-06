"""
Parallel Minimax Search Implementation with Alpha-Beta Pruning.

This module implements a parallel minimax search algorithm optimized for go.
- Parallel search using multiple threads
- Transposition table for caching positions
- Principal variation collection
- Move ordering based on previous search results
- Dynamic batch sizing for position evaluation
- Iterative deepening with time management
- Move ordering using MCTS priors
"""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Set, Dict, Optional, Tuple, Callable, Iterable
import numpy as np
import time
import copy

from alpha_zero.envs.base import BoardGameEnv
from alpha_zero.core.transposition_table import TranspositionTable, NodeType

@dataclass
class SearchWindow:
    """Represents a parallel alpha-beta search window.
    
    Attributes:
        alpha: Lower bound of the search window
        beta: Upper bound of the search window 
        depth: Current search depth
        collected_leaves: List of (board_state, move_path) tuples for batch evaluation
        lock: Thread lock for synchronizing access to collected_leaves
    """
    alpha: float
    beta: float
    depth: int
    collected_leaves: List[tuple]  # [(env_state, path), ...]
    lock: threading.Lock

class ParallelMinimax:
    """Implements parallel minimax search with batched evaluation and iterative deepening.
    
    This class manages parallel search across multiple threads, each exploring different
    search windows. It uses batch evaluation of positions and maintains a transposition
    table to cache search results.
    
    Attributes:
        num_threads: Number of parallel search threads
        min_batch_size: Minimum number of positions to evaluate in a batch
        max_batch_size: Maximum number of positions to evaluate in a batch
        virtual_loss: Penalty applied to positions being searched by other threads
        time_limit: Maximum search time in seconds
        tt_lock: Lock for transposition table access
        evaluation_event: Event to trigger batch evaluation
        base_k: Base number of moves to consider at each node
        min_k: Minimum number of moves to consider
        max_k: Maximum number of moves to consider
    """
    
    def __init__(
        self, 
        num_threads: int = 4,
        min_batch_size: int = 16,
        max_batch_size: int = 128,
        virtual_loss: float = 0.1,
        time_limit: float = 30.0,  # seconds
        base_k: int = 5,  # Base number of moves to consider at each node
        min_k: int = 3,  # Minimum number of moves to consider
        max_k: int = 20,  # Maximum number of moves to consider
    ):
        """Initialize the parallel minimax searcher.
        
        Args:
            num_threads: Number of parallel search threads
            min_batch_size: Minimum positions per batch evaluation
            max_batch_size: Maximum positions per batch evaluation
            virtual_loss: Virtual loss value to discourage thread collision
            time_limit: Maximum search time in seconds
            base_k: Base number of moves to consider at each node
            min_k: Minimum number of moves to consider
            max_k: Maximum number of moves to consider
        """
        self.num_threads = num_threads
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_size = max_batch_size 
        self.virtual_loss = virtual_loss
        self.time_limit = time_limit
        self.tt_lock = threading.Lock()
        self.evaluation_event = threading.Event()
        
        # Parameters for dynamic k-best selection
        self.base_k = base_k
        self.min_k = min_k
        self.max_k = max_k
        
    def get_dynamic_batch_size(self, depth: int, max_depth: int) -> int:
        """Dynamically adjust batch size based on search depth."""
        depth_ratio = depth / max_depth
        # Exponentially decrease batch size as depth increases
        batch_size = int(
            self.max_batch_size * np.exp(-2 * depth_ratio) + 
            self.min_batch_size
        )
        return max(self.min_batch_size, min(batch_size, self.max_batch_size))
        
    def get_dynamic_k(
        self,
        depth: int,
        max_depth: int,
        move_scores: Dict[int, float],
        legal_actions: np.ndarray,
        mcts_prior_map: Dict[Tuple[int, int], float] = None,
        pos_hash: Optional[int] = None,
    ) -> int:
        """Calculate dynamic k-best value based on position complexity and depth.
        
        Args:
            depth: Current search depth
            max_depth: Maximum search depth
            move_scores: Dictionary of move scores
            legal_actions: Boolean mask of legal moves
            mcts_prior_map: Optional dictionary of MCTS priors
            pos_hash: Optional hash of current position
            
        Returns:
            Number of moves to consider
        """
        # Base k value that decreases with depth
        depth_ratio = depth / max_depth
        k = int(self.base_k * (1 + np.exp(-2 * depth_ratio)))
        
        # Adjust based on move score distribution
        if move_scores:
            scores = np.array([move_scores.get(a, 0.0) for a in range(len(legal_actions))])
            scores = scores[legal_actions == 1]  # Only consider legal moves
            if len(scores) > 0:
                # Calculate score spread
                score_range = np.max(scores) - np.min(scores)
                # Increase k if moves have similar scores
                if score_range < 0.2:  # Threshold for "similar" scores
                    k = int(k * 1.5)
                    
        # Consider MCTS priors if available
        if mcts_prior_map and pos_hash is not None:
            priors = []
            for action in range(len(legal_actions)):
                if legal_actions[action]:
                    prior = mcts_prior_map.get((pos_hash, action), 0.0)
                    priors.append(prior)
            if priors:
                priors = np.array(priors)
                # If priors are concentrated, reduce k
                top_prior_sum = np.sum(np.sort(priors)[-3:])  # Sum of top 3 priors
                if top_prior_sum > 0.7:  # If top moves have high probability
                    k = int(k * 0.7)
                    
        # Ensure k stays within bounds
        return max(self.min_k, min(k, self.max_k))
        
    def iterative_deepening_search(
        self,
        env: BoardGameEnv,
        eval_func: Callable,
        max_depth: int,
        k_best: int,
        transposition_table: TranspositionTable,
        mcts_prior_map: Dict[Tuple[int, int], float] = None,
    ) -> Tuple[float, List[int]]:
        """
        Performs iterative deepening search with parallel minimax.
        
        Args:
            env: Board game environment
            eval_func: Position evaluation function
            max_depth: Maximum search depth
            k_best: Number of best moves to consider at each node
            transposition_table: Cache of previously searched positions
            mcts_prior_map: optional dictionary of MCTS priors for move ordering

        Returns:
            Tuple of (best_value, principal_variation)
        """
        start_time = time.time()
        best_value = 0.0
        best_pv = []
        
        # Initialize move ordering scores
        legal_actions = np.where(env.legal_actions == 1)[0]
        move_scores = {action: 0.0 for action in legal_actions}
        
        for current_depth in range(1, max_depth + 1):
            if time.time() - start_time > self.time_limit:
                break
                
            # Adjust batch size for current depth
            self.batch_size = self.get_dynamic_batch_size(current_depth, max_depth)
            
            # Run parallel search at current depth
            value, pv = self.parallel_minimax_search_with_pv(
                env,
                eval_func,
                current_depth,
                k_best,
                transposition_table,
                move_scores,
                mcts_prior_map  # pass along the prior map
            )
            
            # Check if we completed this depth iteration within time
            if time.time() - start_time <= self.time_limit:
                best_value = value
                best_pv = pv
                
                # Update move ordering scores for next iteration
                for action in legal_actions:
                    next_env = copy.deepcopy(env)
                    next_env.step(action)
                    tt_entry = transposition_table.lookup(next_env.zobrist_hash())
                    if tt_entry:
                        # stored_value is at index [1] in the TT entry
                        move_scores[action] = tt_entry[1]
                        
        return best_value, best_pv
        
    def parallel_minimax_search_with_pv(
        self,
        env: BoardGameEnv,
        eval_func: Callable,
        depth: int,
        k_best: int,  # This is now used as a maximum k
        transposition_table: TranspositionTable,
        move_scores: Dict[int, float],
        mcts_prior_map: Dict[Tuple[int, int], float] = None,
    ) -> Tuple[float, List[int]]:
        """
        Enhanced parallel minimax search that returns principal variation.
        Now uses dynamic k-best selection.
        
        Args:
            env: Board game environment
            eval_func: Position evaluation function
            depth: Current search depth
            k_best: Maximum number of moves to consider
            transposition_table: Cache of searched positions
            move_scores: Dictionary of move scores for move ordering
            mcts_prior_map: optional dictionary of MCTS priors for move ordering
            
        Returns:
            (best_value, principal_variation)
        """
        # Get dynamic k for root node
        dynamic_k = self.get_dynamic_k(
            depth=depth,
            max_depth=depth,  # At root, current depth is max depth
            move_scores=move_scores,
            legal_actions=env.legal_actions,
            mcts_prior_map=mcts_prior_map,
            pos_hash=env.zobrist_hash()
        )
        k_best = min(k_best, dynamic_k)  # Use the smaller of the two
        
        # Create search windows
        window_size = 0.2
        windows = []
        for i in range(self.num_threads):
            alpha = -1.0 + i * window_size
            beta = alpha + window_size
            windows.append(SearchWindow(
                alpha=alpha,
                beta=beta,
                depth=depth,
                collected_leaves=[],
                lock=threading.Lock()
            ))
            
        # Shared state
        explored_positions = set()
        explored_lock = threading.Lock()
        best_pv = []
        pv_lock = threading.Lock()
        
        def window_search(window: SearchWindow) -> Optional[Tuple[float, List[int]]]:
            """Search function for a single window, returning PV."""
            
            def should_collect_leaf(env_state, current_depth: int) -> bool:
                """Decide if we should treat this as a leaf for batch evaluation."""
                if current_depth >= depth:
                    return True
                    
                pos_hash = env_state.zobrist_hash()
                with explored_lock:
                    if pos_hash in explored_positions:
                        return False
                    explored_positions.add(pos_hash)
                    
                tt_entry = transposition_table.lookup(pos_hash)
                if tt_entry is not None:
                    stored_depth, stored_value, _ = tt_entry
                    # apply a small virtual loss
                    if stored_depth >= current_depth:
                        stored_value -= self.virtual_loss
                        return False
                        
                return True
                
            def collect_leaves(
                search_env: BoardGameEnv,
                current_depth: int,
                path: List[int],
                alpha: float,
                beta: float
            ) -> None:
                """Recursively collect leaves for batch evaluation."""
                
                if should_collect_leaf(search_env, current_depth):
                    with window.lock:
                        window.collected_leaves.append((
                            copy.deepcopy(search_env),
                            path.copy()
                        ))
                    if len(window.collected_leaves) >= self.batch_size:
                        self.evaluation_event.set()
                    return
                    
                legal_actions = np.where(search_env.legal_actions == 1)[0]
                
                # Move ordering: existing approach (top k by move_scores)
                ordered_moves = [
                    (action, move_scores.get(action, 0.0))
                    for action in legal_actions[:k_best]
                ]
                # If you want partial integration of priors here, you can do so,
                # but typically you'd rely on alpha_beta_search_with_pv for that.
                ordered_moves.sort(key=lambda x: x[1], reverse=search_env.to_play)
                
                for action, _ in ordered_moves:
                    next_env = copy.deepcopy(search_env)
                    next_env.step(action)
                    path.append(action)
                    collect_leaves(next_env, current_depth + 1, path, -beta, -alpha)
                    path.pop()
                    
            # Collect leaves
            collect_leaves(copy.deepcopy(env), 0, [], window.alpha, window.beta)
            if window.collected_leaves:
                self.process_collected_leaves(window, eval_func, transposition_table)
                
            # Final search with PV
            value, pv = self.alpha_beta_search_with_pv(
                copy.deepcopy(env),
                eval_func,
                depth,
                k_best,
                window.alpha,
                window.beta,
                transposition_table,
                move_scores,
                mcts_prior_map  # pass the priors down
            )
            
            # Update best PV if needed
            if value is not None:
                with pv_lock:
                    if not best_pv or value > best_pv[0]:
                        best_pv[:] = [value] + pv
                        
            return (value, pv) if value is not None else None
            
        # Run parallel searches
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_results = [
                executor.submit(window_search, window)
                for window in windows
            ]
            results = [f.result() for f in future_results]
            valid_results = [r for r in results if r is not None]
            
            if not valid_results:
                return 0.0, []
                
            best_result = max(valid_results, key=lambda x: x[0])
            return best_result
            
    def alpha_beta_search_with_pv(
        self,
        env: BoardGameEnv,
        eval_func: Callable,
        depth: int,
        k_best: int,
        alpha: float,
        beta: float,
        transposition_table: TranspositionTable,
        move_scores: Dict[int, float],
        mcts_prior_map: Dict[Tuple[int, int], float] = None,
    ) -> Tuple[float, List[int]]:
        """
        Alpha-beta search that returns principal variation.
        Now uses dynamic k-best selection at each node.
        
        Args:
            env: Board game environment
            eval_func: Position evaluation function
            depth: Current search depth
            k_best: Maximum number of moves to consider
            alpha: Lower bound
            beta: Upper bound
            transposition_table: Cache of searched positions
            move_scores: Dictionary of move scores for move ordering
            mcts_prior_map: Dictionary of MCTS priors for move ordering
            
        Returns:
            (position_value, principal_variation)
        """
        pos_hash = env.zobrist_hash()
        tt_entry = transposition_table.lookup(pos_hash)
        
        # Transposition table check
        if tt_entry is not None:
            stored_depth, stored_value, stored_flag = tt_entry
            if stored_depth >= depth:
                if stored_flag == NodeType.EXACT:
                    return stored_value, []
                elif stored_flag == NodeType.LOWERBOUND:
                    alpha = max(alpha, stored_value)
                elif stored_flag == NodeType.UPPERBOUND:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value, []
                    
        # Base case
        if depth == 0 or env.is_game_over():
            obs = env.observation()
            _, value = eval_func(obs, False)
            with self.tt_lock:
                transposition_table.store(pos_hash, depth, value, NodeType.EXACT)
            return value, []
            
        legal_actions = np.where(env.legal_actions == 1)[0]
        maximizing = env.to_play
        best_value = float('-inf') if maximizing else float('inf')
        best_pv = []

        # Get dynamic k for this node
        dynamic_k = self.get_dynamic_k(
            depth=depth,
            max_depth=k_best,  # Use k_best as max_depth since it's our upper bound
            move_scores=move_scores,
            legal_actions=env.legal_actions,
            mcts_prior_map=mcts_prior_map,
            pos_hash=pos_hash
        )
        k_best = min(k_best, dynamic_k)

        # Score and sort moves
        scored_moves = []
        for action in legal_actions:
            # Combine MCTS prior and move scores
            prior_val = mcts_prior_map.get((pos_hash, action), 0.0) if mcts_prior_map else 0.0
            move_score = move_scores.get(action, 0.0)
            combined_score = 0.7 * prior_val + 0.3 * move_score
            scored_moves.append((action, combined_score))
            
        # Sort by combined score
        scored_moves.sort(key=lambda x: x[1], reverse=maximizing)
        
        # Only consider top k moves
        ordered_moves = scored_moves[:k_best]

        # Alpha-beta over the ordered moves
        for action, _ in ordered_moves:
            next_env = copy.deepcopy(env)
            next_env.step(action)
            
            value, pv = self.alpha_beta_search_with_pv(
                next_env,
                eval_func,
                depth - 1,
                k_best,  # Pass along current k_best as upper bound
                -beta,
                -alpha,
                transposition_table,
                move_scores,
                mcts_prior_map
            )
            value = -value
            
            if maximizing:
                if value > best_value:
                    best_value = value
                    best_pv = [action] + pv
                alpha = max(alpha, value)
            else:
                if value < best_value:
                    best_value = value
                    best_pv = [action] + pv
                beta = min(beta, value)
                
            if beta <= alpha:
                break
                
        # Store in TT
        with self.tt_lock:
            if best_value <= alpha:
                flag = NodeType.UPPERBOUND
            elif best_value >= beta:
                flag = NodeType.LOWERBOUND
            else:
                flag = NodeType.EXACT
            transposition_table.store(pos_hash, depth, best_value, flag)
            
        return best_value, best_pv

    def process_collected_leaves(
        self,
        window: SearchWindow,
        eval_func: Callable,
        transposition_table: TranspositionTable,
    ) -> None:
        """Process collected leaf nodes in batch.
        
        Args:
            window: Search window containing collected leaves
            eval_func: Position evaluation function
            transposition_table: Cache of searched positions
        """
        if not window.collected_leaves:
            return

        # Extract states and paths
        states, paths = zip(*window.collected_leaves)
        
        # Get observations for batch evaluation
        observations = np.stack([state.observation() for state in states])
        
        # Evaluate positions in batch
        _, values = eval_func(observations, True)
        
        # Store results in transposition table - ensure values are properly negated based on player
        with self.tt_lock:
            for state, path, value in zip(states, paths, values):
                pos_hash = state.zobrist_hash()
                transposition_table.store(
                    pos_hash,
                    len(path),  # depth
                    float(value),
                    NodeType.EXACT
                )
        
        # Clear processed leaves
        window.collected_leaves.clear()

def minimax(
    env: BoardGameEnv,
    eval_func: Callable[[np.ndarray, bool], Tuple[Iterable[np.ndarray], Iterable[float]]],
    depth: int,
    k_best: int = None,
    transposition_table: Optional[Dict] = None,
    alpha: float = float('-inf'),
    beta: float = float('inf'),
) -> float:
    """
    Performs a depth-limited minimax search with alpha-beta pruning, move ordering, and transposition tables.

    Args:
        env: The game environment.
        eval_func: Evaluation function that returns action probabilities and predicted values.
        depth: The maximum depth to search.
        k_best: Number of best moves to consider at each depth.
        transposition_table: Table to store and retrieve previously computed states.
        alpha: The alpha value for alpha-beta pruning.
        beta: The beta value for alpha-beta pruning.

    Returns:
        The best evaluation value found within the given depth constraints.
    """
    if depth == 0 or env.is_game_over():
        obs = env.observation()
        _, value = eval_func(obs, False)
        assert isinstance(value, float), f"Expected scalar, got {type(value)}"
        return value

    zobrist_hash = env.zobrist_hash()
    if transposition_table is not None:
        tt_entry = transposition_table.lookup(zobrist_hash)
        if tt_entry is not None:
            stored_depth, stored_value, stored_flag = tt_entry
            if stored_depth >= depth:
                if stored_flag == NodeType.EXACT:
                    return stored_value
                elif stored_flag == NodeType.LOWERBOUND:
                    alpha = max(alpha, stored_value)
                elif stored_flag == NodeType.UPPERBOUND:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value

    legal_actions = np.where(env.legal_actions == 1)[0]

    # Move ordering based on evaluation scores
    move_scores = []
    for action in legal_actions:
        sim_env = copy.deepcopy(env)
        sim_env.step(action)
        obs = sim_env.observation()
        _, value = eval_func(obs, False)
        move_scores.append((action, value))

    # Sort moves by value (descending for maximizing player, ascending for minimizing)
    maximizing_player = env.to_play
    move_scores.sort(key=lambda x: x[1], reverse=maximizing_player)

    if k_best is not None:
        move_scores = move_scores[:k_best]

    best_value = float('-inf') if maximizing_player else float('inf')

    for action, _ in move_scores:
        sim_env = copy.deepcopy(env)
        sim_env.step(action)
        child_value = minimax(
            sim_env,
            eval_func,
            depth - 1,
            k_best,
            transposition_table,
            alpha,
            beta
        )

        if maximizing_player:
            best_value = max(best_value, child_value)
            alpha = max(alpha, best_value)
        else:
            best_value = min(best_value, child_value)
            beta = min(beta, best_value)

        if beta <= alpha:
            break

    # Store the result in the transposition table
    if transposition_table is not None:
        if best_value <= alpha:
            flag = NodeType.UPPERBOUND
        elif best_value >= beta:
            flag = NodeType.LOWERBOUND
        else:
            flag = NodeType.EXACT
        transposition_table.store(zobrist_hash, depth, best_value, flag)

    return best_value
