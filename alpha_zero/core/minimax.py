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
from typing import List, Set, Dict, Optional, Tuple, Callable
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
    """
    
    def __init__(
        self, 
        num_threads: int = 4,
        min_batch_size: int = 16,
        max_batch_size: int = 128,
        virtual_loss: float = 0.1,
        time_limit: float = 30.0  # seconds
    ):
        """Initialize the parallel minimax searcher.
        
        Args:
            num_threads: Number of parallel search threads
            min_batch_size: Minimum positions per batch evaluation
            max_batch_size: Maximum positions per batch evaluation
            virtual_loss: Virtual loss value to discourage thread collision
            time_limit: Maximum search time in seconds
        """
        self.num_threads = num_threads
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_size = max_batch_size 
        self.virtual_loss = virtual_loss
        self.time_limit = time_limit
        self.tt_lock = threading.Lock()
        self.evaluation_event = threading.Event()
        
    def get_dynamic_batch_size(self, depth: int, max_depth: int) -> int:
        """Dynamically adjust batch size based on search depth."""
        depth_ratio = depth / max_depth
        # Exponentially decrease batch size as depth increases
        batch_size = int(
            self.max_batch_size * np.exp(-2 * depth_ratio) + 
            self.min_batch_size
        )
        return max(self.min_batch_size, min(batch_size, self.max_batch_size))
        
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
        k_best: int,
        transposition_table: TranspositionTable,
        move_scores: Dict[int, float],
        mcts_prior_map: Dict[Tuple[int, int], float] = None,
    ) -> Tuple[float, List[int]]:
        """
        Enhanced parallel minimax search that returns principal variation.
        
        Args:
            env: Board game environment
            eval_func: Position evaluation function
            depth: Current search depth
            k_best: Number of best moves to consider
            transposition_table: Cache of searched positions
            move_scores: Dictionary of move scores for move ordering
            mcts_prior_map: optional dictionary of MCTS priors for move ordering
            
        Returns:
            (best_value, principal_variation)
        """
        
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
        
        Now integrates MCTS priors into the move ordering if provided.
        
        Args:
            env: Board game environment
            eval_func: Position evaluation function
            depth: Current search depth
            k_best: Number of best moves to consider
            alpha: Lower bound
            beta: Upper bound
            transposition_table: Cache of searched positions
            move_scores: Dictionary of move scores for move ordering
            mcts_prior_map: Dictionary of MCTS priors for move ordering (if available)
            
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

        # ----- NEW: Incorporate MCTS priors into move ordering -----
        # We'll combine MCTS prior and move_scores into one "combined_score".
        # If no mcts_prior_map is given, we fallback to just move_scores.
        scored_moves = []
        for action in legal_actions:
            prior_val = 0.0
            if mcts_prior_map is not None:
                prior_val = mcts_prior_map.get((pos_hash, action), 0.0)
            
            # Combine with old move_scores however you like:
            # For example, we can do a simple weighted sum:
            #  combined_score = 0.7 * prior_val + 0.3 * move_scores[action]
            # Or tune these weights as you see fit.
            move_score = move_scores.get(action, 0.0)
            combined_score = 0.7 * prior_val + 0.3 * move_score
            
            scored_moves.append((action, combined_score))
        
        # Sort by combined score descending if maximizing, ascending if not
        scored_moves.sort(key=lambda x: x[1], reverse=maximizing)
        
        # Slice top k
        ordered_moves = scored_moves[:k_best]

        # Alpha-beta over the ordered moves
        for action, _ in ordered_moves:
            next_env = copy.deepcopy(env)
            next_env.step(action)
            
            value, pv = self.alpha_beta_search_with_pv(
                next_env,
                eval_func,
                depth - 1,
                k_best,
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
