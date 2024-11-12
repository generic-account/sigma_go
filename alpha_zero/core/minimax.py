from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Set, Dict, Optional, Tuple
import numpy as np
import time

@dataclass
class SearchWindow:
    """Represents a parallel alpha-beta search window."""
    alpha: float
    beta: float
    depth: int
    collected_leaves: List[tuple]  # [(env_state, path), ...]
    lock: threading.Lock

class ParallelMinimax:
    """Implements parallel minimax search with batched evaluation and iterative deepening."""
    
    def __init__(
        self, 
        num_threads: int = 4,
        min_batch_size: int = 16,
        max_batch_size: int = 128,
        virtual_loss: float = 0.1,
        time_limit: float = 30.0  # seconds
    ):
        self.num_threads = num_threads
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.virtual_loss = virtual_loss
        self.time_limit = time_limit
        self.tt_lock = threading.Lock()
        self.evaluation_event = threading.Event()
        
    def get_dynamic_batch_size(self, depth: int, max_depth: int) -> int:
        """
        Dynamically adjust batch size based on search depth.
        Smaller batches for deeper searches to maintain responsiveness.
        """
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
    ) -> Tuple[float, List[int]]:
        """
        Performs iterative deepening search with parallel minimax.
        Returns best value and principal variation.
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
                move_scores
            )
            
            # Check if we completed this depth iteration
            if time.time() - start_time <= self.time_limit:
                best_value = value
                best_pv = pv
                
                # Update move ordering scores for next iteration
                for action in legal_actions:
                    next_env = copy.deepcopy(env)
                    next_env.step(action)
                    tt_entry = transposition_table.lookup(next_env.zobrist_hash())
                    if tt_entry:
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
    ) -> Tuple[float, List[int]]:
        """Enhanced parallel minimax search that returns principal variation."""
        
        # Create search windows with dynamic batch sizes
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
            
        # Shared state for parallel search
        explored_positions = set()
        explored_lock = threading.Lock()
        best_pv = []
        pv_lock = threading.Lock()
        
        def window_search(window: SearchWindow) -> Optional[Tuple[float, List[int]]]:
            """Search function for a single window, now returning PV."""
            
            def should_collect_leaf(env_state, current_depth: int) -> bool:
                """Enhanced leaf collection criteria."""
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
                """Collect leaves with dynamic batching."""
                
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
                
                # Use stored move scores for ordering
                ordered_moves = [
                    (action, move_scores.get(action, 0.0))
                    for action in legal_actions[:k_best]
                ]
                ordered_moves.sort(key=lambda x: x[1], reverse=search_env.to_play)
                
                for action, _ in ordered_moves:
                    next_env = copy.deepcopy(search_env)
                    next_env.step(action)
                    path.append(action)
                    collect_leaves(next_env, current_depth + 1, path, -beta, -alpha)
                    path.pop()
                    
            # Collect and process leaves
            collect_leaves(copy.deepcopy(env), 0, [], window.alpha, window.beta)
            if window.collected_leaves:
                self.process_collected_leaves(window, eval_func, transposition_table)
                
            # Final search with PV collection
            value, pv = self.alpha_beta_search_with_pv(
                copy.deepcopy(env),
                eval_func,
                depth,
                k_best,
                window.alpha,
                window.beta,
                transposition_table,
                move_scores
            )
            
            # Update best PV if this window found a better line
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
                
            # Return best value and its principal variation
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
    ) -> Tuple[float, List[int]]:
        """Alpha-beta search that returns principal variation."""
        pos_hash = env.zobrist_hash()
        tt_entry = transposition_table.lookup(pos_hash)
        
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
        
        # Use move ordering based on stored scores
        ordered_moves = [
            (action, move_scores.get(action, 0.0))
            for action in legal_actions[:k_best]
        ]
        ordered_moves.sort(key=lambda x: x[1], reverse=maximizing)
        
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
                move_scores
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
                
        # Store result in transposition table
        with self.tt_lock:
            if best_value <= alpha:
                flag = NodeType.UPPERBOUND
            elif best_value >= beta:
                flag = NodeType.LOWERBOUND
            else:
                flag = NodeType.EXACT
            transposition_table.store(pos_hash, depth, best_value, flag)
            
        return best_value, best_pv
