# Copyright (c) 2023 Michael Hu. This code is part of the book "The Art of Reinforcement Learning: Fundamentals,
# Mathematics, and Implementation with Python.". This project is released under the MIT License. See the accompanying
# LICENSE file for details.


"""A high-performance, neurally-guided alpha-beta search engine for the AlphaZero agent."""

import time
import threading
from enum import Enum
from typing import Callable, Tuple, Iterable, Any, Optional

import numpy as np

from alpha_zero.envs.base import BoardGameEnv


# A thread-local storage for killer moves to ensure thread safety in parallel search environments.
thread_local = threading.local()


class NodeType(Enum):
    """
    Enumeration for the type of node in the transposition table.
    - EXACT: The stored value is the exact evaluation of the node.
    - LOWERBOUND: The stored value is a lower bound on the evaluation (alpha).
    - UPPERBOUND: The stored value is an upper bound on the evaluation (beta).
    """

    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2


class TranspositionTable:
    """
    A thread-safe transposition table for storing and retrieving previously computed states.
    It uses a dictionary for storage and a lock to handle concurrent access.
    """

    def __init__(self, size=100000):
        """
        Initialize the transposition table with a given size.

        Args:
            size: The maximum number of entries the table can hold.
        """
        self.table = {}
        self.size = size
        self.lock = threading.Lock()

    def store(self, zobrist_hash: int, depth: int, value: float, flag: NodeType, best_move: Optional[int] = None):
        """
        Store a new entry in the transposition table.

        Args:
            zobrist_hash: The hash of the current board state.
            depth: The depth at which this state was evaluated.
            value: The evaluation value of the current state.
            flag: The type of node (EXACT, LOWERBOUND, UPPERBOUND).
            best_move: The best move found from this position.
        """
        with self.lock:
            if len(self.table) >= self.size:
                # Simple FIFO replacement strategy
                self.table.pop(next(iter(self.table)))
            self.table[zobrist_hash] = (depth, value, flag, best_move)

    def lookup(self, zobrist_hash: int) -> Optional[Tuple[int, float, NodeType, Optional[int]]]:
        """
        Retrieve an entry from the transposition table if it exists.

        Args:
            zobrist_hash: The hash of the current board state.

        Returns:
            A tuple containing the depth, value, flag, and best_move of the state if found,
            otherwise None.
        """
        with self.lock:
            return self.table.get(zobrist_hash)


class AlphaBetaEngine:
    """
    A neurally-guided alpha-beta search engine.
    """

    def __init__(
        self,
        eval_func: Callable[[np.ndarray, bool], Tuple[np.ndarray, float]],
        tt_size: int = 100000,
    ):
        """
        Initialize the alpha-beta search engine.

        Args:
            eval_func: The neural network evaluation function.
            tt_size: The size of the transposition table.
        """
        self.eval_func = eval_func
        self.transposition_table = TranspositionTable(size=tt_size)
        self.history_table = {}  # To be implemented
        self.nodes_searched = 0

    def search(
        self,
        env: BoardGameEnv,
        depth: int,
        alpha: float = -np.inf,
        beta: float = np.inf,
        k_best: int = 8,
    ) -> float:
        """
        Performs a depth-limited minimax search with alpha-beta pruning.

        Args:
            env: The game environment.
            depth: The maximum depth to search.
            alpha: The alpha value for alpha-beta pruning.
            beta: The beta value for alpha-beta pruning.
            k_best: Number of best moves to consider at each node, based on policy.

        Returns:
            The best evaluation value found.
        """
        self.nodes_searched = 0
        
        # Initialize thread-local killer moves table if not present
        if not hasattr(thread_local, 'killer_moves'):
            thread_local.killer_moves = {}

        return self._search(env, depth, alpha, beta, k_best)

    def _search(
        self,
        env: BoardGameEnv,
        depth: int,
        alpha: float,
        beta: float,
        k_best: int,
    ) -> float:
        """Recursive helper for the alpha-beta search."""
        
        # 1. Transposition Table Lookup
        zobrist_hash = env.zobrist_hash()
        tt_entry = self.transposition_table.lookup(zobrist_hash)
        if tt_entry is not None:
            stored_depth, stored_value, stored_flag, best_move = tt_entry
            if stored_depth >= depth:
                if stored_flag == NodeType.EXACT:
                    return stored_value
                elif stored_flag == NodeType.LOWERBOUND:
                    alpha = max(alpha, stored_value)
                elif stored_flag == NodeType.UPPERBOUND:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value

        # 2. Terminal Node or Max Depth Reached
        if depth == 0 or env.is_game_over():
            self.nodes_searched += 1
            _, value = self.eval_func(env.observation(), False)
            return value

        # 3. Neural Move Ordering
        legal_actions = env.legal_actions
        policy_probs, _ = self.eval_func(env.observation(), False)
        
        # Combine policy probabilities with legality
        move_scores = {action: policy_probs[action] for action, is_legal in enumerate(legal_actions) if is_legal}
        
        # Sort moves by policy probability in descending order
        sorted_moves = sorted(move_scores.keys(), key=lambda move: move_scores[move], reverse=True)
        
        # Prune to top-k moves
        if k_best is not None:
            sorted_moves = sorted_moves[:k_best]

        # 4. Alpha-Beta Search
        best_value = -np.inf
        best_move = None
        
        for move in sorted_moves:
            sim_env = env.clone()
            sim_env.step(move)
            
            # Negate alpha/beta for the opponent's turn
            value = -self._search(sim_env, depth - 1, -beta, -alpha, k_best)

            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, best_value)
            
            if alpha >= beta:
                # Beta cutoff
                break
        
        # 5. Transposition Table Store
        flag = NodeType.EXACT
        if best_value <= alpha:
            flag = NodeType.UPPERBOUND
        elif best_value >= beta:
            flag = NodeType.LOWERBOUND
            
        self.transposition_table.store(zobrist_hash, depth, best_value, flag, best_move)

        return best_value

    def solver_search(self, env: BoardGameEnv, depth: int = 10) -> Tuple[bool, float]:
        """
        Performs a deep search to find a proven win/loss.

        Args:
            env: The game environment.
            depth: The depth of the search.

        Returns:
            A tuple (is_proven, value).
        """
        value = self.search(env, depth)
        
        # A position is considered "proven" if the value is extremely high or low.
        is_proven = abs(abs(value) - 1.0) < 1e-4
        
        return is_proven, value