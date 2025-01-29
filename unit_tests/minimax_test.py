import unittest
import numpy as np
from unittest.mock import Mock, patch
import threading
from typing import Dict, Tuple

from alpha_zero.core.minimax import ParallelMinimax, SearchWindow
from alpha_zero.core.transposition_table import TranspositionTable, NodeType
from alpha_zero.envs.base import BoardGameEnv

class MockEnv(BoardGameEnv):
    """Mock environment for testing minimax."""
    def __init__(self):
        self.board_size = 3
        self.legal_actions = np.ones(9, dtype=np.int8)
        self.current_player = 1
        self._hash = 0
        
    def observation(self):
        return np.zeros((3, 3, 3))
        
    def step(self, action):
        self._hash = hash(str(action))  # Simple hash for testing
        self.current_player *= -1
        
    def zobrist_hash(self):
        return self._hash
        
    @property
    def to_play(self):
        return self.current_player == 1
        
    def is_game_over(self):
        return False

class TestParallelMinimax(unittest.TestCase):
    def setUp(self):
        self.minimax = ParallelMinimax(
            num_threads=2,
            min_batch_size=4,
            max_batch_size=8,
            time_limit=1.0
        )
        self.env = MockEnv()
        self.tt = TranspositionTable()
        
        # Mock evaluation function
        self.eval_func = Mock(return_value=(None, np.array([0.5])))

    def test_dynamic_batch_size(self):
        """Test batch size adjustment based on depth."""
        size1 = self.minimax.get_dynamic_batch_size(depth=1, max_depth=4)
        size2 = self.minimax.get_dynamic_batch_size(depth=4, max_depth=4)
        
        self.assertTrue(size1 > size2)  # Batch size should decrease with depth
        self.assertTrue(self.minimax.min_batch_size <= size2 <= self.minimax.max_batch_size)

    def test_move_ordering_with_mcts_priors(self):
        """Test that move ordering correctly incorporates MCTS priors."""
        # Setup mock MCTS priors
        env_hash = self.env.zobrist_hash()
        mcts_priors = {
            (env_hash, 0): 0.8,  # Strong prior for move 0
            (env_hash, 1): 0.2,  # Weak prior for move 1
        }
        
        # Setup move scores
        move_scores = {0: 0.3, 1: 0.7}  # Different ordering from priors
        
        # Run search with both priors and move scores
        value, pv = self.minimax.alpha_beta_search_with_pv(
            self.env,
            self.eval_func,
            depth=2,
            k_best=2,
            alpha=-1.0,
            beta=1.0,
            transposition_table=self.tt,
            move_scores=move_scores,
            mcts_prior_map=mcts_priors
        )
        
        # Move 0 should be tried first due to strong prior
        self.assertTrue(len(pv) > 0)
        self.assertEqual(pv[0], 0)

    def test_parallel_search(self):
        """Test that parallel search works correctly."""
        value, pv = self.minimax.parallel_minimax_search_with_pv(
            self.env,
            self.eval_func,
            depth=2,
            k_best=2,
            transposition_table=self.tt,
            move_scores={},
            mcts_prior_map=None
        )
        
        self.assertIsNotNone(value)
        self.assertIsInstance(pv, list)

    def test_transposition_table_usage(self):
        """Test that transposition table is properly used."""
        # First search should populate TT
        self.minimax.alpha_beta_search_with_pv(
            self.env,
            self.eval_func,
            depth=2,
            k_best=2,
            alpha=-1.0,
            beta=1.0,
            transposition_table=self.tt,
            move_scores={},
            mcts_prior_map=None
        )
        
        # Verify TT contains entries
        pos_hash = self.env.zobrist_hash()
        tt_entry = self.tt.lookup(pos_hash)
        self.assertIsNotNone(tt_entry)

    def test_iterative_deepening(self):
        """Test iterative deepening search."""
        value, pv = self.minimax.iterative_deepening_search(
            self.env,
            self.eval_func,
            max_depth=3,
            k_best=2,
            transposition_table=self.tt
        )
        
        self.assertIsNotNone(value)
        self.assertIsInstance(pv, list)

    def test_leaf_collection(self):
        """Test batch leaf collection and processing."""
        window = SearchWindow(
            alpha=-1.0,
            beta=1.0,
            depth=2,
            collected_leaves=[],
            lock=threading.Lock()
        )
        
        # Add some leaves
        window.collected_leaves = [(self.env, [0])]
        
        # Process leaves
        self.minimax.process_collected_leaves(
            window,
            self.eval_func,
            self.tt
        )
        
        # Verify leaves were processed
        self.assertEqual(len(window.collected_leaves), 0)
        self.assertTrue(len(self.tt) > 0)

if __name__ == '__main__':
    unittest.main()