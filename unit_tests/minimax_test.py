import unittest
import numpy as np
from unittest.mock import Mock
import threading
from typing import Dict, Tuple
import copy

from alpha_zero.core.minimax import ParallelMinimax, SearchWindow
from alpha_zero.core.transposition_table import TranspositionTable, NodeType
from alpha_zero.envs.base import BoardGameEnv


class MockEnv(BoardGameEnv):
    """Mock environment for testing minimax."""
    def __init__(self, board_size=3, game_over=False, winner=None):
        self.board_size = board_size
        self.legal_actions = np.ones(board_size * board_size, dtype=np.int8)
        self.current_player = 1
        self._hash = 0
        self._game_over = game_over
        self._winner = winner
        self.board = np.zeros((board_size, board_size))
        
    def observation(self):
        # Simple 3 x board_size x board_size array
        return np.zeros((3, self.board_size, self.board_size))
        
    def step(self, action):
        if not 0 <= action < len(self.legal_actions):
            raise ValueError(f"Invalid action: {action}")
        if self.legal_actions[action] == 0:
            raise ValueError(f"Illegal action: {action}")
        
        # Combine old hash with new action
        self._hash = hash((self._hash, action))
        
        # Flip player
        self.current_player *= -1
        
        # Mark the action as taken
        self.legal_actions[action] = 0
        row, col = divmod(action, self.board_size)
        self.board[row, col] = self.current_player
        
    def clone(self):
        return copy.deepcopy(self)
        
    def zobrist_hash(self):
        return self._hash
        
    @property
    def to_play(self):
        return self.current_player == 1
        
    def is_game_over(self):
        return self._game_over


class TestParallelMinimax(unittest.TestCase):
    def setUp(self):
        # Instantiate the minimax object
        self.minimax = ParallelMinimax(
            num_threads=2,
            min_batch_size=4,
            max_batch_size=8,
            time_limit=1.0
        )
        self.env = MockEnv()
        self.tt = TranspositionTable()

        # Mock evaluation function that returns a vector of values if batch,
        # or a single value if not batch.
        def mock_eval(observations, is_batch):
            if is_batch:
                # observations.shape[0] is the batch size
                batch_size = observations.shape[0]
                # Return 0.5 for each position
                return None, np.full((batch_size,), 0.5)
            else:
                # Single position
                return None, np.array([0.5])
        
        # Use a side-effect or direct function assignment for the evaluator
        self.eval_func = Mock(side_effect=mock_eval)

    def test_dynamic_batch_size_boundaries(self):
        """Test batch size stays within bounds and scales correctly with depth."""
        test_depths = [(1, 4), (2, 4), (4, 4), (8, 8)]
        
        for depth, max_depth in test_depths:
            batch_size = self.minimax.get_dynamic_batch_size(depth, max_depth)
            self.assertGreaterEqual(batch_size, self.minimax.min_batch_size)
            self.assertLessEqual(batch_size, self.minimax.max_batch_size)
            
        # Test batch size decreases with depth
        size1 = self.minimax.get_dynamic_batch_size(1, 8)
        size2 = self.minimax.get_dynamic_batch_size(8, 8)
        self.assertGreater(size1, size2)

    def test_move_ordering_with_mcts_priors(self):
        """Test move ordering with various prior configurations."""
        env_hash = self.env.zobrist_hash()
        
        test_cases = [
            # Strong prior for first move
            {(env_hash, 0): 0.8, (env_hash, 1): 0.2},
            # Equal priors
            {(env_hash, 0): 0.5, (env_hash, 1): 0.5},
            # No priors (None)
            None,
            # Empty priors
            {}
        ]
        
        # Set move_scores to empty so MCTS prior is the main factor
        move_scores = {}
        
        for mcts_priors in test_cases:
            # Depth=1 so the best combined score is chosen at the root
            value, pv = self.minimax.alpha_beta_search_with_pv(
                self.env.clone(),
                self.eval_func,
                depth=1,
                k_best=1,
                alpha=-1.0,
                beta=1.0,
                transposition_table=self.tt,
                move_scores=move_scores,
                mcts_prior_map=mcts_priors
            )
            
            self.assertIsNotNone(value)
            self.assertIsInstance(pv, list)
            self.assertGreaterEqual(len(pv), 1)
            # If we have a strong prior for action=0, verify it's chosen
            if mcts_priors and (env_hash, 0) in mcts_priors:
                self.assertEqual(pv[0], 0)

    def test_parallel_search_thread_safety(self):
        """Test thread safety of parallel search."""
        results = []
        threads = []
        
        def search_thread():
            value, pv = self.minimax.parallel_minimax_search_with_pv(
                self.env.clone(),
                self.eval_func,
                depth=2,
                k_best=2,
                transposition_table=self.tt,
                move_scores={},
                mcts_prior_map=None
            )
            results.append((value, pv))
            
        # Run multiple searches in parallel
        for _ in range(3):
            thread = threading.Thread(target=search_thread)
            thread.start()
            threads.append(thread)
            
        for thread in threads:
            thread.join()
            
        # Verify all searches completed
        self.assertEqual(len(results), 3)
        for value, pv in results:
            self.assertIsNotNone(value)
            self.assertIsInstance(pv, list)

    def test_transposition_table_collisions(self):
        """Test TT behavior with hash collisions and overwrites."""
        small_tt = TranspositionTable(max_size=2)
        
        # Fill TT beyond capacity
        for i in range(4):
            env = MockEnv()
            env.step(i)  # This will generate different hashes
            
            self.minimax.alpha_beta_search_with_pv(
                env,
                self.eval_func,
                depth=2,
                k_best=2,
                alpha=-1.0,
                beta=1.0,
                transposition_table=small_tt,
                move_scores={},
                mcts_prior_map=None
            )
            
        # Verify TT size doesn't exceed max
        self.assertLessEqual(len(small_tt), 2)

    def test_leaf_collection_and_processing(self):
        """Test leaf collection with various batch sizes."""
        window = SearchWindow(
            alpha=-1.0,
            beta=1.0,
            depth=2,
            collected_leaves=[],
            lock=threading.Lock()
        )
        
        # Add multiple leaves
        num_leaves = 5
        for i in range(num_leaves):
            env = MockEnv()
            env.step(i)
            window.collected_leaves.append((env, [i]))
            
        initial_leaves = len(window.collected_leaves)
        self.assertEqual(initial_leaves, num_leaves)
        
        # Process leaves
        self.minimax.process_collected_leaves(
            window,
            self.eval_func,
            self.tt
        )
        
        # Verify processing
        self.assertEqual(len(window.collected_leaves), 0)
        self.assertGreaterEqual(len(self.tt), num_leaves)

    def test_game_over_conditions(self):
        """Test minimax behavior at game-over states."""
        game_over_env = MockEnv(game_over=True)
        
        value, pv = self.minimax.alpha_beta_search_with_pv(
            game_over_env,
            self.eval_func,
            depth=4,  # Should return immediately despite depth
            k_best=2,
            alpha=-1.0,
            beta=1.0,
            transposition_table=self.tt,
            move_scores={},
            mcts_prior_map=None
        )
        
        self.assertIsNotNone(value)
        self.assertEqual(len(pv), 0)  # No moves in PV at terminal state


if __name__ == '__main__':
    unittest.main()
