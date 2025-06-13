import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from alpha_zero.core.minimax import AlphaBetaEngine
from alpha_zero.envs.base import BoardGameEnv

class TestAlphaBetaEngine(unittest.TestCase):

    def setUp(self):
        """Set up a mock environment and a mock evaluation function for testing."""
        self.mock_env = MagicMock(spec=BoardGameEnv)
        self.mock_env.action_dim = 9
        self.mock_env.zobrist_hash.return_value = 0
        self.mock_env.is_game_over.return_value = False
        
        # Configure a default mock for the cloned environment
        self.cloned_env = MagicMock(spec=BoardGameEnv)
        self.cloned_env.is_game_over.return_value = False
        self.cloned_env.zobrist_hash.return_value = 1
        self.cloned_env.legal_actions = np.ones(9)  # Add legal_actions to the cloned mock
        self.mock_env.clone.return_value = self.cloned_env

        # Mock evaluation function
        # It returns a fixed policy and a value that can be controlled.
        self.mock_eval_func = MagicMock()
        self.mock_eval_func.return_value = (np.ones(9) / 9, 0.5)

        self.engine = AlphaBetaEngine(self.mock_eval_func)

    def test_basic_search(self):
        """Test that the search runs and returns a value without errors."""
        self.mock_env.legal_actions = np.ones(9)
        
        value = self.engine.search(self.mock_env, depth=1)
        
        self.assertIsInstance(value, float)
        self.mock_eval_func.assert_called()

    def test_neural_move_ordering(self):
        """Test that moves are explored in the order of their policy probabilities."""
        # Setup policy probabilities where one move is clearly better
        policy_probs = np.array([0.1, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.mock_eval_func.return_value = (policy_probs, 0.5)
        
        self.mock_env.legal_actions = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])

        # We need to track the order of calls to step
        call_order = []
        def step_tracker(action):
            call_order.append(action)
        
        self.cloned_env.step.side_effect = step_tracker

        # The search should prioritize move 2
        self.engine.search(self.mock_env, depth=1, k_best=3)
        
        # Check that the first move explored is the one with the highest policy
        self.assertEqual(call_order[0], 2)

    def test_transposition_table_usage(self):
        """Test that the transposition table is used to cache and retrieve results."""
        self.mock_env.legal_actions = np.ones(9)
        self.mock_env.zobrist_hash.return_value = 12345

        # First search should populate the table
        self.engine.search(self.mock_env, depth=2)
        nodes_first_search = self.engine.nodes_searched
        
        # Reset the mock call count for the eval function
        self.mock_eval_func.reset_mock()

        # Second search on the same state should hit the cache
        self.engine.search(self.mock_env, depth=2)
        
        # The evaluation function should not be called for the root state on the second search
        # It will be called for children, but not the root. A full hit would mean 0 calls.
        # Since we are mocking the hash, we expect a hit at the root.
        self.assertEqual(self.engine.transposition_table.lookup(12345)[0], 2) # Check depth
        
if __name__ == '__main__':
    unittest.main()