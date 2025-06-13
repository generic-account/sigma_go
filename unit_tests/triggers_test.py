# Copyright (c) 2023 Michael Hu. This code is part of the book "The Art of Reinforcement Learning: Fundamentals,
# Mathematics, and Implementation with Python.". This project is released under the MIT License.
# See the accompanying LICENSE file for details.


import unittest
import numpy as np
from unittest.mock import MagicMock, PropertyMock

from alpha_zero.core.triggers import TriggerController
from alpha_zero.core.mcts_v2 import Node


class TestTriggerController(unittest.TestCase):
    def setUp(self):
        """Set up a default trigger controller and mock objects for each test."""
        self.config = {
            'policy_entropy_threshold': 1.0,
            'value_variance_threshold': 0.1,
            'visit_count_threshold': 50,
            'winrate_spread_threshold': 0.3,
            'max_minimax_calls_per_move': 3,
        }
        self.trigger_controller = TriggerController(self.config)
        
        # Mock Node
        self.mock_node = MagicMock(spec=Node)
        
        # Mock BoardGameEnv
        self.mock_board_state = MagicMock()

    def test_policy_entropy_trigger_fires(self):
        """Test that the trigger fires when policy entropy exceeds the threshold."""
        # High entropy policy (e.g., uniform distribution over a few moves)
        high_entropy_policy = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0])
        self.mock_node.child_P = high_entropy_policy
        self.mock_node.N = 10 # Visit count below threshold

        should_fire = self.trigger_controller.should_minimax(self.mock_node, self.mock_board_state, None)
        self.assertTrue(should_fire, "Trigger should fire for high policy entropy.")
        self.assertEqual(self.trigger_controller.minimax_calls_this_move, 1)

    def test_policy_entropy_trigger_not_fires(self):
        """Test that the trigger does not fire for low policy entropy."""
        # Low entropy policy (e.g., confident in one move)
        low_entropy_policy = np.array([0.9, 0.05, 0.05, 0.0, 0.0, 0.0])
        self.mock_node.child_P = low_entropy_policy
        self.mock_node.N = 10 # Visit count below threshold
        
        # Mock child_Q to avoid firing visit count heuristic
        type(self.mock_node).child_Q = MagicMock(return_value=np.array([0.1, 0.1]))
        type(self.mock_node).child_N = PropertyMock(return_value=np.array([1, 1]))


        should_fire = self.trigger_controller.should_minimax(self.mock_node, self.mock_board_state, None)
        self.assertFalse(should_fire, "Trigger should not fire for low policy entropy.")

    def test_visit_count_heuristic_trigger_fires(self):
        """Test that the visit-count heuristic fires when thresholds are met."""
        self.mock_node.child_P = np.array([0.1, 0.1, 0.8]) # Low entropy
        self.mock_node.N = 60 # Visit count above threshold
        
        # Mock child Q-values to have a high spread
        child_q_values = np.array([0.9, 0.2, 0.85])
        child_n_values = np.array([20, 20, 20])
        
        # The mock setup for child_Q and child_N needs to be done carefully
        # Since they are methods/properties that get called inside the function.
        self.mock_node.child_Q.return_value = child_q_values
        self.mock_node.child_N = child_n_values

        should_fire = self.trigger_controller.should_minimax(self.mock_node, self.mock_board_state, None)
        self.assertTrue(should_fire, "Visit-count heuristic should fire with high visit count and win-rate spread.")
        self.assertEqual(self.trigger_controller.minimax_calls_this_move, 1)

    def test_visit_count_heuristic_not_fires_low_spread(self):
        """Test heuristic does not fire when win-rate spread is low."""
        self.mock_node.child_P = np.array([0.1, 0.1, 0.8]) # Low entropy
        self.mock_node.N = 60 # Visit count above threshold
        
        # Low spread Q-values
        self.mock_node.child_Q.return_value = np.array([0.5, 0.45, 0.55])
        self.mock_node.child_N = np.array([20, 20, 20])

        should_fire = self.trigger_controller.should_minimax(self.mock_node, self.mock_board_state, None)
        self.assertFalse(should_fire, "Visit-count heuristic should not fire with low win-rate spread.")

    def test_max_calls_limit(self):
        """Test that the trigger respects the max_minimax_calls_per_move limit."""
        self.mock_node.child_P = np.array([0.25, 0.25, 0.25, 0.25]) # High entropy
        self.mock_node.N = 10

        # Fire trigger up to the limit
        for i in range(self.config['max_minimax_calls_per_move']):
            should_fire = self.trigger_controller.should_minimax(self.mock_node, self.mock_board_state, None)
            self.assertTrue(should_fire, f"Trigger should fire on call {i+1}.")
        
        self.assertEqual(self.trigger_controller.minimax_calls_this_move, self.config['max_minimax_calls_per_move'])

        # Next call should not fire
        should_fire = self.trigger_controller.should_minimax(self.mock_node, self.mock_board_state, None)
        self.assertFalse(should_fire, "Trigger should not fire after reaching max call limit.")

    def test_reset_minimax_calls(self):
        """Test that the reset method resets the call counter."""
        self.trigger_controller.minimax_calls_this_move = 5
        self.trigger_controller.reset()
        self.assertEqual(self.trigger_controller.minimax_calls_this_move, 0)

if __name__ == '__main__':
    unittest.main()