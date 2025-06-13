import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Mock the Node and DummyNode for testing purposes
class DummyNode:
    def __init__(self, num_actions=9):
        self.parent = None
        self.child_W = np.zeros(num_actions, dtype=np.float32)
        self.child_N = np.zeros(num_actions, dtype=np.float32)
        self.child_W_p = np.zeros(num_actions, dtype=np.float32)
        self.child_N_p = np.zeros(num_actions, dtype=np.float32)
        self.N = 0

    def child_Q(self):
        """Mock child_Q for the dummy node."""
        child_N = np.where(self.child_N > 0, self.child_N, 1)
        return self.child_W / child_N

class Node:
    def __init__(self, move=None, parent=None, num_actions=9):
        self.move = move
        self.parent = parent if parent is not None else DummyNode(num_actions)
        self.N = 0
        self.depth = 1
        self.is_terminal_proof = False
        self.num_actions = num_actions
        
        self.child_W = np.zeros(num_actions, dtype=np.float32)
        self.child_N = np.zeros(num_actions, dtype=np.float32)
        self.child_W_p = np.zeros(num_actions, dtype=np.float32)
        self.child_N_p = np.zeros(num_actions, dtype=np.float32)

    def child_Q(self):
        child_N = np.where(self.child_N > 0, self.child_N, 1)
        return self.child_W / child_N

# Import the function to be tested
from alpha_zero.core.mcts_v2 import backup

class TestAdvancedBackpropagation(unittest.TestCase):

    def test_power_mean_backup(self):
        """Test the power-mean backup logic."""
        root = DummyNode()
        node = Node(move=0, parent=root)
        
        backprop_config = {'power_p_schedule': [[0, 2.0]]} # p=2
        
        # Simulate two backups
        backup(node, 0.8, 0.8, backprop_config)
        backup(node, 0.4, 0.4, backprop_config)
        
        # The value propagated to the parent (root) is from the parent's perspective, so it's negated.
        final_w = root.child_W[0]
        self.assertAlmostEqual(final_w, -1.2) # -0.8 + -0.4
        
        final_w_p = root.child_W_p[0]
        self.assertAlmostEqual(final_w_p, 0.8**2 + 0.4**2)
        self.assertEqual(root.child_N_p[0], 2)


    def test_implicit_minimax_backup(self):
        """Test that implicit minimax overrides standard backup."""
        root = DummyNode()
        root.N = 15 # The parent must have enough visits
        node = Node(move=0, parent=root)
        
        # Setup the parent's children to have an outlier
        root.child_N[0] = 5
        root.child_W[0] = 0.1 * 5 # Q = 0.1
        
        root.child_N[1] = 5
        root.child_W[1] = 0.9 * 5 # Q = 0.9 -> outlier
        
        root.child_N[2] = 5
        root.child_W[2] = 0.1 * 5 # Q = 0.1
        
        backprop_config = {
            'implicit_minimax_delta': 0.5,
            'implicit_minimax_min_visits': 10
        }
        
        # Backup a new value from the first child (node).
        # This should trigger the implicit minimax logic at the parent (root).
        backup(node, 0.2, 0.2, backprop_config)
        
        # The update to W for move 0 should be based on the outlier Q (0.9),
        # not the backed-up value (0.2).
        # The value is from the parent's perspective, so it's -0.9.
        # The initial W was 0.5. The update is -0.9. Final W is 0.5 - 0.9 = -0.4.
        self.assertAlmostEqual(root.child_W[0], 0.5 - 0.9)


    def test_solver_proof_backup(self):
        """Test that a solver proof propagates directly."""
        root = DummyNode()
        node = Node(move=0, parent=root)
        node.is_terminal_proof = True
        
        backprop_config = {}
        
        # Backup a proven win
        backup(node, 0.1, 1.0, backprop_config)
        
        # The parent's W should be set such that Q = -1.0 (from parent's perspective)
        # W = Q * N = -1.0 * 1 = -1.0
        self.assertEqual(root.child_W[0], -1.0)

if __name__ == '__main__':
    unittest.main()