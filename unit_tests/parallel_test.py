import unittest
from unittest.mock import Mock, MagicMock
import numpy as np
import torch

from alpha_zero.core.mcts_v2 import Node, parallel_uct_search
from alpha_zero.envs.go import GoEnv

class TestParallelMCTS(unittest.TestCase):
    def setUp(self):
        self.komi = 7.5
        self.env = GoEnv(komi=self.komi)
        
        # Mock eval_func to return fixed policy and value
        self.eval_func = MagicMock()
        
        def mock_eval(obs, batched):
            if batched:
                batch_size = obs.shape[0]
                policies = np.random.rand(batch_size, self.env.action_space.n).astype(np.float32)
                policies /= policies.sum(axis=1, keepdims=True)
                values = np.random.uniform(-1, 1, size=batch_size).tolist()
                return policies, values
            else:
                policy = np.random.rand(self.env.action_space.n).astype(np.float32)
                policy /= policy.sum()
                value = np.random.uniform(-1, 1)
                return policy, value
        
        self.eval_func.side_effect = mock_eval

    def test_no_node_expanded_twice_in_parallel(self):
        """
        This test ensures that in a parallel MCTS search, a node is not selected
        for expansion by multiple threads simultaneously, which would cause a
        'Node already expanded' RuntimeError.
        """
        root_node = Node(to_play=self.env.to_play, num_actions=self.env.action_space.n, parent=None)
        
        # We will monkey-patch the expand function to track which nodes have been expanded.
        expanded_nodes = set()
        original_expand = __import__('alpha_zero.core.mcts_v2').core.mcts_v2.expand
        
        def tracking_expand(node, prior_prob):
            node_id = id(node)
            if node_id in expanded_nodes:
                raise RuntimeError(f"Node {node_id} already expanded.")
            expanded_nodes.add(node_id)
            original_expand(node, prior_prob)

        # Monkey-patch the expand function
        __import__('alpha_zero.core.mcts_v2').core.mcts_v2.expand = tracking_expand

        try:
            # Run parallel search with a high number of simulations and parallelism
            # to increase the chance of a race condition.
            parallel_uct_search(
                env=self.env,
                eval_func=self.eval_func,
                root_node=root_node,
                num_simulations=20,
                num_parallel=4,
                c_puct_base=19652,
                c_puct_init=1.25,
                root_noise=True,
                warm_up=False,
                deterministic=False,
                use_minimax=False,
            )
        finally:
            # Restore the original expand function
            __import__('alpha_zero.core.mcts_v2').core.mcts_v2.expand = original_expand

if __name__ == '__main__':
    unittest.main()