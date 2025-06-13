# Copyright (c) 2023 Michael Hu. This code is part of the book "The Art of Reinforcement Learning: Fundamentals,
# Mathematics, and Implementation with Python.". This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Contains the logic for dynamically triggering minimax search within MCTS."""

import numpy as np
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from alpha_zero.core.mcts_v2 import Node
    from alpha_zero.envs.base import BoardGameEnv


class TriggerController:
    """
    A controller to decide when to trigger a full minimax search based on node statistics and board patterns.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the TriggerController with a given configuration.

        Args:
            config: A dictionary containing trigger thresholds and settings.
        """
        if config is None:
            config = {
                'policy_entropy_threshold': 1.5,
                'value_variance_threshold': 0.1,
                'visit_count_threshold': 50,
                'winrate_spread_threshold': 0.3,
                'max_minimax_calls_per_move': 64,
            }
        self.config = config
        self.minimax_calls_this_move = 0
        self.is_tactical = False

    def reset(self):
        """Resets the counter for minimax calls for a new move."""
        self.minimax_calls_this_move = 0
        self.is_tactical = False

    def should_minimax(self, node: 'Node', board_state: 'BoardGameEnv', last_move: Optional[int]) -> bool:
        """
        Main decision function to determine if minimax search should be triggered.

        Args:
            node: The MCTS node to evaluate.
            board_state: The current state of the board.
            last_move: The last move played to reach this state.

        Returns:
            True if minimax search should be triggered, False otherwise.
        """
        self.is_tactical = False  # Reset at the start of each check

        if self.minimax_calls_this_move >= self.config['max_minimax_calls_per_move']:
            return False

        # 1. Uncertainty Metrics
        policy_entropy = self.calculate_policy_entropy(node.child_P)
        if policy_entropy > self.config['policy_entropy_threshold']:
            self.minimax_calls_this_move += 1
            self.is_tactical = True
            return True

        # Placeholder for value variance
        # value_variance = self.calculate_value_variance(node)
        # if value_variance > self.config['value_variance_threshold']:
        #     self.minimax_calls_this_move += 1
        #     return True

        # 2. Visit-Count Heuristic
        if node.N > self.config['visit_count_threshold']:
            child_q_values = node.child_Q()[node.child_N > 0]
            if len(child_q_values) > 1:
                win_rate_spread = np.max(child_q_values) - np.min(child_q_values)
                if win_rate_spread > self.config['winrate_spread_threshold']:
                    self.minimax_calls_this_move += 1
                    self.is_tactical = True
                    return True

        # 3. Tactical Pattern Detectors (placeholder)
        # if self.detect_tactical_patterns(board_state, last_move):
        #     self.minimax_calls_this_move += 1
        #     return True

        return False

    def calculate_policy_entropy(self, policy_priors: np.ndarray) -> float:
        """
        Calculates the entropy of the policy priors to measure uncertainty.
        H(p) = -Sum(p_i * log(p_i))

        Args:
            policy_priors: A NumPy array of policy probabilities.

        Returns:
            The entropy of the policy distribution.
        """
        # Filter out zero probabilities to avoid log(0)
        non_zero_priors = policy_priors[policy_priors > 0]
        if non_zero_priors.size == 0:
            return 0.0
        return -np.sum(non_zero_priors * np.log(non_zero_priors))

    def calculate_value_variance(self, node: 'Node') -> float:
        """
        Calculates the variance of the Q-values of the children of a node.
        This is a placeholder for now.

        Args:
            node: The MCTS node whose children's Q-value variance is to be calculated.

        Returns:
            The variance of the children's Q-values.
        """
        # This requires tracking simulation results per child, which is not yet implemented.
        # For now, we'll return 0.0
        return 0.0

    def detect_tactical_patterns(self, board_state: 'BoardGameEnv', last_move: Optional[int]) -> bool:
        """
        Detects tactical patterns on the board that might warrant a deeper search.
        This is a placeholder for now.

        Args:
            board_state: The current state of the board.
            last_move: The last move played.

        Returns:
            True if a tactical pattern is detected, False otherwise.
        """
        # Placeholder for pattern detection logic (atari, ladder, snapback)
        return False