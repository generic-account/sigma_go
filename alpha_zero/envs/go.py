# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Go env class."""
from typing import Tuple, Mapping, Text, List
import re
from copy import copy
import numpy as np

from alpha_zero.envs.base import BoardGameEnv
from alpha_zero.envs import go_engine as go
from alpha_zero.utils import sgf_wrapper
from alpha_zero.utils.util import get_time_stamp


class GoEnv(BoardGameEnv):
    """Gym environment for board game Go.

    The main logic and rules are implemented in the go_engine.py module, which is copied from
    Google's Minigo project: https://github.com/tensorflow/minigo

    Please note that the go_engine.py module implements a simplified version of the Tromp-Taylor scoring system,
    which is based on the rules described in the Tromp-Taylor Rules (https://senseis.xmp.net/?TrompTaylorRules) or a similar variation of those rules.

    The Tromp-Taylor rules state that both players are required to capture the stones they believe to be dead before passing.
    Failure to capture these stones will result in them being scored as if they were alive.
    Additionally, empty regions bordering stones of both colors are considered nobody's territory, just like they would be in a seki situation.

    However, it's important to be aware that this implementation cannot handle dead stones before counting the areas.
    Because accurately detecting dead stones at the end of a game is a complex task that often requires the use of additional techniques
    such as using simulation to play more moves, or using neural networks to predict the score.
    Consequently, there is a possibility of incorrect scores for certain games.
    """

    metadata = {'render.modes': ['terminal'], 'players': ['black', 'white']}

    def __init__(
        self,
        komi: float = 7.5,
        num_stack: int = 8,
        max_steps: int = go.N * go.N * 2,
    ) -> None:
        """Initialize the Go environment.

        Args:
            komi: Compensation points for the second player (white), default 7.5.
            num_stack: Number of previous states to stack, default 8.
                The final state is an image containing N x 2 + 1 binary planes.
            max_steps: Maximum number of steps per game, default N x N x 2.
        """

        super().__init__(
            id='Go',
            board_size=go.N,
            num_stack=num_stack,
            black_player_id=go.BLACK,
            white_player_id=go.WHITE,
            has_pass_move=True,
            has_resign_move=True,
        )

        self.komi = komi
        self.max_steps = max_steps

        self.position = go.Position(komi=self.komi)

        self.board = self.position.board
        self.legal_actions = self.position.all_legal_moves()

    def reset(self, **kwargs) -> np.ndarray:
        """Reset the game to its initial state.

        Returns:
            np.ndarray: The initial observation of the game.
        """
        super().reset(**kwargs)

        self.position = go.Position(komi=self.komi)

        self.board = self.position.board
        self.legal_actions = self.position.all_legal_moves()

        return self.observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the game.

        Args:
            action: The action to take.

        Returns:
            A tuple containing:
            - observation (np.ndarray): The current state of the game.
            - reward (float): The reward for the action taken.
            - done (bool): Whether the game has ended.
            - info (dict): Additional information about the game state.

        Raises:
            RuntimeError: If the game is already over.
            ValueError: If the action is invalid or illegal.
        """
        if self.is_game_over():
            raise RuntimeError('Game is over, call reset before using step method.')
        if action is not None and action != self.resign_move and not 0 <= int(action) <= self.action_space.n - 1:
            raise ValueError(f'Invalid action. The action {action} is out of bound.')
        if action is not None and action != self.resign_move and self.legal_actions[int(action)] != 1:
            raise ValueError(f'Illegal action {action}.')

        self.last_move = copy(int(action))
        self.last_player = copy(self.to_play)
        self.steps += 1

        # Resign is not recorded as a move in sgf, instead it's recorded as game result.
        # so no need to add to history
        if action == self.resign_move:
            self.position = self.position.flip_playerturn(mutate=True)
            self.board = self.position.board

            # Make sure the latest board position is always at index 0
            self.board_deltas.appendleft(np.copy(self.board))

            # After game ended, no move should be allowed.
            self.legal_actions = np.zeros(self.action_dim, dtype=np.int8)

            # Resign is always a loss for the player, no need to evaluate the board for score
            self.winner = self.black_player if self.last_player == self.white_player else self.white_player

            # Switch next player
            self.to_play = self.position.to_play

            return self.observation(), -1, True, {}

        # All other moves, including pass move need to be recorded into to history to make a valid sgf record
        self.add_to_history(self.last_player, self.last_move)

        reward = 0.0
        done = False

        # Make a move on the go.Position, this will also handle pass move
        self.position = self.position.play_move(c=self.cc.from_flat(action), color=self.to_play, mutate=True)
        self.board = self.position.board
        self.legal_actions = self.position.all_legal_moves()

        # Make sure the latest board position is always at index 0
        self.board_deltas.appendleft(np.copy(self.board))

        done = self.is_game_over()

        # Rewards are zero except for the final terminal state.
        # And the reward is for the last player, not `to_play` player,
        # which follows our standard MDP practice, where reward function R_t = R(s_t, a_t)
        if done:
            # After game ended, no move should be allowed.
            self.legal_actions = np.zeros(self.action_dim, dtype=np.int8)

            result_str = self.get_result_string()
            if re.match(r'B\+', result_str, re.IGNORECASE):
                self.winner = self.black_player
            elif re.match(r'W\+', result_str, re.IGNORECASE):
                self.winner = self.white_player
            else:
                self.winner = None

            if self.winner is not None:
                if self.last_player == self.winner:
                    reward = 1.0
                else:
                    reward = -1.0

        # Switch next player
        self.to_play = self.position.to_play

        return self.observation(), reward, done, {}

    def render_additional_header(self, outfile, black_stone, white_stone):
        """Render additional header information for the game.

        Args:
            outfile: The output file to write to.
            black_stone: The symbol for black stones.
            white_stone: The symbol for white stones.
        """
        caps = self.get_captures()
        outfile.write(f'{black_stone} captures: {caps[self.black_player]}, {white_stone} captures: {caps[self.white_player]} ')
        outfile.write('\n')
        outfile.write('\n')

    def get_captures(self) -> Mapping[Text, int]:
        """Get the number of captures for each player.

        Returns:
            A dictionary mapping player names to their capture counts.
        """
        return {
            self.black_player: self.position.caps[0],
            self.white_player: self.position.caps[1],
        }

    def is_game_over(self) -> bool:
        """Check if the game is over.

        The game is over if one of the following is true:
            * Someone resigned
            * Reached maximum steps
            * Both players passed

        Returns:
            bool: True if the game is over, False otherwise.
        """
        if self.last_move == self.resign_move:
            return True
        if self.steps >= self.max_steps:
            return True
        # Game is over if two players played pass move in the last two consecutive turns
        if len(self.history) >= 2 and self.history[-1].move == self.pass_move and self.history[-2].move == self.pass_move:
            return True

        return False

    def get_result_string(self) -> str:
        """Get the result of the game as a string.

        Returns:
            str: The result string, e.g., 'B+R' for Black wins by resignation.
        """
        if self.last_move == self.resign_move:
            string = 'B+R' if self.winner == self.black_player else 'W+R'
        else:
            string = self.position.result_string()

        return string

    def to_sgf(self) -> str:
        """Convert the game to SGF (Smart Game Format) string.

        Returns:
            str: The SGF representation of the game.
        """
        return sgf_wrapper.make_sgf(
            board_size=self.board_size,
            move_history=self.history,
            result_string=self.get_result_string(),
            ruleset='Chinese',
            komi=self.komi,
            date=get_time_stamp(),
        )

    def assess_tactical_complexity(self) -> int:
        complexity_score = 0
        
        # 1. Basic board analysis
        num_stones = np.sum(self.board != 0)
        complexity_score += num_stones // 10  # More stones generally means more complexity
        
        # 2. Group analysis (modified)
        liberty_counts = self.position.get_liberties()
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] != 0:  # If there's a stone
                    num_liberties = liberty_counts[i, j]
                    if num_liberties == 1:  # Group in atari
                        complexity_score += 5
                    elif num_liberties <= 3:  # Group under pressure
                        complexity_score += 3
        
        # 3. Pattern recognition (simplified)
        if self._check_for_ko():
            complexity_score += 10
        
        # 4. Territory assessment (simplified)
        black_territory, white_territory = self._estimate_territory()
        contested_points = self.board_size**2 - (black_territory + white_territory)
        complexity_score += contested_points // 5
        
        # 5. Move urgency analysis (simplified)
        urgent_moves = self._find_urgent_moves()
        complexity_score += len(urgent_moves) * 3
        
        # 6. Endgame analysis
        if self._is_endgame():
            complexity_score -= 10  # Endgame is generally less tactically complex
        
        return min(100, max(0, complexity_score))  # Normalize to 0-100 range

    def _check_for_ko(self) -> bool:
        # Simplified ko check
        return self.position.ko is not None

    def _estimate_territory(self) -> Tuple[int, int]:
        # Very rough territory estimation
        return np.sum(self.board == go.BLACK), np.sum(self.board == go.WHITE)

    def _find_urgent_moves(self) -> List[Tuple[int, int]]:
        urgent_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:  # Empty point
                    if self._is_urgent_move((i, j)):
                        urgent_moves.append((i, j))
        return urgent_moves

    def _is_urgent_move(self, point: Tuple[int, int]) -> bool:
        # Check if playing at this point would save a group in atari
        # or capture an opponent's group
        for color in [go.BLACK, go.WHITE]:
            if self._would_capture(point, color):
                return True
            if self._would_escape_atari(point, color):
                return True
        return False

    def _would_capture(self, point: Tuple[int, int], color: int) -> bool:
        # Check if playing at this point would capture any opponent's group
        for n in go.NEIGHBORS[point]:
            if self.board[n] == -color and self.position.get_liberties()[n] == 1:
                return True
        return False

    def _would_escape_atari(self, point: Tuple[int, int], color: int) -> bool:
        # Check if playing at this point would escape atari for any neighboring group
        for n in go.NEIGHBORS[point]:
            if self.board[n] == color and self.position.get_liberties()[n] == 1:
                return True
        return False

    def _is_endgame(self) -> bool:
        # Simple endgame check: if more than 70% of the board is filled
        return np.sum(self.board != 0) > 0.7 * self.board_size**2
