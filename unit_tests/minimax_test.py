from alpha_zero.envs.base import BoardGameEnv
import numpy as np
import copy


def test_minimax():
    class TestEnv(BoardGameEnv):
        def __init__(self):
            self.state = 0
            self.depth = 0
            self.action_dim = 2
            self.legal_actions = np.ones(2, dtype=np.int32)

        def step(self, action):
            print(f"Before step: state={self.state}, depth={self.depth}, action={action}")
            self.state = self.state * 2 + action
            self.depth += 1
            print(f"After step: state={self.state}, depth={self.depth}")
            return None, 0, self.depth >= 3, {}

        def is_game_over(self):
            return self.depth >= 3

        def observation(self):
            return np.array([self.state])

    def eval_func(obs, _):
        state = obs[0]
        value = state / 7  # Normalize to [0, 1]
        print(f"Evaluating state: {state}, calculated value: {value}")
        return np.array([0.5, 0.5]), value

    env = TestEnv()

    def debug_minimax(env, eval_func, depth, alpha=-float('inf'), beta=float('inf'), maximizing_player=True):
        print(f"\nEntering minimax: depth={depth}, maximizing={maximizing_player}")
        print(f"Current state: {env.state}, depth: {env.depth}")

        if depth == 0 or env.is_game_over():
            obs = env.observation()
            _, value = eval_func(obs, False)
            print(f"Leaf node: depth={depth}, state={obs[0]}, value={value}")
            return value

        if maximizing_player:
            max_eval = -float('inf')
            for action in np.where(env.legal_actions == 1)[0]:
                sim_env = copy.deepcopy(env)
                sim_env.step(action)
                eval = debug_minimax(sim_env, eval_func, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                print(f"Max node: depth={depth}, action={action}, value={eval}, max_eval={max_eval}")
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for action in np.where(env.legal_actions == 1)[0]:
                sim_env = copy.deepcopy(env)
                sim_env.step(action)
                eval = debug_minimax(sim_env, eval_func, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                print(f"Min node: depth={depth}, action={action}, value={eval}, min_eval={min_eval}")
                if beta <= alpha:
                    break
            return min_eval

    result = debug_minimax(env, eval_func, 3)

    expected_result = 5 / 7

    print(f"\nMinimax result: {result}")
    print(f"Expected result: {expected_result}")
    print(f"Difference: {abs(result - expected_result)}")

    assert abs(result - expected_result) < 1e-6, "Minimax is not working as expected"


test_minimax()
