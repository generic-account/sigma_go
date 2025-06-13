# Copyright (c) 2023 Michael Hu. This code is part of the book "The Art of Reinforcement Learning: Fundamentals,
# Mathematics, and Implementation with Python.". This project is released under the MIT License. See the accompanying
# LICENSE file for details.


"""A much faster MCTS-Minimax hybrid implementation for AlphaZero.
Where we use Numpy arrays to store node statistics,
and create child nodes on demand.
"""

import copy
import collections
import math
import time
import numpy as np
import logging
from typing import Callable, Tuple, Mapping, Iterable, Any
from enum import Enum
import yaml


from alpha_zero.envs.base import BoardGameEnv
from alpha_zero.core.triggers import TriggerController
from alpha_zero.core.minimax import AlphaBetaEngine

logger = logging.getLogger(__name__)


class DummyNode(object):
    """A placeholder to make computation possible for the root node."""

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
    """Node in the MCTS search tree."""

    def __init__(
        self,
        to_play: int,
        num_actions: np.ndarray,
        move: int = None,
        parent: Any = None,
        depth: int = 0,
    ) -> None:
        self.to_play = to_play
        self.move = move
        self.parent = parent
        self.num_actions = num_actions
        self.depth = depth
        self.is_expanded = False

        self.child_W = np.zeros(num_actions, dtype=np.float32)
        self.child_N = np.zeros(num_actions, dtype=np.float32)
        self.child_P = np.zeros(num_actions, dtype=np.float32)
        
        self.child_W_p = np.zeros(num_actions, dtype=np.float32)
        self.child_N_p = np.zeros(num_actions, dtype=np.float32)

        self.children: Mapping[int, Node] = {}
        self.losses_applied = 0

        self.minimax_eval = None
        self.has_solver_proof = False
        self.is_terminal_proof = False
        self.uncertainty_metrics = {}

        self._W = 0.0
        self._N = 0

    def child_U(self, c_puct_base: float, c_puct_init: float) -> np.ndarray:
        pb_c = math.log((1 + self.N + c_puct_base) / c_puct_base) + c_puct_init
        return pb_c * self.child_P * (math.sqrt(self.N) / (1 + self.child_N))

    def child_Q(self):
        child_N = np.where(self.child_N > 0, self.child_N, 1)
        return self.child_W / child_N

    @property
    def child_losses_applied(self) -> np.ndarray:
        return np.array([self.children[a].losses_applied if a in self.children else 0 for a in range(self.num_actions)], dtype=np.int32)

    @property
    def N(self):
        if self.parent is None:
            return self._N
        if self.move is None:
            return sum(self.child_N)
        return self.parent.child_N[self.move]

    @N.setter
    def N(self, value):
        if self.parent is None:
            self._N = value
        elif self.move is not None:
            self.parent.child_N[self.move] = value

    @property
    def W(self):
        if self.parent is None:
            return self._W
        return self.parent.child_W[self.move]

    @W.setter
    def W(self, value):
        if self.parent is None:
            self._W = value
        elif self.move is not None:
            self.parent.child_W[self.move] = value

    @property
    def Q(self):
        if self.N > 0:
            return self.W / self.N
        return 0.0

    @property
    def has_parent(self) -> bool:
        return isinstance(self.parent, Node)

def best_child(
    node: Node,
    legal_actions: np.ndarray,
    c_puct_base: float,
    c_puct_init: float,
) -> Node:
    if not node.is_expanded:
        raise ValueError('Expand leaf node first.')

    child_Q = node.child_Q()
    child_U = node.child_U(c_puct_base, c_puct_init)
    
    ucb_scores = -child_Q - node.child_losses_applied + child_U
    ucb_scores = np.where(legal_actions, ucb_scores, -9999.0)
    move = np.argmax(ucb_scores)

    assert legal_actions[move]

    if move not in node.children:
        node.children[move] = Node(
            to_play=1 - node.to_play,
            num_actions=node.num_actions,
            move=move,
            parent=node,
            depth=node.depth + 1,
        )
    return node.children[move]

def expand(node: Node, prior_prob: np.ndarray) -> None:
    if node.is_expanded:
        raise RuntimeError('Node already expanded.')
    node.child_P = prior_prob
    node.is_expanded = True

def backup(node: Node, mcts_value: float, minimax_value: float, backprop_config: dict) -> None:
    """Update statistics of the node and all traversed parent nodes using advanced backpropagation."""
    max_use_minimax_depth = 10
    weight = max(0.0, min(1.0, node.depth / max_use_minimax_depth))
    value = mcts_value * (1 - weight) + minimax_value * weight

    current_node = node
    while current_node is not None:
        if current_node.parent is not None:
            parent = current_node.parent
            move = current_node.move
            
            value_for_parent = -value

            if current_node.is_terminal_proof:
                value_for_parent = -minimax_value
            else:
                delta = backprop_config.get('implicit_minimax_delta', 0.9)
                min_visits = backprop_config.get('implicit_minimax_min_visits', 10)
                
                if parent.N >= min_visits:
                    child_Q = parent.child_Q()
                    valid_children = parent.child_N > 0
                    if np.count_nonzero(valid_children) > 1:
                        mean_q = np.mean(child_Q[valid_children])
                        if np.max(np.abs(child_Q[valid_children] - mean_q)) > delta:
                            best_child_idx = np.argmax(np.abs(child_Q))
                            value_for_parent = -child_Q[best_child_idx]

            parent.child_N[move] += 1
            parent.child_W[move] += value_for_parent
            
            p_schedule = backprop_config.get('power_p_schedule', [[0, 1.0]])
            p = p_schedule[-1][1]
            for visit_threshold, p_value in reversed(p_schedule):
                if parent.N >= visit_threshold:
                    p = p_value
                    break
            
            power_val = np.clip(value, -1.0, 1.0)
            parent.child_W_p[move] += np.sign(power_val) * (np.abs(power_val) ** p)
            parent.child_N_p[move] += 1
        else: # Root node
            current_node.N += 1
            current_node.W += value

        value = -value
        current_node = current_node.parent

def add_dirichlet_noise(node: Node, legal_actions: np.ndarray, eps: float = 0.25, alpha: float = 0.03) -> None:
    alphas = np.ones_like(legal_actions) * alpha
    noise = legal_actions * np.random.dirichlet(alphas)
    node.child_P = node.child_P * (1 - eps) + noise * eps

def generate_search_policy(child_N: np.ndarray, temperature: float, legal_actions: np.ndarray) -> np.ndarray:
    if temperature > 0.0:
        exp = max(1.0, min(5.0, 1.0 / temperature))
        child_N = np.power(child_N, exp)
    pi_probs = legal_actions * child_N
    sums = np.sum(pi_probs)
    if sums > 0:
        pi_probs /= sums
    return pi_probs

def log_timing(start_time, message):
    elapsed = time.perf_counter() - start_time
    logger.info(f'{message}: {elapsed:.2f} seconds')

def uct_search(
    env: BoardGameEnv,
    eval_func: Callable[[np.ndarray, bool], Tuple[Iterable[np.ndarray], Iterable[float]]],
    root_node: Node,
    num_simulations: int,
    c_puct_base: float,
    c_puct_init: float,
    root_noise: bool = False,
    warm_up: bool = False,
    deterministic: bool = False,
    use_minimax: bool = False,
    k_best: int = 1,
    depth: int = 1,
    trigger_config_path: str = 'config/trigger_config.yaml',
) -> Tuple[int, np.ndarray, float, float, Node]:
    start_time = time.perf_counter()

    try:
        with open(trigger_config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Trigger config file not found at {trigger_config_path}. Using default values.")
        config = {}

    trigger_controller = TriggerController(config)
    alpha_beta_engine = AlphaBetaEngine(eval_func)
    backprop_config = config.get('backprop_config', {})

    if root_node is None:
        root_node = Node(to_play=env.to_play, num_actions=env.action_space.n, parent=None)

    if not root_node.is_expanded:
        prior_prob, value = eval_func(env.observation(), False)
        expand(root_node, prior_prob)
        backup(root_node, value, value, backprop_config)

    legal_actions = env.legal_actions
    
    dirichlet_alpha = 0.03
    dirichlet_eps = 0.25
    if root_noise:
        add_dirichlet_noise(root_node, legal_actions, dirichlet_eps, dirichlet_alpha)
    
    trigger_controller.reset()

    for _ in range(num_simulations):
        sim_env = env.clone()
        node = root_node
        last_move = None
        rollout_len = 0

        while node.is_expanded:
            node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init)
            sim_env.step(node.move)
            last_move = node.move
            rollout_len += 1
            if sim_env.is_game_over():
                break

        if sim_env.is_game_over():
            result = sim_env.get_result()
            backup(node, -result, -result, backprop_config)
            continue

        prior_prob, leaf_value = eval_func(sim_env.observation(), False)
        expand(node, prior_prob)
        minimax_value = leaf_value

        if use_minimax and rollout_len < 4:
            minimax_value = alpha_beta_engine.search(sim_env, depth=2, k_best=8)

        if use_minimax and trigger_controller.should_minimax(node, sim_env, last_move):
            search_depth = 4 if trigger_controller.is_tactical else 2
            search_k_best = 4 if trigger_controller.is_tactical else 8
            minimax_value = alpha_beta_engine.search(sim_env, depth=search_depth, k_best=search_k_best)
            node.minimax_eval = minimax_value

        remaining_empty = np.sum(sim_env.board == -1) if hasattr(sim_env, 'board') else 0
        if use_minimax and hasattr(sim_env, 'board_size') and sim_env.board_size == 9 and (remaining_empty <= 12 or node.depth >= 8):
            is_proven, proven_value = alpha_beta_engine.solver_search(sim_env)
            if is_proven:
                node.is_terminal_proof = True
                minimax_value = proven_value

        backup(node, leaf_value, minimax_value, backprop_config)

    temperature = 1.0 if warm_up else 0.1
    search_pi = generate_search_policy(root_node.child_N, temperature, legal_actions)

    move = None
    next_root_node = None
    best_child_Q = 0.0

    if deterministic:
        move = np.argmax(root_node.child_N)
    else:
        while move is None or (warm_up and env.has_pass_move and move == env.pass_move) or legal_actions[move] != 1:
            move = np.random.choice(np.arange(search_pi.shape[0]), p=search_pi)

    next_root_node = root_node.children[move]
    next_root_node.parent = None
    best_child_Q = -next_root_node.Q

    log_timing(start_time, "UCT search")
    return (move, search_pi, root_node.Q, best_child_Q, next_root_node)

def add_virtual_loss(node: Node) -> None:
    """Propagate a virtual loss to the traversed path."""
    vloss = +1
    while node is not None:
        node.losses_applied += 1
        node.W += vloss
        node = node.parent


def revert_virtual_loss(node: Node) -> None:
    """Undo virtual loss to the traversed path."""
    vloss = -1
    while node is not None:
        if node.losses_applied > 0:
            node.losses_applied -= 1
            node.W += vloss
        node = node.parent


def parallel_uct_search(
    env: BoardGameEnv,
    eval_func: Callable[[np.ndarray, bool], Tuple[Iterable[np.ndarray], Iterable[float]]],
    root_node: Node,
    num_simulations: int,
    num_parallel: int,
    c_puct_base: float,
    c_puct_init: float,
    root_noise: bool = False,
    warm_up: bool = False,
    deterministic: bool = False,
    use_minimax: bool = False,
    k_best: int = 1,
    depth: int = 1,
    trigger_config_path: str = 'config/trigger_config.yaml',
) -> Tuple[int, np.ndarray, float, float, Node]:
    start_time = time.perf_counter()

    try:
        with open(trigger_config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Trigger config file not found at {trigger_config_path}. Using default values.")
        config = {}

    trigger_controller = TriggerController(config)
    alpha_beta_engine = AlphaBetaEngine(eval_func)
    backprop_config = config.get('backprop_config', {})

    if root_node is None:
        root_node = Node(to_play=env.to_play, num_actions=env.action_space.n, parent=None)

    if not root_node.is_expanded:
        prior_prob, value = eval_func(env.observation(), False)
        expand(root_node, prior_prob)
        backup(root_node, value, value, backprop_config)

    legal_actions = env.legal_actions
    
    dirichlet_alpha = 0.03
    dirichlet_eps = 0.25

    if root_noise:
        add_dirichlet_noise(root_node, legal_actions, dirichlet_eps, dirichlet_alpha)
    
    trigger_controller.reset()
    
    for _ in range(num_simulations // num_parallel):
        leaves = []
        failsafe = 0

        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            sim_env = env.clone()
            node = root_node
            
            while node.is_expanded:
                node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init)
                sim_env.step(node.move)
                if sim_env.is_game_over():
                    break
            
            if sim_env.is_game_over():
                result = sim_env.get_result()
                backup(node, -result, -result, backprop_config)
            else:
                add_virtual_loss(node)
                leaves.append((node, sim_env))

        if leaves:
            observations = np.array([l[1].observation() for l in leaves])
            all_prior_probs, all_leaf_values = eval_func(observations, True)

            for i, (node, sim_env) in enumerate(leaves):
                revert_virtual_loss(node)
                prior_prob, leaf_value = all_prior_probs[i], all_leaf_values[i]
                expand(node, prior_prob)
                
                minimax_value = leaf_value
                
                if use_minimax:
                    if node.depth < 4:
                         minimax_value = alpha_beta_engine.search(sim_env, depth=2, k_best=8)

                    if trigger_controller.should_minimax(node, sim_env, node.move):
                        search_depth = 4 if trigger_controller.is_tactical else 2
                        search_k_best = 4 if trigger_controller.is_tactical else 8
                        minimax_value = alpha_beta_engine.search(sim_env, depth=search_depth, k_best=search_k_best)
                        node.minimax_eval = minimax_value
                
                backup(node, leaf_value, minimax_value, backprop_config)

    temperature = 1.0 if warm_up else 0.1
    search_pi = generate_search_policy(root_node.child_N, temperature, legal_actions)

    move = None
    next_root_node = None
    best_child_Q = 0.0

    if deterministic:
        move = np.argmax(root_node.child_N)
    else:
        while move is None or (warm_up and env.has_pass_move and move == env.pass_move) or legal_actions[move] != 1:
            move = np.random.choice(np.arange(search_pi.shape[0]), p=search_pi)

    next_root_node = root_node.children[move]
    next_root_node.parent = None
    best_child_Q = -next_root_node.Q

    log_timing(start_time, "Parallel UCT search")
    return (move, search_pi, root_node.Q, best_child_Q, next_root_node)
