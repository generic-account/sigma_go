"""A much faster MCTS-Minimax hybrid implementation for AlphaZero.
Where we use Numpy arrays to store node statistics,
and create child nodes on demand.

It is based of Micheal Hu's implementation of AlphaZero

This implementation combines MCTS with minimax search to get the best of both approaches:
- MCTS for selective tree expansion and exploration 
- Minimax for tactical calculation and pruning
And includes various optimizations for speed and memory efficiency.

The hybrid approach works by:
Using MCTS to guide the high-level search and identify promising variations then
switching to minimax search to evaluate the most promising variations in detail. 
The minimax search is done in a depth-limited manner to avoid the exponential branching factor.
Lastly, the values are backpropagated up the MCTS tree.

The positions are evaluated from the current player's perspective.

For example, in a two-player zero-sum game:

        A           Black to move (root)
       / \
      B   C         White to move
     / \
    D   E           Black to move

When evaluating positions:
- Node A represents Black's turn to move
- Nodes B,C represent positions after White's moves
- Nodes D,E represent positions after Black's moves

The evaluation scores are always from the perspective of the player to move.
So when selecting the best child of node A:

1. If B has score 0.8 and C has score 0.3 (from White's perspective)
2. We negate these scores to get Black's perspective: -0.8 and -0.3
3. Black should choose C since max(-0.8, -0.3) = -0.3

This is implemented by negating child Q-values during selection:
    ucb_scores = -node.child_Q() + node.child_U()
"""

import copy
import collections
import math
import time
import numpy as np
import logging
from typing import Callable, Tuple, Mapping, Iterable, Any, Dict

from alpha_zero.core.transposition_table import TranspositionTable, NodeType
from alpha_zero.core.minimax import ParallelMinimax
from alpha_zero.envs.base import BoardGameEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more detailed output
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DummyNode(object):
    """A placeholder to make computation possible for the root node."""

    def __init__(self):
        self.parent = None
        self.child_W = collections.defaultdict(float)
        self.child_N = collections.defaultdict(float)

class Node:
    """Node in the MCTS search tree."""

    def __init__(
        self,
        to_play: int,
        num_actions: np.ndarray,
        move: int = None,
        parent: Any = None,
        depth: int = 0,  # Added to keep track of depth of the node in MCTS tree
    ) -> None:
        """
        Args:
            to_play: the id of the current player.
            num_actions: number of total actions, including illegal move.
            prior: a prior probability of the node for a specific action, could be empty in case of root node.
            move: the action associated with the prior probability.
            parent: the parent node, could be a `DummyNode` if this is the root node.
            depth: the depth of the node in the MCTS tree.
        """

        self.to_play = to_play
        self.move = move
        self.parent = parent
        self.num_actions = num_actions
        self.depth = depth
        self.is_expanded = False

        self.child_W = np.zeros(num_actions, dtype=np.float32)
        self.child_N = np.zeros(num_actions, dtype=np.float32)
        self.child_P = np.zeros(num_actions, dtype=np.float32)

        self.children: Mapping[int, Node] = {}

        # Number of virtual losses on this node, only used in 'parallel_uct_search'
        self.losses_applied = 0

    def child_U(self, c_puct_base: float, c_puct_init: float) -> np.ndarray:
        """Returns a 1D numpy.array contains prior score for all child."""
        pb_c = math.log((1 + self.N + c_puct_base) / c_puct_base) + c_puct_init
        return pb_c * self.child_P * (math.sqrt(self.N) / (1 + self.child_N))

    def child_Q(self) -> np.ndarray:
        """Returns a 1D numpy.array contains mean action value for all child."""
        # Avoid division by zero
        child_N = np.where(self.child_N > 0, self.child_N, 1)

        return self.child_W / child_N

    @property
    def N(self) -> float:
        """The number of visits for current node is stored at parent's level."""
        return self.parent.child_N[self.move]

    @N.setter
    def N(self, value) -> None:
        """The total number of visits for current node at parent's level."""
        self.parent.child_N[self.move] = value

    @property
    def W(self) -> float:
        """The total value for current node is stored at parent's level."""
        return self.parent.child_W[self.move]

    @W.setter
    def W(self, value: float) -> None:
        """The total value for current node is stored at parent's level."""
        self.parent.child_W[self.move] = value

    @property
    def Q(self) -> float:
        """Returns the mean action value Q(s, a)."""
        if self.parent.child_N[self.move] > 0:
            return self.parent.child_W[self.move] / self.parent.child_N[self.move]
        else:
            return 0.0

    @property
    def has_parent(self) -> bool:
        """Check if the node has a parent."""
        return isinstance(self.parent, Node)

def best_child(
    node: Node,
    legal_actions: np.ndarray,
    c_puct_base: float,
    c_puct_init: float,
    child_to_play: int,
) -> Node:
    """Returns best child node with maximum action value Q plus an upper confidence bound U.
    And creates the selected best child node if not already exists.

    Args:
        node: the current node in the search tree.
        legal_actions: a 1D bool numpy.array mask for all actions,
                where `1` represents legal move and `0` represents illegal move.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        child_to_play: the player id for children nodes.

    Returns:
        The best child node corresponding to the UCT score.

    Raises:
        ValueError:
            if the node instance itself is a leaf node.
    """
    if not node.is_expanded:
        raise ValueError('Expand leaf node first.')

    # The child Q value is evaluated from the opponent perspective. when we select the best child for node,
    # we want to do so from node.to_play's perspective, so we always switch the sign for node.child_Q values,
    # this is required since we're talking about two-player, zero-sum games.

    # This is hybrid because the minimax value is used to evaluate the node thereby affecting the overall
    # UCB score of the node. However, it is weighted when backupdating the node's child_Q statistics.
    hybrid_ucb_scores = -node.child_Q() + node.child_U(c_puct_base, c_puct_init)

    scores = np.where(legal_actions == 1, hybrid_ucb_scores, -9999)
    move = np.argmax(scores)

    assert legal_actions[move] == 1

    if move not in node.children:
        node.children[move] = Node(
            to_play=child_to_play, num_actions=node.num_actions, move=move, parent=node, depth=node.depth + 1
        )

    return node.children[move]


def expand(
    node: Node, 
    prior_prob: np.ndarray, 
    env_hash: int, 
    mcts_prior_map: Dict[Tuple[int, int], float]
) -> None:
    """Expand all actions, including illegal actions.

    Args:
        node: current leaf node in the search tree.
        prior_prob: 1D numpy.array contains prior probabilities of the state for all actions.
        env_hash: zobrist hash of the current board position
        mcts_prior_map: dictionary to store action priors for minimax search

    Raises:
        ValueError:
            if node instance already expanded.
            if input argument `prior` is not a valid 1D float numpy.array.
    """
    if node.is_expanded:
        raise RuntimeError('Node already expanded.')
    if (
        not isinstance(prior_prob, np.ndarray)
        or len(prior_prob.shape) != 1
        or prior_prob.dtype not in (np.float32, np.float64)
    ):
        raise ValueError(f'Expect `prior_prob` to be a 1D float numpy.array, got {prior_prob}')

    node.child_P = prior_prob
    node.is_expanded = True

    # Store priors for non-zero probability actions
    for action, p in enumerate(prior_prob):
        if p > 0:
            mcts_prior_map[(env_hash, action)] = float(p)  # Convert to float to ensure compatibility

def confidence_weighted_value(mcts_value: float, minimax_value: float) -> float:
    """Combine MCTS and Minimax values using a confidence-weighted approach.

    Args:
        mcts_value: the evaluation value evaluated from MCTS algorithm.
        minimax_value: the evaluation value evaluated from minimax algorithm.

    Returns:
        a float value represents the combined evaluation value.

    Raises:
        ValueError:
            if input argument `value` is not float data type.
    """
    if not isinstance(mcts_value, float) or not isinstance(minimax_value, float):
        raise ValueError("Both mcts_value and minimax_value must be floats.")

    # Need testing to determine the best value for max_use_minimax_depth
    # It determines the depth at which the minimax value is used exclusively
    diff = abs(mcts_value - minimax_value)

    if diff < 0.2:
        # Weighted towards MCTS
        alpha = 0.7
    else:
        # Weighted towards Minimax
        alpha = 0.3

    return alpha * mcts_value + (1 - alpha) * minimax_value

def backup(node: Node, mcts_value: float, minimax_value: float) -> None:
    """Update statistics of the node and all traversed parent nodes.

    Args:
        node: current leaf node in the search tree.
        mcts_value: the evaluation value evaluated from 'the mcts algorithm of the current player's perspective.
        minimax_value: the evaluation value evaluated from minimax algorithm of the current player's perspective.

    Raises:
        ValueError:
            if input argument `value` is not float data type.
    """

    if not isinstance(mcts_value, float) or not isinstance(minimax_value, float):
        raise ValueError("Both mcts_value and minimax_value must be floats.")

    # Need testing to determine the best weights for the confidence-weighted approach
    combined_value = confidence_weighted_value(mcts_value, minimax_value)

    while isinstance(node, Node):
        node.N += 1
        node.W += combined_value
        node = node.parent
        combined_value = -1 * combined_value


def add_dirichlet_noise(node: Node, legal_actions: np.ndarray, eps: float = 0.25, alpha: float = 0.03) -> None:
    """Add dirichlet noise to a given node.

    Args:
        node: the root node we want to add noise to.
        legal_actions: a 1D bool numpy.array mask for all actions,
            where `1` represents legal move and `0` represents illegal move.
        eps: epsilon constant to weight the priors vs. dirichlet noise.
        alpha: parameter of the dirichlet noise distribution.

    Raises:
        ValueError:
            if input argument `node` is not expanded.
            if input argument `eps` or `alpha` is not float type
                or not in the range of [0.0, 1.0].
    """

    if not isinstance(node, Node) or not node.is_expanded:
        raise ValueError('Expect `node` to be expanded')
    if not isinstance(eps, float) or not 0.0 <= eps <= 1.0:
        raise ValueError(f'Expect `eps` to be a float in the range [0.0, 1.0], got {eps}')
    if not isinstance(alpha, float) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f'Expect `alpha` to be a float in the range [0.0, 1.0], got {alpha}')

    alphas = np.ones_like(legal_actions) * alpha
    noise = legal_actions * np.random.dirichlet(alphas)

    node.child_P = node.child_P * (1 - eps) + noise * eps


def generate_search_policy(child_N: np.ndarray, temperature: float, legal_actions: np.ndarray) -> np.ndarray:
    """Returns a policy action probabilities after MCTS search,
    proportional to its exponentiated visit count.

    Args:
        child_N: the visit number of the children nodes from the root node of the search tree.
        temperature: a parameter controls the level of exploration.
        legal_actions: a 1D bool numpy.array mask for all actions,
            where `1` represents legal move and `0` represents illegal move.

    Returns:
        a 1D numpy.array contains the action probabilities after MCTS search.

    Raises:
        ValueError:
            if input argument `temperature` is not float type or not in range (0.0, 1.0].
    """
    if not isinstance(temperature, float) or not 0 < temperature <= 1.0:
        raise ValueError(f'Expect `temperature` to be float type in the range (0.0, 1.0], got {temperature}')

    child_N = legal_actions * child_N

    if temperature > 0.0:
        # Simple hack to avoid overflow when call np.power over large numbers
        exp = max(1.0, min(5.0, 1.0 / temperature))
        child_N = np.power(child_N, exp)

    assert np.all(child_N >= 0) and not np.any(np.isnan(child_N))
    pi_probs = child_N
    sums = np.sum(pi_probs)
    if sums > 0:
        pi_probs /= sums

    return pi_probs

def add_virtual_loss(node: Node) -> None:
    """Propagate a virtual loss to the traversed path.

    Args:
        node: current leaf node in the search tree.
    """
    # This is a loss for both players in the traversed path,
    # since we want to avoid multiple threads to select the same path.
    # However since we'll be switching the sign for child_Q when selecting the best child,
    # here we use +1 instead of -1.
    vloss = +1
    while isinstance(node, Node):
        node.losses_applied += 1
        node.W += vloss
        node = node.parent


def revert_virtual_loss(node: Node) -> None:
    """Undo virtual loss to the traversed path.

    Args:
        node: current leaf node in the search tree.
    """

    vloss = -1
    while isinstance(node, Node):
        if node.losses_applied > 0:
            node.losses_applied -= 1
            node.W += vloss
        node = node.parent

def hybrid_uct_search(
    env: BoardGameEnv,
    eval_func: Callable[[np.ndarray, bool], Tuple[Iterable[np.ndarray], Iterable[float]]],
    root_node: Node,
    c_puct_base: float,
    c_puct_init: float,
    num_simulations: int,
    num_parallel: int,
    k_best: int,
    max_depth: int,
    root_noise: bool = False,
    warm_up: bool = False,
    deterministic: bool = False,
    num_minimax_threads: int = 4,
    minimax_time_limit: float = 30.0,
    max_minimax_leaves: int = 3,  # NEW: Only run Minimax on top 20 leaves, for instance
) -> Tuple[int, np.ndarray, float, float, Node]:
    """
    Hybrid search combining MCTS with parallel minimax, but selectively applying Minimax
    only to a few top-ranked leaves in each MCTS batch.
    """
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f'Expect `env` to be a valid BoardGameEnv instance, got {env}')
    if not 1 <= num_simulations:
        raise ValueError(f'Expect `num_simulations` to be a positive integer, got {num_simulations}')
    if env.is_game_over():
        raise RuntimeError('Game is over.')

    start_time = time.perf_counter()

    # Initialize parallel minimax searcher
    parallel_minimax = ParallelMinimax(
        num_threads=num_minimax_threads,
        min_batch_size=16,
        max_batch_size=128,
        virtual_loss=0.1,
        time_limit=minimax_time_limit
    )

     # Initialize transposition table
    transposition_table = TranspositionTable()

    # Initialize MCTS map
    mcts_prior_map: Dict[Tuple[int, int], float] = {}

    # Create root node if needed
    if root_node is None:
        prior_prob, value = eval_func(env.observation(), False)
        root_node = Node(to_play=env.to_play, num_actions=env.action_dim, parent=DummyNode())
        expand(root_node, prior_prob, env.hash(), mcts_prior_map)  # The expand call might differ in your code
        backup(root_node, value, value)

    assert root_node.to_play == env.to_play
    root_legal_actions = env.legal_actions

    # Add Dirichlet noise if requested (for exploration at the root)
    if root_noise:
        add_dirichlet_noise(root_node, root_legal_actions)

   
    # ----------------------------
    # Main MCTS Loop
    # ----------------------------
    while root_node.N < num_simulations + num_parallel:
        leaves = []
        failsafe = 0

        # 1) SELECTION: Collect up to num_parallel leaves
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            node = root_node

            # Copy the environment for simulation
            sim_env = copy.deepcopy(env)
            obs = sim_env.observation()
            done = sim_env.is_game_over()

            # Follow best_child() down until leaf or terminal
            while node.is_expanded and not done:
                node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init, sim_env.opponent_player)
                obs, reward, done, _ = sim_env.step(node.move)

            assert node.to_play == sim_env.to_play

            # If terminal, directly back up final reward
            if done:
                # The sign is reversed because from node.to_play's perspective, reward is the opponent's result
                backup(node, -reward, -reward)
                continue

            add_virtual_loss(node)
            leaves.append((node, obs))

        # 2) EVALUATION (Neural Net + Selective Minimax)
        if leaves:
            # Evaluate all leaves via the neural net (policy + value)
            batched_nodes, batched_obs = map(list, zip(*leaves))
            prior_probs, mcts_values = eval_func(np.stack(batched_obs, axis=0), True)

            # ---- NEW: Rank leaves, choose top few for Minimax ----
            leaf_indices = list(range(len(batched_nodes)))
            leaf_indices.sort(
                key=lambda i: (
                    abs(mcts_values[i])              # Magnitude of MCTS value
                    + 0.2 * (1.0 - batched_nodes[i].depth / 30)  # Example depth bonus
                ),
                reverse=True
            )
            top_leaf_indices = leaf_indices[:max_minimax_leaves]

            # Build envs only for top Minimax leaves
            minimax_envs = []
            for idx in top_leaf_indices:
                node_i = batched_nodes[idx]
                # Re-simulate from root to that leaf
                sim_env = copy.deepcopy(env)
                replay_path = []
                cur = node_i
                while cur.has_parent:
                    replay_path.append(cur.move)
                    cur = cur.parent
                for move in reversed(replay_path):
                    sim_env.step(move)

                minimax_envs.append((idx, sim_env))  # (index, environment)

            # Run parallel Minimax on these top leaves
            minimax_results = {}
            for (idx, sim_env) in minimax_envs:
                val, _ = parallel_minimax.iterative_deepening_search(
                    sim_env,
                    eval_func,
                    max_depth,
                    k_best,
                    transposition_table,
                    mcts_prior_map
                )
                minimax_results[idx] = val

            # 3) BACKUP: Combine MCTS + Minimax for each leaf
            for i, (leaf_node, prior_prob, mcts_val) in enumerate(zip(batched_nodes, prior_probs, mcts_values)):
                revert_virtual_loss(leaf_node)

                # Expand if not expanded
                if not leaf_node.is_expanded:
                    expand(leaf_node, prior_prob, env.hash(), mcts_prior_map)  # adapt if needed

                if i in minimax_results:
                    # For top leaves, use the Minimax result
                    backup(leaf_node, mcts_val, minimax_results[i])
                else:
                    # For non-top leaves, skip Minimax and back up MCTS only (or do an approximate approach)
                    backup(leaf_node, mcts_val, mcts_val)

    # -----------------------------------------
    # Move Selection after MCTS completes
    # -----------------------------------------
    search_pi = generate_search_policy(
        root_node.child_N,
        1.0 if warm_up else 0.1,
        root_legal_actions
    )

    move = None
    next_root_node = None
    best_child_Q = 0.0

    # Deterministic vs. probabilistic selection
    if deterministic:
        move = np.argmax(root_node.child_N)
    else:
        while move is None or (warm_up and env.has_pass_move and move == env.pass_move) or root_legal_actions[move] != 1:
            move = np.random.choice(np.arange(search_pi.shape[0]), p=search_pi)

    # If chosen move already in children, preserve that subtree
    if move in root_node.children:
        next_root_node = root_node.children[move]
        N, W = copy.copy(next_root_node.N), copy.copy(next_root_node.W)
        next_root_node.parent = DummyNode()
        next_root_node.move = None
        next_root_node.N = N
        next_root_node.W = W
        best_child_Q = -next_root_node.Q

    assert root_legal_actions[move] == 1

    end_time = time.perf_counter()
    logger.info(f"Total search time: {end_time - start_time:.2f}s")

    return move, search_pi, root_node.Q, best_child_Q, next_root_node