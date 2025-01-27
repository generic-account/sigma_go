"""
A much faster MCTS-Minimax hybrid implementation for AlphaZero.
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
    level=logging.INFO,  # Change to logging.DEBUG for even more details
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
        depth: int = 0,  # track depth of this node in MCTS tree
    ) -> None:
        """
        Args:
            to_play: the id of the current player.
            num_actions: number of total actions, including illegal move.
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

        self.children: Mapping[int, 'Node'] = {}

        # For parallel MCTS: how many virtual losses have been applied on this node
        self.losses_applied = 0

    @property
    def N(self) -> float:
        """The number of visits for current node is stored at parent's level."""
        return self.parent.child_N[self.move]

    @N.setter
    def N(self, value) -> None:
        """Set the total number of visits for current node at parent's level."""
        self.parent.child_N[self.move] = value

    @property
    def W(self) -> float:
        """The total value for current node is stored at parent's level."""
        return self.parent.child_W[self.move]

    @W.setter
    def W(self, value: float) -> None:
        """Set the total value for current node at parent's level."""
        self.parent.child_W[self.move] = value

    @property
    def Q(self) -> float:
        """Mean action value Q(s, a) for this node."""
        if self.parent.child_N[self.move] > 0:
            return self.parent.child_W[self.move] / self.parent.child_N[self.move]
        else:
            return 0.0

    @property
    def has_parent(self) -> bool:
        """Check if the node has a parent (i.e. is not the root)."""
        return isinstance(self.parent, Node)

    def child_U(self, c_puct_base: float, c_puct_init: float) -> np.ndarray:
        """Compute the U = c_puct * P * sqrt(sum(N)) / (1 + N_a) term for each child."""
        pb_c = math.log((1 + self.N + c_puct_base) / c_puct_base) + c_puct_init
        return pb_c * self.child_P * (math.sqrt(self.N) / (1 + self.child_N))

    def child_Q(self) -> np.ndarray:
        """Compute Q for each child as W_a / N_a."""
        child_N = np.where(self.child_N > 0, self.child_N, 1)
        return self.child_W / child_N


def best_child(
    node: Node,
    legal_actions: np.ndarray,
    c_puct_base: float,
    c_puct_init: float,
    child_to_play: int,
) -> Node:
    """Selects the child with the maximum UCB score.

    Args:
        node: the current node in the search tree.
        legal_actions: a 1D bool numpy.array for which moves are legal.
        c_puct_base: for the UCB exploration term.
        c_puct_init: for the UCB exploration term.
        child_to_play: the next player to move in these child nodes.

    Returns:
        The best child node (creating it if needed).
    """
    if not node.is_expanded:
        raise ValueError('Expand leaf node first before calling best_child.')

    # Q is from the child's perspective. We switch sign for the current node's perspective:
    hybrid_ucb_scores = -node.child_Q() + node.child_U(c_puct_base, c_puct_init)

    scores = np.where(legal_actions == 1, hybrid_ucb_scores, -9999)
    move = np.argmax(scores)
    assert legal_actions[move] == 1

    if move not in node.children:
        logger.info(f"best_child: Creating child node for move={move} at depth={node.depth+1}.")
        node.children[move] = Node(
            to_play=child_to_play,
            num_actions=node.num_actions,
            move=move,
            parent=node,
            depth=node.depth + 1
        )

    logger.info(
        f"best_child: Node depth={node.depth}, picking move={move} with UCB score={scores[move]:.4f}"
    )

    return node.children[move]


def expand(
    node: Node,
    prior_prob: np.ndarray,
    env_hash: int,
    mcts_prior_map: Dict[Tuple[int, int], float]
) -> None:
    """Expand a leaf node: assign child prior probabilities and mark is_expanded=True."""
    if node.is_expanded:
        raise RuntimeError('Node is already expanded.')

    if (
        not isinstance(prior_prob, np.ndarray)
        or len(prior_prob.shape) != 1
        or prior_prob.dtype not in (np.float32, np.float64)
    ):
        raise ValueError("prior_prob must be a 1D float array.")

    node.child_P = prior_prob
    node.is_expanded = True

    # Store priors in mcts_prior_map for Minimax ordering
    for action, p in enumerate(prior_prob):
        if p > 0:
            mcts_prior_map[(env_hash, action)] = float(p)

    logger.info(
        f"expand: Expanded node at depth={node.depth}, assigned priors for {len(prior_prob)} actions."
    )


def confidence_weighted_value(mcts_value: float, minimax_value: float) -> float:
    """Combine MCTS vs. Minimax values via a confidence-based approach."""
    diff = abs(mcts_value - minimax_value)

    # Example simple approach:
    if diff < 0.2:
        # Weighted more towards MCTS if they're close
        alpha = 0.7
    else:
        # Weighted more towards Minimax if they disagree significantly
        alpha = 0.3

    combined = alpha * mcts_value + (1 - alpha) * minimax_value
    logger.info(
        f"confidence_weighted_value: MCTS={mcts_value:.3f}, Minimax={minimax_value:.3f}, "
        f"Diff={diff:.3f}, alpha={alpha:.2f}, Combined={combined:.3f}"
    )
    return combined


def backup(node: Node, mcts_value: float, minimax_value: float) -> None:
    """Backpropagates results up the tree from a leaf to the root."""
    combined_value = confidence_weighted_value(mcts_value, minimax_value)
    original_combined = combined_value

    logger.info(
        f"backup: Starting from leaf at depth={node.depth}, MCTS={mcts_value:.3f}, "
        f"Minimax={minimax_value:.3f}, Combined={original_combined:.3f}"
    )

    while isinstance(node, Node):
        node.N += 1
        node.W += combined_value
        logger.info(
            f"backup: Node depth={node.depth}, updated N={node.N}, W={node.W:.3f} "
            f"(combined_value={combined_value:.3f})"
        )
        node = node.parent
        combined_value = -combined_value  # flip sign for the parent


def add_dirichlet_noise(node: Node, legal_actions: np.ndarray, eps: float = 0.25, alpha: float = 0.03) -> None:
    """Adds Dirichlet noise to the root node's prior probabilities for exploration."""
    if not isinstance(node, Node) or not node.is_expanded:
        raise ValueError('Expect `node` to be an expanded Node.')
    if not isinstance(eps, float) or not 0.0 <= eps <= 1.0:
        raise ValueError(f'eps must be in [0,1], got {eps}')
    if not isinstance(alpha, float) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f'alpha must be in [0,1], got {alpha}')

    alphas = np.ones_like(legal_actions) * alpha
    noise = legal_actions * np.random.dirichlet(alphas)

    node.child_P = node.child_P * (1 - eps) + noise * eps

    logger.info("add_dirichlet_noise: Applied Dirichlet noise to root node's prior probabilities.")


def generate_search_policy(child_N: np.ndarray, temperature: float, legal_actions: np.ndarray) -> np.ndarray:
    """Convert visit counts to a probability distribution for move selection."""
    if not isinstance(temperature, float) or not (0 < temperature <= 1.0):
        raise ValueError("temperature must be in (0,1].")

    visits = child_N.copy()
    visits = legal_actions * visits
    if temperature > 0.0:
        exp = max(1.0, min(5.0, 1.0 / temperature))
        visits = np.power(visits, exp)

    sums = np.sum(visits)
    if sums > 0:
        visits /= sums

    logger.info("generate_search_policy: Generated policy distribution from child visit counts.")
    return visits


def add_virtual_loss(node: Node) -> None:
    """Add a virtual loss to discourage multiple threads exploring the same path."""
    vloss = +1
    while isinstance(node, Node):
        node.losses_applied += 1
        node.W += vloss
        node = node.parent


def revert_virtual_loss(node: Node) -> None:
    """Undo a previously-applied virtual loss."""
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
    num_minimax_threads: int = 4,
    minimax_time_limit: float = 30.0,
    max_minimax_leaves: int = 3,  # Only run Minimax on top X leaves
    root_noise: bool = False,
    warm_up: bool = False,
    deterministic: bool = False,
) -> Tuple[int, np.ndarray, float, float, Node]:
    """Hybrid MCTS–Minimax search, with selective Minimax application and confidence weighting."""
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f"Expect `env` to be a valid BoardGameEnv instance, got {env}")
    if env.is_game_over():
        raise RuntimeError("Game is already over.")

    logger.info("hybrid_uct_search: Starting hybrid MCTS-Minimax search.")
    start_time = time.perf_counter()

    # Initialize parallel minimax searcher
    parallel_minimax = ParallelMinimax(
        num_threads=num_minimax_threads,
        min_batch_size=16,
        max_batch_size=128,
        virtual_loss=0.1,
        time_limit=minimax_time_limit
    )

    # Create or load a transposition table
    transposition_table = TranspositionTable()

    # A map of (zobrist_hash, action) -> prior for better move ordering in Minimax
    mcts_prior_map: Dict[Tuple[int, int], float] = {}

    # If we have no root node, build one
    if root_node is None:
        logger.info("hybrid_uct_search: Creating new root node.")
        prior_prob, init_value = eval_func(env.observation(), False)
        root_node = Node(
            to_play=env.to_play,
            num_actions=env.action_dim,
            parent=DummyNode()
        )
        expand(root_node, prior_prob, env.hash(), mcts_prior_map)
        backup(root_node, init_value, init_value)

    assert root_node.to_play == env.to_play

    # Optionally add Dirichlet noise at the root for exploration
    root_legal_actions = env.legal_actions
    if root_noise:
        add_dirichlet_noise(root_node, root_legal_actions)
        logger.info("hybrid_uct_search: Added Dirichlet noise to root node's prior.")

    # ----------------------------
    # Main MCTS Loop
    # ----------------------------
    logger.info(f"hybrid_uct_search: Starting main MCTS loop with up to {num_simulations} simulations.")
    while root_node.N < num_simulations + num_parallel:
        leaves = []
        failsafe = 0

        # Collect up to num_parallel leaves for batch processing
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            node = root_node
            sim_env = copy.deepcopy(env)
            done = sim_env.is_game_over()

            # DESCENT: Follow best_child down until an unexpanded node or terminal
            while node.is_expanded and not done:
                node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init, sim_env.opponent_player)
                _, reward, done, _ = sim_env.step(node.move)

            if done:
                logger.info(f"MCTS selection: reached terminal state, reward={reward:.3f}.")
                backup(node, -reward, -reward)
                continue

            # Not terminal, so we have a leaf
            add_virtual_loss(node)
            leaves.append((node, sim_env.observation()))

        # EVALUATION PHASE
        if leaves:
            # Evaluate all leaves in one batch with the neural net
            batched_nodes, batched_obs = map(list, zip(*leaves))
            prior_probs, mcts_values = eval_func(np.stack(batched_obs, axis=0), True)

            logger.info(f"hybrid_uct_search: Collected {len(leaves)} leaves, evaluating with NN, then Minimax on top {max_minimax_leaves}.")

            # Sort leaves by some priority measure (example: absolute MCTS value, plus a small depth bonus)
            leaf_indices = list(range(len(batched_nodes)))
            leaf_indices.sort(
                key=lambda i: (
                    abs(mcts_values[i])
                    + 0.2 * (1.0 - batched_nodes[i].depth / 30)  # example depth factor
                ),
                reverse=True
            )
            top_leaf_indices = leaf_indices[:max_minimax_leaves]

            # Build environments for top leaves
            minimax_envs = []
            for idx in top_leaf_indices:
                node_i = batched_nodes[idx]
                sim_env = copy.deepcopy(env)

                # Replay moves from root to node_i
                path = []
                cur = node_i
                while cur.has_parent:
                    path.append(cur.move)
                    cur = cur.parent
                for move in reversed(path):
                    sim_env.step(move)

                minimax_envs.append((idx, sim_env))

            # Run Minimax on top leaves
            minimax_results = {}
            for (idx, sim_env) in minimax_envs:
                val, _ = parallel_minimax.iterative_deepening_search(
                    sim_env,
                    eval_func,
                    max_depth,
                    k_best,
                    transposition_table,
                    mcts_prior_map  # pass the MCTS priors
                )
                minimax_results[idx] = val
                logger.info(f"Minimax: Leaf index={idx}, depth={batched_nodes[idx].depth}, Minimax value={val:.3f}")

            # BACKUP for each leaf
            for i, (leaf_node, prior_prob, mcts_val) in enumerate(zip(batched_nodes, prior_probs, mcts_values)):
                revert_virtual_loss(leaf_node)

                # Expand if not expanded
                if not leaf_node.is_expanded:
                    expand(leaf_node, prior_prob, env.hash(), mcts_prior_map)

                # Combine MCTS & Minimax or fallback if not in top
                if i in minimax_results:
                    backup(leaf_node, mcts_val, minimax_results[i])
                else:
                    # If we didn't run Minimax, back up MCTS alone or do a simpler fallback
                    backup(leaf_node, mcts_val, mcts_val)

    # -----------------------------------------
    # Move Selection
    # -----------------------------------------
    logger.info("hybrid_uct_search: MCTS complete, selecting move from root_node.")
    search_pi = generate_search_policy(
        root_node.child_N,
        1.0 if warm_up else 0.1,
        root_legal_actions
    )

    move = None
    next_root_node = None
    best_child_Q = 0.0

    if deterministic:
        move = np.argmax(root_node.child_N)
        logger.info(f"Move Selection: Deterministic, chose move={move} with max visits.")
    else:
        # Sample from the search distribution
        while move is None or (warm_up and env.has_pass_move and move == env.pass_move) or root_legal_actions[move] != 1:
            move = np.random.choice(np.arange(search_pi.shape[0]), p=search_pi)
        logger.info(f"Move Selection: Sampled move={move} from search_pi distribution.")

    if move in root_node.children:
        next_root_node = root_node.children[move]
        # Keep stats for the new root
        N, W = copy.copy(next_root_node.N), copy.copy(next_root_node.W)
        next_root_node.parent = DummyNode()
        next_root_node.move = None
        next_root_node.N = N
        next_root_node.W = W
        best_child_Q = -next_root_node.Q

    end_time = time.perf_counter()
    logger.info(f"hybrid_uct_search: Completed in {end_time - start_time:.2f}s. Final chosen move={move}.")

    return move, search_pi, root_node.Q, best_child_Q, next_root_node
