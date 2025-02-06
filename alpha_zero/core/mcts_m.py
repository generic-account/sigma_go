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
from alpha_zero.core.minimax import ParallelMinimax, minimax
from alpha_zero.envs.base import BoardGameEnv
from alpha_zero.envs import go_engine as go

# Configure logging
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
        
        # Track which moves have been considered for expansion
        self.considered_moves = set()
        
        # Parameters for progressive widening
        self.alpha = 0.25  # Controls growth rate of number of children
        self.C = 4  # Base number of children to consider
        
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

    def get_expansion_count(self) -> int:
        """Calculate number of moves that should be considered based on visit count."""
        if self.N == 0:
            return self.C
        return min(
            self.num_actions,
            max(self.C, int(self.C * np.power(self.N, self.alpha)))
        )

    def get_next_moves_to_expand(self, legal_actions: np.ndarray) -> np.ndarray:
        """Get the next set of moves to consider for expansion.
        
        Args:
            legal_actions: Boolean mask of legal moves
            
        Returns:
            Array of move indices to consider expanding
        """
        # Get number of moves we should consider at current visit count
        target_count = self.get_expansion_count()
        
        # If we've already considered enough moves, return empty array
        if len(self.considered_moves) >= target_count:
            return np.array([], dtype=np.int32)
            
        # Get scores for unexpanded moves
        unexpanded_moves = []
        unexpanded_scores = []
        
        for move in range(self.num_actions):
            if legal_actions[move] and move not in self.considered_moves:
                unexpanded_moves.append(move)
                # Score = prior probability + small random noise for tie-breaking
                unexpanded_scores.append(self.child_P[move] + np.random.uniform(0, 1e-6))
                
        if not unexpanded_moves:
            return np.array([], dtype=np.int32)
            
        # Sort by score and select top moves up to target_count
        moves_array = np.array(unexpanded_moves)
        scores_array = np.array(unexpanded_scores)
        sorted_indices = np.argsort(-scores_array)  # Descending order
        
        num_new = min(
            target_count - len(self.considered_moves),
            len(unexpanded_moves)
        )
        
        selected_moves = moves_array[sorted_indices[:num_new]]
        self.considered_moves.update(selected_moves)
        
        return selected_moves

    def child_U(self, c_puct_base: float, c_puct_init: float) -> np.ndarray:
        """Compute UCB score with variance-aware exploration bonus.
        
        This implementation uses:
        1. Running variance of Q-values to estimate uncertainty
        2. Visit-count based uncertainty
        3. Progressive widening to control exploration
        4. Uncertainty bonus that scales with tree depth
        """
        # Base PUCT formula component
        pb_c = math.log((1 + self.N + c_puct_base) / c_puct_base) + c_puct_init
        
        # Calculate empirical variance of Q-values
        # We use the squared difference from parent's Q-value as a proxy for variance
        parent_q = self.Q if self.has_parent else 0.0
        child_q = self.child_Q()
        value_var = np.square(child_q - parent_q)
        
        # Calculate visit-count based uncertainty
        # Less visited nodes have higher uncertainty
        visit_uncertainty = 1.0 / np.sqrt(1 + self.child_N)
        
        # Progressive widening factor
        # Reduces exploration as we get more visits
        prog_width = np.power(self.N + 1, -0.25)
        
        # Depth-based scaling of uncertainty
        # Deeper nodes get less uncertainty bonus
        depth_scale = math.exp(-self.depth / 20.0)
        
        # Combine different uncertainty measures
        total_uncertainty = (
            0.5 * value_var +  # Value variance component
            0.3 * visit_uncertainty +  # Visit count component
            0.2 * self.child_P  # Prior probability component
        )
        
        # Scale uncertainty by progressive widening and depth
        uncertainty_bonus = total_uncertainty * prog_width * depth_scale
        
        # Final UCB formula combines:
        # 1. Standard PUCT term
        # 2. Enhanced uncertainty bonus
        # 3. Mask for moves not yet considered (progressive widening)
        considered_mask = np.array([i in self.considered_moves for i in range(self.num_actions)], dtype=np.float32)
        return considered_mask * (
            pb_c * self.child_P * (math.sqrt(self.N) / (1 + self.child_N)) +  # Standard PUCT
            c_puct_init * uncertainty_bonus  # Uncertainty bonus
        )

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
    mcts_prior_map: Dict[Tuple[int, int], float],
    legal_actions: np.ndarray
) -> None:
    """Expand a leaf node with selective expansion using progressive widening.
    
    Args:
        node: Node to expand
        prior_prob: Prior probabilities for all moves
        env_hash: Hash of current environment state
        mcts_prior_map: Map of (hash, action) -> prior for move ordering
        legal_actions: Boolean mask of legal moves
    """
    if node.is_expanded:
        raise RuntimeError('Node is already expanded.')

    if (
        not isinstance(prior_prob, np.ndarray)
        or len(prior_prob.shape) != 1
        or prior_prob.dtype not in (np.float32, np.float64)
    ):
        raise ValueError("prior_prob must be a 1D float array.")

    # Store full prior probabilities
    node.child_P = prior_prob
    
    # Get initial set of moves to consider
    moves_to_expand = node.get_next_moves_to_expand(legal_actions)
    
    # Store priors in mcts_prior_map for Minimax ordering
    # Only store for moves we're actually considering
    for move in moves_to_expand:
        if prior_prob[move] > 0:
            mcts_prior_map[(env_hash, move)] = float(prior_prob[move])

    node.is_expanded = True

    logger.info(
        f"expand: Expanded node at depth={node.depth}, "
        f"considering {len(moves_to_expand)}/{len(prior_prob)} moves initially"
    )


def confidence_weighted_value(mcts_value: float, minimax_value: float) -> float:
    """Combine MCTS vs. Minimax values using adaptive sigmoid-based blending.
    
    The blending weight is determined by:
    1. The difference between MCTS and Minimax evaluations
    2. The absolute values of the evaluations (extreme values are more reliable)
    3. A sigmoid function to smoothly transition between weights
    4. The relative confidence of each evaluation method
    
    Args:
        mcts_value: Value from MCTS evaluation (-1 to 1)
        minimax_value: Value from Minimax search (-1 to 1)
        
    Returns:
        Weighted combination of the two values
    """
    # Calculate absolute difference between evaluations
    diff = abs(mcts_value - minimax_value)
    
    # Calculate confidence factors based on absolute values
    # Values closer to -1 or 1 are considered more reliable
    mcts_conf = 1.0 / (1.0 + np.exp(-5 * (abs(mcts_value) - 0.5)))  # Sigmoid centered at 0.5
    minimax_conf = 1.0 / (1.0 + np.exp(-5 * (abs(minimax_value) - 0.5)))
    
    # Base alpha starts at 0.5 and adjusted by confidence difference
    base_alpha = 0.5 + 0.3 * (mcts_conf - minimax_conf)
    
    # Agreement factor - how much the evaluations agree/disagree
    # Centered at diff=0.3, steeper slope for faster transition
    agreement_factor = 1.0 / (1.0 + np.exp(8 * (diff - 0.3)))
    
    # Final alpha combines base confidence with agreement
    # When evaluations agree (high agreement_factor), use confidence-based weighting
    # When they disagree (low agreement_factor), bias towards the more confident evaluation
    alpha = base_alpha * agreement_factor + (0.5 + 0.3 * np.sign(mcts_conf - minimax_conf)) * (1 - agreement_factor)
    
    # Ensure alpha stays in [0.2, 0.8] range to maintain influence from both sources
    alpha = min(0.8, max(0.2, alpha))
    
    return alpha * mcts_value + (1 - alpha) * minimax_value


def compute_power_mean(values: np.ndarray, p: float = 2.0) -> float:
    """Compute the power mean (generalized mean) of a set of values.
    
    The power mean with exponent p is defined as:
    M_p(x) = (1/n * sum(x_i^p))^(1/p)
    
    Special cases:
    p = 1: arithmetic mean
    p = 2: quadratic mean
    p → ∞: maximum
    p → -∞: minimum
    
    Args:
        values: Array of values to compute mean over
        p: Power parameter (default 2.0 for quadratic mean)
        
    Returns:
        Power mean value
    """
    if len(values) == 0:
        return 0.0
    
    # Handle extreme p values for numerical stability
    if p > 100:  # Approximate max
        return np.max(values)
    elif p < -100:  # Approximate min
        return np.min(values)
        
    # Standard power mean calculation
    return np.power(np.mean(np.power(np.abs(values), p)), 1.0/p) * np.sign(np.mean(values))


def get_implicit_minimax_value(node: Node, to_play: int) -> float:
    """Calculate implicit minimax value for a node based on visit counts and Q-values.
    
    This implements the implicit minimax backup strategy from the paper:
    "Monte Carlo Tree Search with Implicit Minimax Backups"
    
    The key idea is to weight child values based on their visit counts,
    but use power means to approximate min/max operations.
    
    Args:
        node: The node to compute implicit minimax value for
        to_play: Current player (used to determine min vs max)
        
    Returns:
        Implicit minimax value for the node
    """
    if not node.is_expanded or node.N == 0:
        return node.Q
        
    # Get Q-values and visit counts for all children
    child_Q = node.child_Q()
    child_N = node.child_N
    
    # Filter to only visited children
    mask = child_N > 0
    if not np.any(mask):
        return node.Q
        
    values = child_Q[mask]
    visits = child_N[mask]
    
    # Weight values by visit counts
    weights = visits / np.sum(visits)
    weighted_values = values * weights
    
    # Use appropriate power mean based on player
    # Positive power for maximizing player, negative for minimizing
    p = 4.0 if to_play else -4.0
    
    return compute_power_mean(weighted_values, p)


def get_max_child_value(node: Node, to_play: int) -> float:
    """Get the maximum/minimum child value based on player perspective.
    
    For max player, returns maximum child value.
    For min player, returns minimum child value.
    
    Args:
        node: Current node
        to_play: Current player (used to determine max vs min)
        
    Returns:
        Maximum/minimum child Q-value
    """
    if not node.is_expanded or node.N == 0:
        return node.Q
        
    child_Q = node.child_Q()
    child_N = node.child_N
    
    # Only consider visited children
    mask = child_N > 0
    if not np.any(mask):
        return node.Q
        
    values = child_Q[mask]
    
    # For max player, return maximum value
    # For min player, return minimum value
    return np.max(values) if to_play else np.min(values)


def is_critical_position(
    node: Node,
    mcts_value: float,
    minimax_value: float,
    critical_threshold: float = 0.7
) -> bool:
    """Determine if a position is critical and warrants maximum backpropagation.
    
    A position is considered critical if:
    1. It has extreme evaluation (near win/loss)
    2. There's large disagreement between MCTS and minimax
    3. It shows sharp changes in evaluation
    4. It's part of a forcing sequence
    
    Args:
        node: Current node
        mcts_value: MCTS evaluation
        minimax_value: Minimax evaluation
        critical_threshold: Threshold for considering position critical
        
    Returns:
        Boolean indicating if position is critical
    """
    # Check for extreme evaluations
    if abs(mcts_value) > critical_threshold or abs(minimax_value) > critical_threshold:
        return True
        
    # Check for large evaluation disagreement
    if abs(mcts_value - minimax_value) > 0.5:
        return True
        
    # Check for sharp evaluation changes from parent
    if node.has_parent and node.parent.N > 0:
        parent_eval = node.parent.Q
        eval_change = abs(mcts_value - (-parent_eval))
        if eval_change > 0.5:
            return True
            
    # Check for forcing sequences (low branching factor)
    if node.is_expanded:
        legal_moves = np.sum(node.child_N > 0)
        if legal_moves <= 3:  # Small number of viable moves suggests forcing sequence
            return True
            
    return False


def backup(node: Node, mcts_value: float, minimax_value: float) -> None:
    """Enhanced backpropagation using maximum propagation for critical positions.
    
    This implementation combines:
    1. Traditional MCTS averaging
    2. Maximum backpropagation for critical positions
    3. Implicit minimax through power means
    4. Explicit minimax values from search
    
    The backup strategy adapts based on:
    - Position criticality (use max backup for critical positions)
    - Node depth (deeper nodes prefer max backup)
    - Evaluation extremity (extreme values propagate more directly)
    - Move forcing (forcing sequences use max backup)
    
    Args:
        node: Current node to backup from
        mcts_value: Value from MCTS evaluation
        minimax_value: Value from explicit minimax search
    """
    # Start with confidence-weighted combination of MCTS and explicit minimax
    explicit_combined = confidence_weighted_value(mcts_value, minimax_value)
    
    while isinstance(node, Node):
        # Get implicit minimax value through power means
        implicit_value = get_implicit_minimax_value(node, node.to_play)
        
        # Get maximum child value for critical positions
        max_value = get_max_child_value(node, node.to_play)
        
        # Determine if position is critical
        is_critical = is_critical_position(node, mcts_value, minimax_value)
        
        # Compute adaptive mixing factors
        visit_factor = min(1.0, node.N / 100.0)
        depth_factor = min(1.0, node.depth / 10.0)
        
        # Calculate value volatility
        volatility = 0.0
        if node.has_parent:
            sibling_values = []
            for child in node.parent.children.values():
                if child.N > 0:
                    sibling_values.append(child.Q)
            if sibling_values:
                volatility = np.std(sibling_values)
        
        # Critical positions get higher weight for max_value
        max_weight = 0.0
        if is_critical:
            # Base weight for critical positions
            max_weight = 0.4
            # Increase weight based on factors
            max_weight += 0.2 * depth_factor  # Deeper nodes
            max_weight += 0.2 * volatility    # Higher volatility
            max_weight += 0.2 * visit_factor  # More visited nodes
            max_weight = min(0.8, max_weight)
        
        # Remaining weight split between implicit and explicit values
        remaining_weight = 1.0 - max_weight
        implicit_ratio = 0.4 * visit_factor + 0.4 * volatility + 0.2 * depth_factor
        implicit_ratio = min(0.8, max(0.2, implicit_ratio))
        
        # Combine all three value sources
        combined_value = (
            max_weight * max_value +
            remaining_weight * (
                implicit_ratio * implicit_value +
                (1 - implicit_ratio) * explicit_combined
            )
        )
        
        # Update node statistics
        node.N += 1
        node.W += combined_value
        
        logger.info(
            f"backup: Node depth={node.depth}, N={node.N}, "
            f"explicit={explicit_combined:.3f}, implicit={implicit_value:.3f}, "
            f"max={max_value:.3f}, combined={combined_value:.3f} "
            f"(max_weight={max_weight:.2f}, is_critical={is_critical})"
        )
        
        # Prepare for parent node
        node = node.parent
        explicit_combined = -explicit_combined  # Flip sign for opponent's perspective


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


def log_timing(start_time, message, move=None):
    """Helper function to log timing information."""
    elapsed = time.perf_counter() - start_time
    if move is not None:
        logger.info(f"Move {move} completed in {elapsed:.2f} seconds")
    else:
        logger.info(f"{message}: {elapsed:.2f} seconds")


def is_tactical_position(
    node: Node,
    env: BoardGameEnv,
    mcts_value: float,
    visit_threshold: int = 50,
) -> bool:
    """Determine if a position is tactical and requires minimax analysis.
    
    A position is considered tactical if:
    1. It has low visit counts (indicating uncertainty)
    2. It has high value volatility among siblings
    3. It involves captures or threats
    4. There are sharp evaluation changes
    """
    # Skip if node has too many visits (well-explored)
    if node.N > visit_threshold:
        return False
        
    # Check for value volatility among siblings
    if node.has_parent:
        sibling_values = node.parent.child_Q()
        value_std = np.std(sibling_values[sibling_values != 0])
        if value_std > 0.3:  # High volatility threshold
            return True
            
    # Check for captures or material imbalance in Go
    if hasattr(env, 'position') and hasattr(env.position, 'board'):
        # For Go: Check if there are groups with few liberties
        for y in range(env.position.board.shape[0]):
            for x in range(env.position.board.shape[1]):
                if env.position.board[y,x] != go.EMPTY:
                    # Get neighbors of this stone
                    neighbors = [
                        (y-1, x), (y+1, x),
                        (y, x-1), (y, x+1)
                    ]
                    # Count empty neighbor positions (liberties)
                    liberty_count = 0
                    for ny, nx in neighbors:
                        if (0 <= ny < env.position.board.shape[0] and 
                            0 <= nx < env.position.board.shape[1] and
                            env.position.board[ny,nx] == go.EMPTY):
                            liberty_count += 1
                    
                    if 1 <= liberty_count <= 2:  # Stone in atari or near atari
                        return True
                
    # Check for sharp evaluation changes
    if node.has_parent and abs(node.Q - node.parent.Q) > 0.5:
        return True
        
    return False


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
    max_minimax_leaves: int = 3,
    root_noise: bool = False,
    warm_up: bool = False,
    deterministic: bool = False,
) -> Tuple[int, np.ndarray, float, float, Node]:

    # Initialize search statistics
    search_stats = {
        'total_time': 0.0,
        'mcts_time': 0.0,
        'minimax_time': 0.0,
        'minimax_calls': 0,
        'minimax_improvements': 0,
        'critical_positions': 0,
        'avg_eval_change': 0.0,
        'max_eval_change': 0.0,
        'tactical_positions': [],  # Track positions where minimax helped
        'eval_improvements': []    # Track evaluation improvements
    }

    start_time = time.perf_counter()

    # Initialize transposition table and MCTS prior map
    transposition_table = TranspositionTable()
    mcts_prior_map = {}

    # Initialize parallel minimax searcher
    parallel_minimax = ParallelMinimax(
        num_threads=num_minimax_threads,
        time_limit=minimax_time_limit,
        base_k=k_best,
        min_k=3,
        max_k=20,
    )

    # Create root node if needed
    if root_node is None:
        prior_prob, value = eval_func(env.observation(), False)
        root_node = Node(to_play=env.to_play, num_actions=env.action_dim, parent=DummyNode())
        expand(root_node, prior_prob, env.zobrist_hash(), mcts_prior_map, env.legal_actions)
        backup(root_node, value, value)

    assert root_node.to_play == env.to_play
    root_legal_actions = env.legal_actions

    if root_noise:
        add_dirichlet_noise(root_node, root_legal_actions)

    # Show initial board state
    logger.info(f"\nCurrent board state:\n{env.position}")

    # Main MCTS loop
    while root_node.N < num_simulations:
        node = root_node
        sim_env = copy.deepcopy(env)
        mcts_start = time.perf_counter()

        # Selection phase
        while node.is_expanded and not sim_env.is_game_over():
            node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init, sim_env.opponent_player)
            sim_env.step(node.move)

        # Store MCTS priors for move ordering
        if node.parent is not None:
            mcts_prior_map[(sim_env.zobrist_hash(), node.move)] = node.parent.child_P[node.move]

        # Get MCTS evaluation
        prior_prob, mcts_value = eval_func(sim_env.observation(), False)

        # Check if position is tactical
        is_tactical = is_tactical_position(node, sim_env, mcts_value)
        if is_tactical:
            search_stats['critical_positions'] += 1
            minimax_start = time.perf_counter()
            
            # Show board state for tactical positions
            logger.info(f"\nAnalyzing tactical position at depth {node.depth}:")
            logger.info(f"{sim_env.position}")
            
            # Run parallel minimax search
            minimax_value, pv = parallel_minimax.iterative_deepening_search(
                sim_env,
                eval_func,
                max_depth,
                k_best,
                transposition_table,
                mcts_prior_map
            )
            
            search_stats['minimax_time'] += time.perf_counter() - minimax_start
            search_stats['minimax_calls'] += 1
            
            # Track evaluation changes
            eval_diff = abs(minimax_value - mcts_value)
            search_stats['avg_eval_change'] = (search_stats['avg_eval_change'] * (search_stats['minimax_calls'] - 1) + eval_diff) / search_stats['minimax_calls']
            search_stats['max_eval_change'] = max(search_stats['max_eval_change'], eval_diff)
            
            if eval_diff > 0.3:
                search_stats['minimax_improvements'] += 1
                search_stats['tactical_positions'].append({
                    'depth': node.depth,
                    'mcts_value': mcts_value,
                    'minimax_value': minimax_value,
                    'eval_diff': eval_diff,
                    'pv_length': len(pv)  # Track principal variation length
                })
                search_stats['eval_improvements'].append(eval_diff)
            
            # Only expand if not already expanded
            if not node.is_expanded:
                expand(node, prior_prob, sim_env.zobrist_hash(), mcts_prior_map, sim_env.legal_actions)
            backup(node, mcts_value, minimax_value)
        else:
            # Only expand if not already expanded
            if not node.is_expanded:
                expand(node, prior_prob, sim_env.zobrist_hash(), mcts_prior_map, sim_env.legal_actions)
            backup(node, mcts_value, mcts_value)

        search_stats['mcts_time'] += time.perf_counter() - mcts_start

        # Show progress every 100 simulations with current board
        if root_node.N % 100 == 0:
            logger.info(f"\nProgress: {root_node.N}/{num_simulations} simulations")
            logger.info(f"Current board state:\n{env.position}")

    # Move selection
    search_pi = generate_search_policy(root_node.child_N, 1.0 if warm_up else 0.1, root_legal_actions)
    move = None
    next_root_node = None
    best_child_Q = 0.0

    if deterministic:
        move = np.argmax(root_node.child_N)
    else:
        while move is None or (warm_up and env.has_pass_move and move == env.pass_move) or root_legal_actions[move] != 1:
            move = np.random.choice(np.arange(search_pi.shape[0]), p=search_pi)

    if move in root_node.children:
        next_root_node = root_node.children[move]
        N, W = copy.copy(next_root_node.N), copy.copy(next_root_node.W)
        next_root_node.parent = DummyNode()
        next_root_node.move = None
        next_root_node.N = N
        next_root_node.W = W
        best_child_Q = -next_root_node.Q

    # Log final search statistics and analysis
    search_stats['total_time'] = time.perf_counter() - start_time
    logger.info("\nSearch completed:")
    logger.info(f"Total time: {search_stats['total_time']:.2f}s")
    logger.info(f"MCTS time: {search_stats['mcts_time']:.2f}s")
    logger.info(f"Minimax time: {search_stats['minimax_time']:.2f}s")
    logger.info(f"Critical positions: {search_stats['critical_positions']}")
    
    if search_stats['minimax_calls'] > 0:
        logger.info(f"Minimax improvements: {search_stats['minimax_improvements']}/{search_stats['minimax_calls']}")
        logger.info(f"Average eval change: {search_stats['avg_eval_change']:.3f}")
        
        # Additional analysis for improving minimax
        logger.info("\nAnalysis:")
        if search_stats['tactical_positions']:
            avg_depth = sum(p['depth'] for p in search_stats['tactical_positions']) / len(search_stats['tactical_positions'])
            logger.info(f"Average depth of tactical positions: {avg_depth:.1f}")
            logger.info(f"Distribution of eval improvements: {np.percentile(search_stats['eval_improvements'], [25, 50, 75])}")
            
            # Analyze where minimax helped most
            max_improvement_pos = max(search_stats['tactical_positions'], key=lambda x: x['eval_diff'])
            logger.info(f"Largest improvement: {max_improvement_pos['eval_diff']:.3f} at depth {max_improvement_pos['depth']}")

    logger.info(f"Selected move: {move} (Q-value: {best_child_Q:.3f})")

    return move, search_pi, root_node.Q, best_child_Q, next_root_node


def count_group_liberties(env: BoardGameEnv) -> int:
    """Count number of groups with 1-2 liberties to measure tactical complexity."""
    if not hasattr(env, 'position') or not hasattr(env.position, 'board'):
        return 0
        
    critical_groups = 0
    for y in range(env.position.board.shape[0]):
        for x in range(env.position.board.shape[1]):
            if env.position.board[y,x] != go.EMPTY:
                neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
                liberty_count = 0
                for ny, nx in neighbors:
                    if (0 <= ny < env.position.board.shape[0] and 
                        0 <= nx < env.position.board.shape[1] and
                        env.position.board[ny,nx] == go.EMPTY):
                        liberty_count += 1
                if 1 <= liberty_count <= 2:
                    critical_groups += 1
    return critical_groups
