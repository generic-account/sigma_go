# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Evaluate SigmaGo vs AlphaZero on Go."""
from absl import flags
import os
import sys
import torch
import logging
import colorlog
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'sigmago_vs_alphazero_{timestamp}.log')
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler - logs only analysis metrics
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    
    # Add a filter to only log final analysis metrics
    class AnalysisFilter(logging.Filter):
        def filter(self, record):
            analysis_keywords = [
                'Search completed:', 'Total time:', 'MCTS time:', 'Minimax time:',
                'Critical positions:', 'Minimax improvements:',
                'Average eval change:', 'Max eval change:',
                'Selected move:', 'Significant improvement found:',
                'Analysis:'  # New keyword for additional analysis
            ]
            return any(keyword in record.msg for keyword in analysis_keywords)
    
    file_handler.addFilter(AnalysisFilter())
    
    # Console handler - shows everything including board state
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 9, 'Board size for Go.')
flags.DEFINE_float('komi', 7.5, 'Komi rule for Go.')
flags.DEFINE_integer(
    'num_stack',
    8,
    'Stack N previous states, the state is an image of N x 2 + 1 binary planes.',
)

flags.DEFINE_integer('num_res_blocks', 10, 'Number of residual blocks in the neural network.')
flags.DEFINE_integer('num_filters', 128, 'Number of filters for the conv2d layers in the neural network.')
flags.DEFINE_integer(
    'num_fc_units',
    128,
    'Number of hidden units in the linear layer of the neural network.',
)

flags.DEFINE_string(
    'sigmago_ckpt',
    './checkpoints/go/9x9/training_steps_154000.ckpt',
    'Load the checkpoint file for SigmaGo (black player).',
)
flags.DEFINE_string(
    'alphazero_ckpt',
    './checkpoints/go/9x9/training_steps_139000.ckpt',
    'Load the checkpoint file for AlphaZero (white player).',
)

flags.DEFINE_integer('num_simulations', 160, 'Number of iterations per MCTS search.')
flags.DEFINE_integer(
    'num_parallel',
    8,
    'Number of leaves to collect before using the neural network to evaluate the positions during MCTS search, 1 means no parallel search.',
)

# SigmaGo specific parameters (minimax)
flags.DEFINE_integer('depth', 2, 'Max depth of minimax search for SigmaGo')
flags.DEFINE_integer('k_best', 5, 'The number of best actions to consider in minimax search for SigmaGo.')
flags.DEFINE_integer('num_minimax_threads', 4, 'Number of threads for parallel minimax search in SigmaGo')
flags.DEFINE_float('minimax_time_limit', 30.0, 'Time limit in seconds for minimax search in SigmaGo')
flags.DEFINE_integer('max_minimax_leaves', 3, 'Maximum number of leaves to evaluate in minimax for SigmaGo')

flags.DEFINE_float('c_puct_base', 19652, 'Exploration constants balancing priors vs. search values.')
flags.DEFINE_float('c_puct_init', 1.25, 'Exploration constants balancing priors vs. search values.')

flags.DEFINE_bool('human_vs_ai', True, 'Black player is human, default on.')
flags.DEFINE_bool('show_steps', False, 'Show step number on stones, default off.')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

# Initialize flags
FLAGS(sys.argv)

os.environ['BOARD_SIZE'] = str(FLAGS.board_size)

from alpha_zero.envs.go import GoEnv
from alpha_zero.envs.gui import BoardGameGui
from alpha_zero.core.network import AlphaZeroNet
from alpha_zero.core.pipeline import create_mcts_player, set_seed, disable_auto_grad
from alpha_zero.utils.util import create_logger


def main():
    # Set up logging first
    logger = setup_logging()
    logger.info("Starting new game: SigmaGo vs AlphaZero")
    
    set_seed(FLAGS.seed)
    runtime_device = 'cpu'
    if torch.cuda.is_available():
        runtime_device = 'cuda'
    elif torch.backends.mps.is_available():
        runtime_device = 'mps'

    eval_env = GoEnv(komi=FLAGS.komi, num_stack=FLAGS.num_stack)

    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    def network_builder():
        return AlphaZeroNet(
            input_shape,
            num_actions,
            FLAGS.num_res_blocks,
            FLAGS.num_filters,
            FLAGS.num_fc_units,
        )

    def load_checkpoint_for_net(network, ckpt_file, device):
        if ckpt_file and os.path.isfile(ckpt_file):
            loaded_state = torch.load(ckpt_file, map_location=torch.device(device))
            network.load_state_dict(loaded_state['network'])
        else:
            logger.warning(f'Invalid checkpoint file "{ckpt_file}"')

    def sigmago_player_builder(ckpt_file, device):
        """Creates a SigmaGo player that uses MCTS with minimax."""
        network = network_builder().to(device)
        disable_auto_grad(network)
        load_checkpoint_for_net(network, ckpt_file, device)
        network.eval()

        return create_mcts_player(
            network=network,
            device=device,
            num_simulations=FLAGS.num_simulations,
            num_parallel=FLAGS.num_parallel,
            k_best=FLAGS.k_best,
            depth=FLAGS.depth,
            num_minimax_threads=FLAGS.num_minimax_threads,
            minimax_time_limit=FLAGS.minimax_time_limit,
            max_minimax_leaves=FLAGS.max_minimax_leaves,
            root_noise=False,
            deterministic=True,
            use_minimax=True,  # Enable minimax for SigmaGo
        )

    def alphazero_player_builder(ckpt_file, device):
        """Creates an AlphaZero player that uses pure MCTS."""
        network = network_builder().to(device)
        disable_auto_grad(network)
        load_checkpoint_for_net(network, ckpt_file, device)
        network.eval()

        return create_mcts_player(
            network=network,
            device=device,
            num_simulations=FLAGS.num_simulations,
            num_parallel=FLAGS.num_parallel,
            root_noise=False,
            deterministic=True,
            use_minimax=False,  # Disable minimax for AlphaZero
        )

    # Wrap players for the GUI program
    def wrap_player(player) -> int:
        def act(env):
            action, *_ = player(env, None, FLAGS.c_puct_base, FLAGS.c_puct_init, False)
            return action
        return act

    # Create players
    if FLAGS.human_vs_ai:
        black_player = 'human'
        white_player = sigmago_player_builder(FLAGS.sigmago_ckpt, runtime_device)
        white_player = wrap_player(white_player)
        white_name = "AlphaZero"
        black_name = "Human"
    else:
        # AI vs AI: SigmaGo (Black) vs AlphaZero (White)
        black_player = alphazero_player_builder(FLAGS.alphazero_ckpt, runtime_device)
        black_player = wrap_player(black_player)
        white_player = alphazero_player_builder(FLAGS.alphazero_ckpt, runtime_device)
        white_player = wrap_player(white_player)
        black_name = "SigmaGo"
        white_name = "AlphaZero"

    game_gui = BoardGameGui(
        eval_env,
        black_player=black_player,
        white_player=white_player,
        show_steps=FLAGS.show_steps,
        delay=1000,
    )

    game_gui.start()


if __name__ == '__main__':
    main()
