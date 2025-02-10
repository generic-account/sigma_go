#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Define simulation parameters
NUM_GAMES=10
NUM_SIMULATIONS=400
NUM_PARALLEL=8

# Define network parameters
NUM_RES_BLOCKS=10
NUM_FILTERS=128
NUM_FC_UNITS=128

# Define MCTS parameters
C_PUCT_BASE=19652
C_PUCT_INIT=1.25

# Define SigmaGo Minimax parameters
DEPTH=3
K_BEST=5
NUM_MINIMAX_THREADS=4
MINIMAX_TIME_LIMIT=30.0
MAX_MINIMAX_LEAVES=3

# Define the black model checkpoint (SigmaGo)
black_model=154000

# Define the list of white models checkpoints (AlphaZero)
white_models=(151000 152000 153000 145000 160000 159000 149000 146000 150000 147000)

# Create logs directory if it doesn't exist
mkdir -p ./logs/mass_matches

# Loop over each white model and run matches against the black model
for white_model in "${white_models[@]}"
do
    echo "Starting matches: SigmaGo (${black_model}) vs AlphaZero (${white_model})"
    
    python3 -m eval_play.eval_agent_go_cmd \
        --board_size=9 \
        --komi=7.5 \
        --num_stack=8 \
        --num_res_blocks=${NUM_RES_BLOCKS} \
        --num_filters=${NUM_FILTERS} \
        --num_fc_units=${NUM_FC_UNITS} \
        --num_simulations=${NUM_SIMULATIONS} \
        --num_parallel=${NUM_PARALLEL} \
        --c_puct_base=${C_PUCT_BASE} \
        --c_puct_init=${C_PUCT_INIT} \
        --depth=${DEPTH} \
        --k_best=${K_BEST} \
        --num_minimax_threads=${NUM_MINIMAX_THREADS} \
        --minimax_time_limit=${MINIMAX_TIME_LIMIT} \
        --max_minimax_leaves=${MAX_MINIMAX_LEAVES} \
        --black_ckpt=./checkpoints/go/9x9/training_steps_${black_model}.ckpt \
        --white_ckpt=./checkpoints/go/9x9/training_steps_${white_model}.ckpt \
        --human_vs_ai=false \
        --seed=1

    # Sleep briefly between matches to ensure clean separation of logs
    sleep 5
done

echo "All matches completed"