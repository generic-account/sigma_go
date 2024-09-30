#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Define simulation parameters
NUM_GAMES=10
NUM_SIMULATIONS_BLACK=90
NUM_SIMULATIONS_WHITE=200
USE_MINIMAX_BLACK=true
USE_MINIMAX_WHITE=false

# Define network parameters
NUM_RES_BLOCKS=10
NUM_FILTERS=128
NUM_FC_UNITS=128

# Define Minimax parameters
DEPTH=3
KBEST=5

# Define the black model checkpoint
black_model=154000

# Define the list of white models checkpoints
white_models=(151000 152000 153000 145000 160000 159000 149000 146000 150000 147000)

# Loop over each white model and run matches against the black model
for white_model in "${white_models[@]}"
do
    python3 -m alpha_zero.play.eval_agent_go_mass_matches \
        --num_games=${NUM_GAMES} \
        --num_simulations_black=${NUM_SIMULATIONS_BLACK} \
        --num_simulations_white=${NUM_SIMULATIONS_WHITE} \
        --num_res_blocks=${NUM_RES_BLOCKS} \
        --num_filters=${NUM_FILTERS} \
        --num_fc_units=${NUM_FC_UNITS} \
        --depth=${DEPTH} \
        --k_best=${KBEST} \
        --save_match_dir=./9x9_matches/${black_model}_vs_${white_model} \
        --black_ckpt=./checkpoints/go/9x9/training_steps_${black_model}.ckpt \
        --white_ckpt=./checkpoints/go/9x9/training_steps_${white_model}.ckpt \
        --use_minimax_black=${USE_MINIMAX_BLACK} \
        --use_minimax_white=${USE_MINIMAX_WHITE}
done