import sente
import random
from visualization.katago import KataGo

# START OF SIGMAGO + ALPHA GO PREP ########
import os
os.environ['BOARD_SIZE'] = '9'
import sys
import torch

from alpha_zero.envs.coords import CoordsConvertor

converter = CoordsConvertor(9)

from alpha_zero.envs.go import GoEnv
from alpha_zero.core.network import AlphaZeroNet
from alpha_zero.core.pipeline import create_mcts_player, disable_auto_grad

runtime_device = 'cpu'
if torch.cuda.is_available():
    runtime_device = 'cuda'
elif torch.backends.mps.is_available():
    runtime_device = 'mps'

eval_env = GoEnv(komi=0, num_stack=8)

input_shape = eval_env.observation_space.shape
num_actions = eval_env.action_space.n

def network_builder():
    return AlphaZeroNet(
        input_shape,
        num_actions,
        10,
        128,
        128,
    )
def load_checkpoint_for_net(network, ckpt_file, device):
    loaded_state = torch.load(ckpt_file, map_location=torch.device(device))
    network.load_state_dict(loaded_state['network'])

def mcts_player_builder(ckpt_file, device, minimax=False):
    network = network_builder().to(device)
    disable_auto_grad(network)
    load_checkpoint_for_net(network, ckpt_file, device)
    network.eval()

    return create_mcts_player(
        network=network,
        device=device,
        num_simulations=400,
        num_parallel=8,
        root_noise=False,
        deterministic=True,
    )

# END OF SIGMAGO + ALPHAGO PREP #######

class Info:
    def __init__(self, size, pDist, passProb, value):
        # normalize
        sum=passProb
        for row in pDist:
            for val in row:
                sum+=val
        for i in range(size):
            for j in range(size):
                pDist[i][j]/=sum
        passProb/=sum
        self.pDist = pDist
        self.passProb = passProb
        self.value = value
        
def agEvaluate(moves, size) -> Info:
    if (size == 19):
        ans = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(20):
            ans[random.randint(0, size-1)][random.randint(0, size-1)] = random.random()
        return Info(size, ans, random.random(), random.random())
    else:
        eval_env.reset()
        for move in moves:
            if move == "pass":
                eval_env.step(81);
            else:
                eval_env.step(converter.to_flat(move));
        query = mcts_player_builder("checkpoints/go/9x9/training_steps_160000.ckpt", runtime_device, False)(eval_env, None, 19652, 1.25, 0)
        pDist = query[1]
        passProb = (pDist[-1] + 100) % 100
        pDist = pDist[:81].reshape(9, 9);
        return Info(size, pDist, passProb, query[2]);

def sgEvaluate(moves, size) -> Info:
    if (size == 19):
        ans = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(20):
            ans[random.randint(0, size-1)][random.randint(0, size-1)] = random.random()
        return Info(size, ans, random.random(), random.random())
    else:
        eval_env.reset()
        for move in moves:
            if move == "pass":
                eval_env.step(81);
            else:
                eval_env.step(converter.to_flat(move));
        query = mcts_player_builder("checkpoints/go/9x9/training_steps_160000.ckpt", runtime_device, True)(eval_env, None, 19652, 1.25, 1000)
        pDist = query[1]
        passProb = (pDist[-1] + 100) % 100
        pDist = pDist[:81].reshape(9, 9);
        return Info(size, pDist, passProb, query[2]);


def kgEvaluate(moves, size) -> Info:
    # add paths to katago executable, analysis config, and model bin.gz file
    katago = KataGo("/opt/homebrew/Cellar/katago/1.14.1/bin/katago", "/Users/LIMSOKCHEA/.katago/default_analysis.cfg", "/Users/LIMSOKCHEA/.katago/default_model.bin.gz")

    query = katago.query(size, moves, 0)

    pDist = [[0 for _ in range(size)] for _ in range(size)]
    passProb = 0

    for move in query["moveInfos"]:
        if move["move"] == "pass":
            passProb = move["winrate"]
        else:
            map = {"A":0,
                    "B":1,
                    "C":2,
                    "D":3,
                    "E":4,
                    "F":5,
                    "G":6,
                    "H":7,
                    "J":8,
                    "K":9,
                    "L":10,
                    "M":11,
                    "N":12,
                    "O":13,
                    "P":14,
                    "Q":15,
                    "R":16,
                    "S":17,
                    "T":18,
                    "U":19}
            row = map[move["move"][0]]
            col = size - int(move["move"][1])
            pDist[row][col] = move["winrate"]

    katago.close()

    return Info(size, pDist, passProb, query["rootInfo"]["winrate"])