import argparse
import yaml
import pprint
from optimal_agents.utils.loader import Parameters
from optimal_agents.utils.trainer import train_rl

import turibolt as bolt
import os

data_path = os.path.join(bolt.ARTIFACT_DIR, "data")

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str, required=True, help="Config file for experiment")
parser.add_argument("--output", "-o", type=str, required=False, default=None)
parser.add_argument("--arena", "-a", type=str, required=False, default=None, help="The name of the arena class")
args = parser.parse_args()

params = Parameters.load(args.path)
params['arena'] = args.arena

train_rl(params, path=data_path)
