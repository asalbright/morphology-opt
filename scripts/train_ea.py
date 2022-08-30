import argparse
import yaml
import pprint
from optimal_agents.utils.loader import Parameters
from optimal_agents.utils.trainer import train_ea

import turibolt as bolt
import os

data_path = os.path.join(bolt.ARTIFACT_DIR, "data")

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str, required=True, help="Config file for experiment")
parser.add_argument("--output", "-o", type=str, required=False, default=None)
args = parser.parse_args()

params = Parameters.load(args.path)
train_ea(params, path=data_path)
