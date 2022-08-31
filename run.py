"""Running RL on both 2D and 3D morphologies on multiple arenas"""
import turibolt as bolt
import os

config = bolt.get_current_config()

params_path = ['configs/locomotion2d/2d_eval.yaml', 'configs/locomotion3d/3d_eval.yaml']
arenas_2d = ['EmptyCorridor', 'GapsCorridor', 'HurdlesCorridor', 'GM_Terrain']
arenas_3d = ['EmptyCorridor', 'GapsCorridor', 'WallsCorridor', 'HurdlesCorridor', 'GM_Terrain3D']

for j, path in enumerate(params_path):
    config['command'] = 'python scripts/train_rl.py'
    config["is_parent"] = False

    if j < 1:   # 2D case
        for arena in arenas_2d:
            config['arguments'] = [{'path':path}, {'arena':arena}]
            bolt.submit(config)

    else:       # 3D case
        for arena in arenas_3d:
            config['arguments'] = [{'path':path}, {'arena':arena}]
            bolt.submit(config)