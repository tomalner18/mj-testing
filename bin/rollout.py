#!/usr/bin/env python3
import logging
import click
import json
import time
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple
from multiprocessing import Pool

from mae_envs.viewer.rollout_viewer import RolloutViewer
from mae_envs.wrappers.multi_agent import JoinMultiAgentActions
from mujoco_worldgen.util.envs import examine_env, load_env
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.util.parse_arguments import parse_arguments

def parallel_examine_env(args):
    env_name, kwargs, size, core_dir, envs_dir, xmls_dir, env_viewer = args
    examine_env(env_name, kwargs, size, core_dir=core_dir, envs_dir=envs_dir, xmls_dir=xmls_dir, env_viewer=env_viewer)


@click.command()
@click.argument('argv', nargs=-1, required=False)
def main(argv):
    names, kwargs = parse_arguments(argv)

    env_name = names[0]

    core_dir = abspath(join(dirname(__file__), '..'))
    envs_dir = 'mae_envs/envs',
    xmls_dir = 'xmls',
    iterations = 10


    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    size_times = dict()
    if len(names) == 1:  # examine the environment
        for i in range(iterations):
            print("Iteration: ", i)
            for size in sizes:
                print("Size: ", size)
                args_list = [(env_name, kwargs, size, core_dir, envs_dir, xmls_dir, RolloutViewer)]
                st = time.time()
                with Pool(size) as p:
                    p.map(parallel_examine_env, args_list * size)
                dt = time.time() - st
                # Write reset and rollout times to size marked file
                if size not in size_times:
                    size_times[size] = [dt]
                else:
                    size_times[size].append(dt)

                print("Time: ", dt)
        with open(f"rollout_times_100_ovr1.json", "w") as f:
            json.dump(size_times, f)

if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()