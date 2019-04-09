#! /usr/bin/env python

# -*- coding: utf-8 -*-
"""Train or test algorithms on OpenManipulator Reacher-v0 on Gazebo.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import argparse
import importlib

<<<<<<< HEAD
import algorithms.common.helper_functions as common_utils
from config.environment.open_manipulator import config as env_cfg
=======
from config.environment.open_manipulator import config as env_cfg

import algorithms.common.helper_functions as common_utils
>>>>>>> a7e00fe21da638552ec615d39ba50a1d4d1e0d00
from envs.open_manipulator.open_manipulator_reacher_env import OpenManipulatorReacherEnv

# configurations
parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility"
)
parser.add_argument("--algo", type=str, default="td3", help="choose an algorithm")
parser.add_argument(
    "--test", dest="test", action="store_true", help="test mode (no training)"
)
parser.add_argument(
    "--load-from", type=str, help="load the saved model and optimizer at the beginning"
)
parser.add_argument(
    "--off-render", dest="render", action="store_false", help="turn off rendering"
)
parser.add_argument(
    "--render-after",
    type=int,
    default=0,
    help="start rendering after the input number of episode",
)
parser.add_argument("--log", dest="log", action="store_true", help="turn on logging")
parser.add_argument("--save-period", type=int, default=200, help="save model period")
parser.add_argument("--episode-num", type=int, default=20000, help="total episode num")
parser.add_argument(
    "--max-episode-steps", type=int, default=-1, help="max episode step"
)
parser.add_argument(
    "--demo-path", type=str, default="data/reacher_demo.pkl", help="demonstration path"
)

parser.set_defaults(test=False)
parser.set_defaults(load_from=None)
parser.set_defaults(render=True)
parser.set_defaults(log=False)
args = parser.parse_args()


def main():
    """Main."""
    # env initialization
    env = OpenManipulatorReacherEnv(env_cfg)

    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # agent initialization
<<<<<<< HEAD
    module_path = "config.agent.open_manipulator_reacher_v0." + args.algo
    agent = importlib.import_module(module_path)
    agent = agent.get(env, args)
=======
    module_path = "examples.open_manipulator_reacher_v0." + args.algo
    agent_config = importlib.import_module(module_path)
    agent = agent_config.get_agent(env, args)
>>>>>>> a7e00fe21da638552ec615d39ba50a1d4d1e0d00

    # run
    if args.test:
        agent.test()
    else:
        agent.train()


if __name__ == "__main__":
    main()
