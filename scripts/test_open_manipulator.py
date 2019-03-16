#!/usr/bin/env python

import os
import pickle
import time as timer
from math import cos, pi, radians, sin

import numpy as np

import rospy
from env.OMReacher_v0 import robotEnv



def test_reset():
    env = robotEnv()
    obs = env.reset()
    return obs


def test_one_step():
    env = robotEnv()
    obs = env.reset()
    act = -2*pi*sin(i/2000.0)*np.ones(6)
    next_obs, reward, done = env.step(act)
    return next_obs, reward, done


if __name__ == '__main__':
    obs = test_reset()
    next_obs, reward, done = test_one_step()
    print("test_reset:", obs)
    print("test_one_step:", next_obs, reward, done)
