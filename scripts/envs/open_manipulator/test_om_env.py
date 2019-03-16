#!/usr/bin/env python

import numpy as np
# import gym
import time as timer
import time
from math import pi, cos, sin, radians
from OMReacher_v0 import robotEnv # REFACTOR TO REMOVE GOALS
import rospy
# Leveraging demonstration is same as TD3


import pickle
import os
# apply hindsight experience replay.


if __name__ == '__main__':
    env = robotEnv()
    _ = env.reset()
    r = rospy.Rate(100)
    for i in range(2000):

        act = -2*pi*sin(i/2000.0)*np.ones(6)

        _, _, _ = env.step(action = act, step=i)
        r.sleep()