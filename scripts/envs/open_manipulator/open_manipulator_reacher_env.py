#! usr/bin/env python

import gym
import numpy as np
from gym.utils import seeding

from ros_interface import (
    OpenManipulatorRosGazeboInterface,
    OpenManipulatorRosRealInterface,
)


class OpenManipulatorReacherEnv(gym.Env):
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_mode = self.cfg.env_mode
        self._max_episode_steps = self.cfg.max_episode_steps
        self.reward_rescale_ratio = self.cfg.reward_rescale_ratio
        self.reward_func = self.cfg.reward_func

        assert self.env_mode in ["sim", "real"]
        if self.env_mode == "sim":
            self.ros_interface = OpenManipulatorRosGazeboInterface(self.cfg)
        else:
            self.ros_interface = OpenManipulatorRosRealInterface()

        self.episode_steps = 0
        self.done = False
        self.reward = 0

        self.action_space = self.ros_interface.action_space
        self.observation_space = self.ros_interface.observation_space
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Function executed each time step.

        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.

        Args:
            action: action
        Returns:
            Tuple of obs, reward_rescale * reward, done
        """
        if action is None:
            action = np.array([1, 1, 1, 1, 1, 1])

        self.done = False

        if self.episode_steps == self._max_episode_steps:
            self.done = True
            self.episode_steps = 0

        act = action.flatten().tolist()
        self.ros_interface.set_joints_position(act)

        if self.env_mode == "sim":
            self.reward = self.compute_reward()
            if self.ros_interface.check_for_termination():
                print("Terminates current Episode : OUT OF BOUNDARY")
            elif self.ros_interface.check_for_success():
                print("Succeeded current Episode")
        obs = self.ros_interface.get_observation()

        self.episode_steps += 1

        return obs, self.reward_rescale_ratio * self.reward, self.done, None

    def reset(self):
        """Attempt to reset the simulator.

        Since we randomize initial conditions, it is possible to get into
        a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an
        object is within the hand).

        In this case, we just keep randomizing until we eventually achieve
        a valid initial
        configuration.

        Returns:
            obs (array) : Array of joint position, joint velocity, joint effort
        """
        self.ros_interface.reset_gazebo_world()
        obs = self.ros_interface.get_observation()

        return obs

    def compute_reward(self):
        """Computes shaped/sparse reward for each episode.

        Returns:
            reward (Float64) : L2 distance of current distance and squared sum velocity.
        """
        cur_dist = self.ros_interface.get_dist()
        if self.reward_func == "sparse":
            # 1 for success else 0
            reward = cur_dist <= self.ros_interface.distance_threshold
            reward = reward.astype(np.float32)
        elif self.reward_func == "l2":
            # - L2 distance
            reward = - cur_dist
        else:
            raise ValueError

        return reward

    def render(self):
        pass
