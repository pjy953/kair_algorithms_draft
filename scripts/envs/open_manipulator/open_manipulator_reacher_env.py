# ! usr/bin/env python

import time

import gym
import numpy as np
from gym.utils import seeding

import rospy
from gazebo_msgs.srv import GetModelState
from ros_interface import (
    OpenManipulatorRosGazeboInterface,
    OpenManipulatorRosRealInterface,
)


class OpenManipulatorReacherEnv(gym.Env):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = self.cfg.mode
        self.max_steps = self.cfg.max_steps
        self.reward_rescale_ratio = self.cfg.reward_rescale_ratio
        self.reward_func = self.cfg.reward_func

        assert self.mode in ["sim", "real"]
        if self.mode == "sim":
            self.ros_interface = OpenManipulatorRosGazeboInterface()
        else:
            self.ros_interface = OpenManipulatorRosRealInterface()

        self.episode_steps = 0
        self.done = False
        self.reward = 0
        self.tic = 0.0
        self.toc = 0.0
        self.elapsed = 0.0

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

        self.prev_tic = self.tic
        self.tic = time.time()
        self.elapsed = time.time() - self.prev_tic
        self.done = False

        if self.episode_steps == self.max_steps:
            self.done = True
            self.episode_steps = 0

        act = action.flatten().tolist()
        self.ros_interface.set_joints_position(act)

        if not self.model == "sim":
            self.reward = self._compute_reward()
            if self._check_for_termination():
                rospy.logwarn("Terminates current Episode : OUT OF BOUNDARY")
            elif self._check_for_success():
                rospy.logwarn("Succeeded current Episode")
        _joint_pos, _joint_vels, _joint_effos = self.ros_interface.get_joints_states()

        if np.mod(self.episode_steps, 10) == 0:
            rospy.logwarn("PER STEP ELAPSED : ", self.elapsed)
            rospy.logwarn("SPARSE REWARD : ", self.reward_rescale * self.reward)
            rospy.logwarn("CURRNET EE POSITINO: ", self.ros_interface.gripper_position)
            rospy.logwarn("ACIONS: ", act)

        obs = np.array([_joint_pos, _joint_vels, _joint_effos])
        self.episode_steps += 1

        return obs, self.reward_rescale * self.reward, self.done

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
        self.ros_interface._reset_gazebo_world()
        _joint_pos, _joint_vels, _joint_effos = self.ros_interface.get_joints_states()
        obs = np.array([_joint_pos, _joint_vels, _joint_effos])

        return obs

    def _check_robot_moving(self):
        """Check if robot has reached its initial pose.

        Returns:
            True if not stopped.
        """
        while not rospy.is_shutdown():
            if self.moving_state == "STOPPED":
                break
        return True

    def _compute_reward(self):
        """Computes shaped/sparse reward for each episode.

        Returns:
            reward (Float64) : L2 distance of current distance and squared sum velocity.
        """
        cur_dist = self._get_dist()
        if self.reward_func == "sparse":
            # 1 for success else 0
            reward = cur_dist <= self.ros_interface.distance_threshold
            reward = reward.astype(np.float32)
            return reward
        else:
            # -L2 distance
            reward = -cur_dist - self.squared_sum_vel
            return reward

    def _get_dist(self):
        """Get distance between end effector pose and object pose.

        Returns:
            L2 norm of end effector pose and object pose.
        """
        rospy.wait_for_service("/gazebo/get_model_state")

        try:
            object_state_srv = rospy.ServiceProxy(
                "/gazebo/get_model_state", GetModelState
            )
            object_state = object_state_srv("block", "world")
            object_pose = [
                object_state.pose.position.x,
                object_state.pose.position.y,
                object_state.pose.position.z,
            ]
            self._obj_pose = np.array(object_pose)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))

        # FK state of robot
        end_effector_pose = np.array(self.ros_interface.get_gripper_position())

        return np.linalg.norm(end_effector_pose - self._obj_pose)