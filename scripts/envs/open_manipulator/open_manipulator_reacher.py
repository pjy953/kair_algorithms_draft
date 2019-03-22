# ! usr/bin/env python

import rospy
import numpy as np
from gazebo_msgs.srv import GetModelState

from open_manipulator_base import OpenManipulatorBase

# Global variables
X_MIN = 0.1
X_MAX = 0.5
Y_MIN = -0.3
Y_MAX = 0.3
Z_MIN = 0.0
Z_MAX = 0.6
TERM_COUNT = 10
SUC_COUNT = 10


class OpenManipulatorReacher(OpenManipulatorBase):
    def __init__(self):
        self.reward_type = 'sparse'
        self.termination_count = 0
        self.success_count = 0

    def check_robot_moving(self):
        """Check if robot has reached its initial pose.
        """
        while not rospy.is_shutdown():
            if self.moving_state == "STOPPED":
                break
        return True

    def check_for_success(self):
        """Check if the agent has succeeded the episode.
        """
        _dist = self._get_dist()
        if _dist < self.distance_threshold:
            self.success_count += 1
            if self.success_count == SUC_COUNT:
                self.done = True
                self.success_count = 0
                return True
        else:
            return False

    def check_for_termination(self):
        """Check if the agent has reached undesirable state. If so, terminate
        the episode early.
        """
        _ee_pose = self.get_gripper_position()
        if not ((X_MIN < _ee_pose[0] < X_MAX) and (Y_MIN < _ee_pose[1] < Y_MAX) and (
                Z_MIN < _ee_pose[1] < Z_MAX)):
            self.termination_count += 1
        if self.termination_count == TERM_COUNT:
            self.done = True
            self.termination_count = 0
            return True
        else:
            return False

    def compute_reward(self):
        """Computes shaped/sparse reward for each episode.
        """
        cur_dist = self._get_dist()
        if self.reward_type == 'sparse':
            return (cur_dist <= self.distance_threshold).astype(
                np.float32)  # 1 for success else 0
        else:
            return -cur_dist - self.squared_sum_vel  # -L2 distance
            # -l2_norm(joint_vels)

    def get_distance(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            object_state_srv = rospy.ServiceProxy('/gazebo/get_model_state',
                                                  GetModelState)
            object_state = object_state_srv("block", "world")
            self._obj_pose = np.array(
                [object_state.pose.position.x, object_state.pose.position.y,
                    object_state.pose.position.z])
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))
        _ee_pose = np.array(self.get_gripper_position())  # FK state of robot
        return np.linalg.norm(_ee_pose - self._obj_pose)
