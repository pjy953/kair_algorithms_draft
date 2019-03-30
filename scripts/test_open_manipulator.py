#!/usr/bin/env python

from envs.open_manipulator import OpenManipulatorEnv
import rospy
import numpy as np
from math import pi, cos, sin, radians

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from open_manipulator_msgs.msg import (
JointPosition,
KinematicsPose,
OpenManipulatorState,
)

from open_manipulator_msgs.srv import (
GetJointPosition,
GetKinematicsPose,
SetActuatorState,
SetDrawingTrajectory,
SetJointPosition,
SetKinematicsPose,
)

overhead_orientation = Quaternion(
                            x=-0.00142460053167,
                            y=0.999994209902,
                            z=-0.00177030764765,
                            w=0.00253311793936)

def test_reset():
    env = OpenManipulatorEnv()
    obs = env.reset()
    # assert obs in specific boundary

def test_forward():
    env = OpenManipulatorEnv()
    obs = env.reset()
    _pose = Pose()
    _pose.position.x = 0.4
    _pose.position.y = 0.0
    _pose.position.z = 0.1
    _pose.orientation.x = 0.0
    _pose.orientation.y = 0.0
    _pose.orientation.z = 0.0
    _pose.orientation.w = 1.0    
    forward_pose= KinematicsPose()
    forward_pose.pose = _pose
    forward_pose.max_accelerations_scaling_factor = 0.0
    forward_pose.max_velocity_scaling_factor = 0.0
    forward_pose.tolerance = 0.0
    try:
        task_space_srv = rospy.ServiceProxy('/open_manipulator/goal_task_space_path', SetKinematicsPose)
        resp_delete = task_space_srv("arm","gripper", forward_pose, 2.0 )
    except rospy.ServiceException as e:
        rospy.loginfo("Path planning service call failed: {0}".format(e))

def test_rotate():
    _qpose = JointPosition()
    _qpose.joint_name = ['joint1','joint2','joint3','joint4']
    _qpose.position = [0.5, 0.0, 0.0, 0.5]
    _qpose.max_accelerations_scaling_factor = 0.0
    _qpose.max_velocity_scaling_factor = 0.0
    try:
        task_space_srv = rospy.ServiceProxy('/open_manipulator/goal_joint_space_path_from_present', SetJointPosition)
        resp_delete = task_space_srv("arm", _qpose, 2.0 )
    except rospy.ServiceException, e:
        rospy.loginfo("Path planning service call failed: {0}".format(e))
    _qpose.position[0] += -1.0
    _qpose.position[3] += -1.0
    try:
        resp_delete = task_space_srv("arm", _qpose, 2.0 )
    except rospy.ServiceException, e:
        rospy.loginfo("Path planning service call failed: {0}".format(e))
    # define actions
    # assert obs in specific boundary

def test_block_loc():
    env = OpenManipulatorEnv()
    for iter in range(20):
        b_pose =Pose()
        b_pose.position.x = np.random.uniform(0.15, .20)
        b_pose.position.y = np.random.uniform(-0.2, 0.2)
        b_pose.position.z = 0.00
        b_pose.orientation = overhead_orientation
        env._load_target_block(block_pose=b_pose)
        rospy.sleep(2.0)
        env._delete_target_block()
    # block generation code
    # assert block in specific boundary (gripper's movable area)


def test_achieve_goal():
    env = OpenManipulatorEnv()
    for iter in range(20):
        b_pose =Pose()
        b_pose.position.x = np.random.uniform(0.25, .6)
        b_pose.position.y = np.random.uniform(-0.4, 0.4)
        b_pose.position.z = 0.00
        b_pose.orientation = overhead_orientation
        env._load_target_block(block_pose=b_pose)
        r_pose = Pose()
        r_pose.position = b_pose.position
        r_pose.position.z = 0.08
        forward_pose= KinematicsPose()
        forward_pose.pose = r_pose
        forward_pose.max_accelerations_scaling_factor = 0.0
        forward_pose.max_velocity_scaling_factor = 0.0
        forward_pose.tolerance = 0.0
        try:
            task_space_srv = rospy.ServiceProxy('/open_manipulator/goal_task_space_path', SetKinematicsPose)
            resp_delete = task_space_srv("arm","gripper", forward_pose, 2.0 )
        except rospy.ServiceException, e:
            rospy.loginfo("Path planning service call failed: {0}".format(e))
        rospy.sleep(5.0)
        env._delete_target_block()

def test_workspace_limit():
    """ TODO: add static block 
    """
    env = OpenManipulatorEnv()
    for iter in range(100):
        _polar_rad = np.random.uniform(0.134, 0.32)
        _polar_theta = np.random.uniform(-pi*0.7/4, pi*0.7/4)

        b_pose =Pose()
        b_pose.position.x = _polar_rad*cos(_polar_theta)
        b_pose.position.y = _polar_rad*sin(_polar_theta)
        b_pose.position.z = np.random.uniform(0.05, 0.28)
        b_pose.orientation = overhead_orientation
        env._load_target_block(block_pose=b_pose)
        r_pose = Pose()
        r_pose.position = b_pose.position
        forward_pose= KinematicsPose()
        forward_pose.pose = r_pose
        forward_pose.max_accelerations_scaling_factor = 0.0
        forward_pose.max_velocity_scaling_factor = 0.0
        forward_pose.tolerance = 0.0
        try:
            task_space_srv = rospy.ServiceProxy('/open_manipulator/goal_task_space_path', SetKinematicsPose)
            resp_delete = task_space_srv("arm","gripper", forward_pose, 3.0 )
        except rospy.ServiceException, e:
            rospy.loginfo("Path planning service call failed: {0}".format(e))
        rospy.sleep(3.0)
        env._check_for_termination()
        env._delete_target_block()


    # define actions
    # define goal
    # assert gripper reach goal


if __name__ == '__main__':
    # test_reset()
    # test_forward()
    # test_rotate()
    # test_block_loc()
    # test_achieve_goal()
    test_workspace_limit()