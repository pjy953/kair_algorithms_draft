#!/usr/bin/env python
import threading
from math import pi, pow

import numpy as np

# ROS Imports
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

import modern_robotics as r

# Local Imports
import open_manipulator_description as s

####################
# GLOBAL VARIABLES #
####################
DAMPING = 0.02  # 0.00
JOINT_VEL_LIMIT = 2  # 2rad/s


class VelocityControl(object):
    def __init__(self):
        rospy.loginfo("Creating VelocityController class")

        # Grab M0 and Blist from open_manipulator_description.py
        self.M0 = s.M  # Zero config of manipulator
        self.Blist = s.Blist  # 6x4 screw axes mx of right arm
        self.Slist = s.Slist

        # Shared variables
        self.mutex = threading.Lock()
        self.damping = rospy.get_param("~damping", DAMPING)
        self.joint_vel_limit = rospy.get_param("~joint_vel_limit", JOINT_VEL_LIMIT)
        self.q = np.zeros(4)  # Joint angles
        self.q_desired = np.zeros(4)
        self.qdot = np.zeros(4)  # Joint velocities
        self.T_goal = r.FKinSpace(self.M0, self.Slist, self.q[0:4])

        self.init = False
        self.is_joint_states_cb = False
        self.is_ref_pose_cb = False
        # Subscriber
        self.joint_states_sub = rospy.Subscriber(
            "/open_manipulator/joint_states", JointState, self.joint_states_cb
        )
        self.ref_pose_sub = rospy.Subscriber("/teacher/ik_vel/", Pose, self.ref_pose_cb)
        # Command publisher
        self.j1_pos_command_pub = rospy.Publisher(
            "/open_manipulator/joint1_position/command", Float64, queue_size=3
        )
        self.j2_pos_command_pub = rospy.Publisher(
            "/open_manipulator/joint2_position/command", Float64, queue_size=3
        )
        self.j3_pos_command_pub = rospy.Publisher(
            "/open_manipulator/joint3_position/command", Float64, queue_size=3
        )
        self.j4_pos_command_pub = rospy.Publisher(
            "/open_manipulator/joint4_position/command", Float64, queue_size=3
        )

        self.joint_pos_command_to_dxl_pub = rospy.Publisher(
            "/open_manipulator/joint_position/command", Float64MultiArray, queue_size=3
        )
        self.end_effector_pose_pub = rospy.Publisher(
            "/open_manipulator/end_effector_pose", Pose, queue_size=3
        )

        rospy.set_param("bool_demo_run", False)
        self.r = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self.is_joint_states_cb is True:
                if self.init is False:
                    self.q_desired = list(self.q)
                    self.init = True
                self.calc_joint_vel()
            self.r.sleep()

    def joint_states_cb(self, joint_states):
        self.is_joint_states_cb = True
        i = 0
        while i < 4:
            self.q[i] = joint_states.position[i + 2]
            self.qdot[i] = joint_states.velocity[i + 2]
            i += 1

    def ref_pose_cb(self, some_pose):  # Takes target pose, returns ref se3
        self.is_ref_pose_cb = True
        p = np.array([some_pose.position.x, some_pose.position.y, some_pose.position.z])

        with self.mutex:
            goal_tmp = r.FKinSpace(self.M0, self.Slist, np.array([0.0, 0.0, 0.0, 0.0]))
            goal_tmp[0:3, 3] = p
        with self.mutex:
            self.T_goal = goal_tmp

    def get_phi(self, R_d, R_cur):
        phi = -0.5 * (
            np.cross(R_d[0:3, 0], R_cur[0:3, 0])
            + np.cross(R_d[0:3, 1], R_cur[0:3, 1])
            + np.cross(R_d[0:3, 2], R_cur[0:3, 2])
        )
        return phi

    def joint_limit_check(self, q_target):
        q_limit_L = [-pi * 0.9, -pi * 0.57, -pi * 0.3, -pi * 0.57]
        q_limit_H = [pi * 0.9, pi * 0.5, pi * 0.44, pi * 0.65]
        for i in range(4):
            if q_target[i] < q_limit_L[i]:
                q_target[i] = q_limit_L[i]
            elif q_target[i] > q_limit_H[i]:
                q_target[i] = q_limit_H[i]
        return q_target

    def calc_joint_vel(self):

        rospy.logdebug("Calculating joint velocities...")

        # Body stuff
        Tbs = self.M0

        # Desired config: base to desired - Tbd
        with self.mutex:
            q_now = self.q

        T_cur = r.FKinSpace(Tbs, self.Slist, q_now)
        T_sd = self.T_goal[:]

        e = np.zeros(6)
        e[0:3] = self.get_phi(T_sd[0:3, 0:3], T_cur[0:3, 0:3])
        e[3:6] = T_sd[0:3, 3] - T_cur[0:3, 3]
        # Construct BODY JACOBIAN for current config
        Jb = r.JacobianBody(self.Blist, q_now)

        Jv = np.dot(T_cur[0:3, 0:3], Jb[3:6, :])

        # Jw = Jb[0:3,:]
        # JwInv = np.dot(np.linalg.inv(np.dot(Jw.T,Jw)+ pow(0.05, 2)*np.eye(4)),Jw.T,)
        # nullJv = np.eye(4) - np.dot(np.dot(Jv.T, np.linalg.inv(np.dot(Jv,Jv.T))), Jv)

        # Desired ang vel - Eq 5 from Chiaverini & Siciliano, 1994
        # Managing singularities: naive least-squares damping
        invterm = np.linalg.inv(np.dot(Jv, Jv.T) + pow(self.damping, 2) * np.eye(3))
        kp = 2.0
        qdot_new = np.dot(np.dot(Jv.T, invterm), kp * e[3:6])

        # Scaling joint velocity
        minus_v = abs(np.amin(qdot_new))
        plus_v = abs(np.amax(qdot_new))
        if minus_v > plus_v:
            scale = minus_v
        else:
            scale = plus_v
        if scale > self.joint_vel_limit:
            qdot_new = 2.0 * (qdot_new / scale) * self.joint_vel_limit
        self.qdot = qdot_new  # 1x7

        dt = 0.01
        if self.is_ref_pose_cb is True:
            self.q_desired = self.q_desired + qdot_new * dt
        if self.is_ref_pose_cb is False:
            self.q_desired = self.q_desired

        self.q_desired = self.joint_limit_check(self.q_desired)

        self.j1_pos_command_pub.publish(self.q_desired[0])
        self.j2_pos_command_pub.publish(self.q_desired[1])
        self.j3_pos_command_pub.publish(self.q_desired[2])
        self.j4_pos_command_pub.publish(self.q_desired[3])
        self.joint_pos_command_to_dxl_pub.publish(data=self.q_desired)

        ee_pose = Pose()
        ee_pose.position.x = T_cur[0, 3]
        ee_pose.position.y = T_cur[1, 3]
        ee_pose.position.z = T_cur[2, 3]

        self.end_effector_pose_pub.publish(ee_pose)

        return


def main():
    rospy.init_node("velocity_control")

    try:
        VelocityControl()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()


if __name__ == "__main__":
    main()
