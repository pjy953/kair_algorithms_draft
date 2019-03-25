#!/usr/bin/env python
# ROS Imports
import os
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
import json
from collections import OrderedDict


class DemoLogger(object):
    def __init__(self):
        print("Logging Program on")
        self.q_current = [0.0, 0.0, 0.0, 0.0]
        self.q_desired = [0.0, 0.0, 0.0, 0.0]
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_z = 0.0
        self.control_duration = 0.0

        self.joint_states_sub = rospy.Subscriber(
            "/open_manipulator/joint_states", JointState, self.joint_states_cb
        )
        self.joint_pos_command_sub = rospy.Subscriber(
            "/open_manipulator/joint_position/command",
            Float64MultiArray,
            self.joint_command_cb,
        )
        self.goal_pose_sub = rospy.Subscriber(
            "/teacher/ik_vel/", Pose, self.goal_pose_cb
        )

        self.is_record = False

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if (rospy.get_param('bool_demo_run') is True):
                if (self.is_record is False):
                    self.data = OrderedDict()
                    self.data["current q"] = []
                    self.data["desired q"] = []
                    self.data["goal"] = []
                    self.data["control duration"] = self.control_duration
                    self.is_record = True
                self.log_data()

            if (self.is_record is True and rospy.get_param('bool_demo_run') is False):
                with open('demo_data.json', 'w') as make_file:
                    json.dump(self.data, make_file, ensure_ascii=False)
                break
            rate.sleep()


    def joint_states_cb(self, joint_states):
        for i in range(4):
            self.q_current[i] = joint_states.position[i + 2]

    def joint_command_cb(self, joint_desired):
        for i in range(4):
            self.q_desired[i] = joint_desired.data[i]

    def goal_pose_cb(self, goal):
        self.goal_x = goal.position.x
        self.goal_y = goal.position.y
        self.goal_z = goal.position.z
        self.control_duration = goal.orientation.w

    def log_data(self):
        self.data["current q"].append(self.q_current)
        self.data["desired q"].append(self.q_desired)
        self.data["goal"].append([self.goal_x, self.goal_y, self.goal_z])



def main():
    rospy.init_node("demo_logger")

    try:
        DemoLogger()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
