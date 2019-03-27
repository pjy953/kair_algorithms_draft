#!/usr/bin/env python
import sys
from math import pi

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

Deg2Rad = pi / 180
Rad2Deg = 1 / Deg2Rad
q_init = [0.0, 0.0, 0.0, 0.0]
is_q_init = False
q_cur = np.zeros(4)


def cubic(time, time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
    x_t = x_0

    if time < time_0:
        x_t = x_0

    elif time > time_f:
        x_t = x_f
    else:
        elapsed_time = time - time_0
        total_time = time_f - time_0
        total_time2 = total_time * total_time
        total_time3 = total_time2 * total_time
        total_x = x_f - x_0

        x_t = (
            x_0
            + x_dot_0 * elapsed_time
            + (
                3 * total_x / total_time2
                - 2 * x_dot_0 / total_time
                - x_dot_f / total_time
            )
            * elapsed_time
            * elapsed_time
            + (-2 * total_x / total_time3 + (x_dot_0 + x_dot_f) / total_time2)
            * elapsed_time
            * elapsed_time
            * elapsed_time
        )

    return x_t


def joint_limit_check(q_target):
    q_limit_L = [-pi * 0.9, -pi * 0.57, -pi * 0.3, -pi * 0.57]
    q_limit_H = [pi * 0.9, pi * 0.5, pi * 0.44, pi * 0.65]
    for i in range(4):
        if q_target[i] < q_limit_L[i]:
            q_target[i] = q_limit_L[i]
            print ("Out of Joint Limit! Target Changed to Limit Value")
        elif q_target[i] > q_limit_H[i]:
            q_target[i] = q_limit_H[i]
            print ("Out of Joint Limit! Target Changed to Limit Value")
    return q_target


def joint_states_cb(joint_states):
    global is_q_init
    if is_q_init is False:
        i = 0
        while i < 4:
            q_init[i] = joint_states.position[i + 2]
            i += 1
        is_q_init = True
    for i in range(4):
        q_cur[i] = joint_states.position[i + 2]


def main():
    rospy.init_node("joint_space_publisher")
    rospy.Subscriber("/open_manipulator/joint_states", JointState, joint_states_cb)
    joint1_command_pub = rospy.Publisher(
        "/open_manipulator/joint1_position/command", Float64, queue_size=3
    )
    joint2_command_pub = rospy.Publisher(
        "/open_manipulator/joint2_position/command", Float64, queue_size=3
    )
    joint3_command_pub = rospy.Publisher(
        "/open_manipulator/joint3_position/command", Float64, queue_size=3
    )
    joint4_command_pub = rospy.Publisher(
        "/open_manipulator/joint4_position/command", Float64, queue_size=3
    )
    joint_command_pub = rospy.Publisher(
        "/open_manipulator/joint_position/command", Float64MultiArray, queue_size=3
    )

    rate = rospy.Rate(100)

    comment = "Type 4 Joint Target Position(degree): " + "ex) 0.0 0.0 0.0 0.0\n"
    if sys.version_info[0] == 2:
        j1_target, j2_target, j3_target, j4_target = [
            float(j_command) for j_command in raw_input(comment).split()
        ]
    elif sys.version_info[0] == 3:
        j1_target, j2_target, j3_target, j4_target = [
            float(j_command) for j_command in input(comment).split()
        ]
    j1_target = j1_target * Deg2Rad
    j2_target = j2_target * Deg2Rad
    j3_target = j3_target * Deg2Rad
    j4_target = j4_target * Deg2Rad
    j_target = np.array([j1_target, j2_target, j3_target, j4_target])
    j_target = joint_limit_check(j_target)

    time_init = rospy.get_rostime().secs + rospy.get_rostime().nsecs * 10 ** -9
    while not rospy.is_shutdown():
        time = rospy.get_rostime().secs + rospy.get_rostime().nsecs * 10 ** -9
        j1 = cubic(time, time_init, time_init + 3.0, q_init[0], j_target[0], 0.0, 0.0)
        j2 = cubic(time, time_init, time_init + 3.0, q_init[1], j_target[1], 0.0, 0.0)
        j3 = cubic(time, time_init, time_init + 3.0, q_init[2], j_target[2], 0.0, 0.0)
        j4 = cubic(time, time_init, time_init + 3.0, q_init[3], j_target[3], 0.0, 0.0)

        joint1_command_pub.publish(j1)
        joint2_command_pub.publish(j2)
        joint3_command_pub.publish(j3)
        joint4_command_pub.publish(j4)
        joint_command_pub.publish(data=np.array([j1, j2, j3, j4]))

        if np.mean(np.abs(j_target - q_cur)) < 0.01:
            break
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
