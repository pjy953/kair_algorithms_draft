import numpy as np

# DH matrix
# Blist: The joint screw axes in the end-effector frame when the
# manipulator is at the home position,


Blist = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 0.278, 0.0],
        [0.0, 1.0, 0.0, 0.1281, 0.0, -0.278],
        [0.0, 1.0, 0.0, 0.0, 0.0, -0.254],
        [0.0, 1.0, 0.0, 0.0, 0.0, -0.13],
    ]
)
Blist = Blist.T

# Homogen TF matrix from base to robot's end-effector

# M: The home configuration (position and orientation) of the
#         end-effector
# Should make it automated
# use tf.transformation to change from ee_pose + quaternion to homogen tf


M = np.array(
    [
        [1.0, 0.0, 0.0, 0.29],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.20305],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

Slist = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, -0.012, 0.0],
        [0.0, 1.0, 0.0, -0.07495, 0.0, 0.012],
        [0.0, 1.0, 0.0, -0.20305, 0.0, 0.036],
        [0.0, 1.0, 0.0, -0.20305, 0.0, 0.16],
    ]
)
Slist = Slist.T
