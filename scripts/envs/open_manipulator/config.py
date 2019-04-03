import numpy as np

from math import pi
from geometry_msgs.msg import Point, Pose, Quaternion


# Global variables
term_count = 10
suc_count = 10

overhead_orientation = Quaternion(
    x=-0.00142460053167, y=0.999994209902, z=-0.00177030764765, w=0.00253311793936
)

joint_limits = {
    "hi": {
        "j1": pi * 0.9,
        "j2": pi * 0.5,
        "j3": pi * 0.44,
        "j4": pi * 0.65,
        "grip": -0.001,
    },
    "lo": {
        "j1": -pi * 0.9,
        "j2": -pi * 0.57,
        "j3": -pi * 0.3,
        "j4": -pi * 0.57,
        "grip": 0.019,
    },
}

# Global variables
action_dim = 5  # Cartesian
observation_dim = (25, )

# terminal condition
inner_rad = 0.134
outer_rad = 0.3
lower_rad = 0.384
inner_z = 0.321
outer_z = 0.250
lower_z = 0.116

# block generation condition
rand_pose = Pose(
    position=Point(
        x=np.random.uniform(0.1, 0.15),
        y=np.random.uniform(0, 50.6),
        z=np.random.uniform(0.0, 0.1),
    ),
    orientation=overhead_orientation,
)

mode = 'sim'
max_episode_steps = 100
distance_threshold = 0.1

reward_rescale_ratio = 1.0
reward_func = "sparse"
control_mode = "position"
