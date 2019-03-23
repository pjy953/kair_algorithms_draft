ROS_DISTRO=kinetic
CATKIN_WS=/root/catkin_ws
KAIR=$CATKIN_WS/src/kair_algorithms_draft

cd $CATKIN_WS && \
	source /opt/ros/$ROS_DISTRO/setup.bash && \
	source devel/setup.bash && \
	roslaunch open_manipulator_controller open_manipulator_controller.launch use_platform:=false & \
	roslaunch open_manipulator_gazebo open_manipulator_gazebo.launch gui:=false &
	rosrun kair_algorithms test_open_manipulator.py
