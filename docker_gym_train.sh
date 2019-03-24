ROS_DISTRO=kinetic
CATKIN_WS=/root/catkin_ws
KAIR=$CATKIN_WS/src/kair_algorithms_draft

cd $KAIR/scripts; \
   python run_lunarlander_continuous.py --algo sac --off-render
