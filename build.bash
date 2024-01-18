#!/bin/bash

source /opt/ros/humble/setup.bash

colcon build --packages-select tb3_op
colcon build --packages-select tb3_sq
colcon build --packages-select tb3_learning_py

#colcon build --packages-select hakoniwa_turtlebot3 hakoniwa_turtlebot3_rviz hakoniwa_turtlebot3_description
