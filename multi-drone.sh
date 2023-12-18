#!/bin/bash 

gnome-terminal --tab --working-directory=/home/burlion/multi_drone2/first_order/gazebo/ -- python drone1.py

gnome-terminal --tab --working-directory=/home/burlion/multi_drone2/first_order/gazebo/ -- python drone2.py

gnome-terminal --tab --working-directory=/home/burlion/multi_drone2/first_order/realdrones/ -- python onboard_drone3.py