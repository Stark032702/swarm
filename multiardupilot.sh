	#!/bin/bash 

gnome-terminal --tab --working-directory=/home/burlion/ardupilot/ArduCopter/ -e "sim_vehicle.py -v ArduCopter -f gazebo-drone1 -I0" 
gnome-terminal --tab --working-directory=/home/burlion/ardupilot/ArduCopter/ -e "sim_vehicle.py -v ArduCopter -f gazebo-drone2 -I1" 
gnome-terminal --tab --working-directory=/home/burlion/ardupilot/ArduCopter/ -e "sim_vehicle.py -v ArduCopter -f gazebo-drone3 -I2" 
