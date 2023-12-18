#!/bin/bash 

gnome-terminal --tab -- roslaunch iq_sim apm.launch fcu_url:=udp://127.0.0.1:14550@14555 mavros_ns:=/drone1 tgt_system:=1
gnome-terminal --tab -- roslaunch iq_sim apm.launch fcu_url:=udp://127.0.0.1:14560@14565 mavros_ns:=/drone2 tgt_system:=2
gnome-terminal --tab -- roslaunch iq_sim apm.launch fcu_url:=udp://127.0.0.1:14570@14575 mavros_ns:=/drone3 tgt_system:=3