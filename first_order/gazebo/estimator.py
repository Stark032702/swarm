from __future__ import division

import rospy
from geometry_msgs.msg import Quaternion, Vector3, PoseStamped, TwistStamped
from mavros_msgs.srv import CommandTOL, CommandBool, SetModeRequest, CommandBoolRequest, SetMode
from geographic_msgs.msg import GeoPoseStamped
from mavros_msgs.msg import State
from control_helper import MavrosHelper
from pymavlink import mavutil

from threading import Thread
from math import pi
import time
import numpy as np
from onboard_codac import *
import random

class Estimator: 

    def __init__(self, drones, lows, highs, vel_lows, vel_highs): 
        self.drones = drones
        self.highs = highs 
        self.lows = lows
        self.drone_pos_1 = self
