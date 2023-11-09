from __future__ import division

import rospy
from geometry_msgs.msg import Quaternion, Vector3, PoseStamped, TwistStamped
from mavros_msgs.srv import CommandTOL, CommandBool, SetModeRequest, CommandBoolRequest, SetMode
from geographic_msgs.msg import GeoPoseStamped
from mavros_msgs.msg import State
from control_helper import MavrosHelper
from pymavlink import mavutil
# from std_msgs.msg import Header
from threading import Thread
# from tf.transformations import quaternion_from_euler
from math import pi
import time
import numpy as np
from onboard_codac import *
import random


def circle_trajectory(r, t): 
    x = r * np.cos(t)
    dx = -r * np.sin(t)
    y = r * np.sin(t)
    dy= r * np.cos(t)
    return x, dx, y, dy

def formation(d): 
    r11_star = np.array([0, 0]).reshape((2, 1))
    r12_star = np.array([d* np.cos(pi / 6), d* (1 + np.sin(pi / 6))]).reshape((2, 1))
    r13_star = np.array([-d* np.cos(pi / 6), d* (1 + np.sin(pi / 6))]).reshape((2, 1))
    r14_star = np.array([0, d]).reshape((2, 1))
    r1_star_x = np.array([r11_star[0], r12_star[0], r13_star[0], r14_star[0]])
    r1_star_y = np.array([r11_star[1], r12_star[1], r13_star[1], r14_star[1]])

    r21_star = -r12_star
    r22_star = np.array([0, 0]).reshape((2, 1))
    r23_star = np.array([-2 * d* np.cos(pi / 6), 0]).reshape((2, 1))
    r24_star = np.array([-d* np.cos(pi / 6), -d* np.sin(pi / 6)]).reshape((2, 1))
    r2_star_x = np.array([r21_star[0], r22_star[0], r23_star[0], r24_star[0]])
    r2_star_y = np.array([r21_star[1], r22_star[1], r23_star[1], r24_star[1]])

    r31_star = -r13_star
    r32_star = -r23_star
    r33_star = np.array([0, 0]).reshape((2, 1))
    r34_star = np.array([d* np.cos(pi / 6), -d* np.sin(pi / 6)]).reshape((2, 1))
    r3_star_x = np.array([r31_star[0], r32_star[0], r33_star[0], r34_star[0]])
    r3_star_y = np.array([r31_star[1], r32_star[1], r33_star[1], r34_star[1]])

    return r1_star_x, r1_star_y, r2_star_x, r2_star_y, r3_star_x, r3_star_y

class MavrosOffboardAttctlTest2(MavrosHelper):
    def __init__(self, drone_id='drone2'):
        """Test offboard attitude control"""
        super().__init__('drone2')
        self.att = PoseStamped()  # for position
        self.vel = TwistStamped()  # for velocity
        self.gps = PoseStamped()  # gps coordinates
        self.fpv_1_loc = PoseStamped()
        self.fpv_3_loc = PoseStamped()
        self.att_setpoint_pub = rospy.Publisher('/drone2/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.att_velocity_pub = rospy.Publisher('/drone2/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        self.takeoff_command = rospy.ServiceProxy("/drone2/mavros/cmd/takeoff", CommandTOL)
        self.arming = rospy.ServiceProxy("/drone2/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/drone2/mavros/set_mode", SetMode)

        def state_cb(msg):
            global current_state
            current_state = msg

        self.state_sub = rospy.Subscriber("/drone2/mavros/state", State, callback=state_cb)
        self.gps_sub = rospy.Subscriber('/drone2/mavros/local_position/pose', PoseStamped, self.update_gps)
        
        self.fpv_1_loc_sub =rospy.Subscriber('/drone1/mavros/local_position/pose', PoseStamped, self.update_fpv_1)
        self.fpv_3_loc_sub = rospy.Subscriber('/drone3/mavros/local_position/pose', PoseStamped, self.update_fpv_3)

        """
        Subscribe to the positions of other drones 
        """

        self.x2 = np.zeros((2, 1))
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 10, -1)

        time.sleep(4)

    def state_cb(msg):
        global current_state
        current_state = msg

    def update_fpv_1(self, data):  # update the location of fpv 1
        self.fpv_1_loc = data

    def update_fpv_2(self, data):  # update the location of fpv 2
        self.fpv_2_loc = data

    def update_fpv_3(self, data):  # update the location of fpv 3
        self.fpv_3_loc = data

    def update_gps(self, data):
        self.gps = data
        # print(data)
    
    def update_global_gps(self, data): 
        self.global_gps = data 

    def tearDown(self):
        super(MavrosOffboardAttctlTest2, self).tearDown()

    def send_att(self):
        rate = rospy.Rate(10)  # Hz


        while not rospy.is_shutdown():
            self.att.header.stamp = rospy.Time.now()
            self.att_setpoint_pub.publish(self.att)
            self.att_velocity_pub.publish(self.vel)
            rospy.Subscriber('/drone1/mavros/local_position/pose', PoseStamped, self.update_gps)  # update the gps cord

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def run(self, d): 
        rospy.loginfo("Run mission")

        timeout = 10  # (int) seconds
        loop_freq = 50  # Hz
        rate = rospy.Rate(loop_freq)
        crossed = False
        r = 10
        count = 1
        wn = 0
        theta = 0
        kp = 1.5
        height = 2
        rate2 = rospy.Rate(2)
        # for ardupilot takeoff command is required for flying
        last_req = rospy.Time.now()

        x10 = np.array([[0.0], [0.0]])
        x20 = np.array([[0.0], [0.0]])
        x30 = np.array([[-1.0], [-2.0]])

        x1 = np.zeros((2, 1))
        x3 = np.zeros((2,1))
        
        self.set_mode("GUIDED", 5)
        time.sleep(5)
        self.arming.call(True)
        time.sleep(5)
        while self.gps.pose.position.z <= height - 1:
            print(self.gps.pose.position.z)
            if (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                self.arming.call(True)
                last_req = rospy.Time.now()
            rate2.sleep()
            self.takeoff_command(0, 0, 0, 0, height)

            print("sending takeoff command")

        bool_start = False
        eps = 1
        
        while not(bool_start):
            self.att.pose.position.y = x20[0,0]
            self.att.pose.position.x = x20[1,0] 
            self.att.pose.position.z = height
            self.att_setpoint_pub.publish(self.att)
            print("Waiting for other drones")
            x1[0, 0] = self.fpv_1_loc.pose.position.x
            x1[1, 0] = self.fpv_1_loc.pose.position.y
            self.x2[0, 0] = self.gps.pose.position.x
            self.x2[1, 0] = self.gps.pose.position.y
            if np.abs(self.att.pose.position.x - x20[1,0]) < eps and np.abs(self.fpv_1_loc.pose.position.x - x10[1,0]) < eps :#and norm(x3 - x30) :
                bool_start = True
                t0_wait = time.time()
                while time.time() - t0_wait < 2.:
                    self.att.pose.position.y = x20[1, 0]
                    self.att.pose.position.x = x20[0, 0]
                    self.att.pose.position.z = height
                    self.att_setpoint_pub.publish(self.att)
            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)
        t0 = time.time()  # mission time
        t = time.time() - t0

        theta=0
        while 0 <= theta <= 8 * np.pi:
            t = time.time() - t0 
            x,dx,y,dy = circle_trajectory(r, t)
            x1[0,0] = self.fpv_1_loc.pose.position.x
            x1[1,0] = self.fpv_1_loc.pose.position.y
            r1_star_x, r1_star_y, r2_star_x, r2_star_y, r3_star_x, r3_star_y = formation(d)
            count += 1
            wn += 0.002
            theta = wn * count * 0.01
            self.att.pose.position.x = 2 * ((x1[0,0])) - 5 
            self.att.pose.position.y =2 * (x1[1,0]) 
            print(theta, self.att.pose.position.x, self.att.pose.position.y)
            self.att.pose.position.z = 2.5

            self.att_setpoint_pub.publish(self.att)
            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)
if __name__ == '__main__':
    rospy.init_node('control_node', anonymous=True)
    control = MavrosOffboardAttctlTest2()
    control.run(0.8)