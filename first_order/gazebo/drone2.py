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
    dx = -0.2 * r * np.sin(t)
    y = r * np.sin(t)
    dy= 0.2* r * np.cos(t)
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
        self.att_global = GeoPoseStamped()
        self.vel = TwistStamped()  # for velocity
        self.true_vel = TwistStamped()
        self.gps = PoseStamped()  # gps coordinates
        self.fpv_1_loc = PoseStamped()
        self.fpv_1_vel = TwistStamped()
        self.fpv_3_loc = PoseStamped()
        self.fpv_3_vel = TwistStamped()
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
        self.vel_sub = rospy.Subscriber('/drone2/mavros/local_position/velocity', TwistStamped,self.update_vel)

        self.fpv_1_loc_sub = rospy.Subscriber('/drone1/mavros/local_position/pose', PoseStamped, self.update_fpv_1)
        self.fpv_1_vel_sub = rospy.Subscriber('/drone1/mavros/local_position/velocity', TwistStamped, self.update_vel_fpv1 )
        self.fpv_3_loc_sub = rospy.Subscriber('/drone3/mavros/local_position/pose', PoseStamped, self.update_fpv_3)
        self.fpv_3_vel_sub = rospy.Subscriber('/drone3/mavros/local_position/velocity', TwistStamped, self.update_vel_fpv3 )

        """
        Subscribe to the positions of other drones 
        """

        self.x1_hat = Interval2Vector([0,0],[0,0])
        self.x2_hat = Interval2Vector([0,0], [0,0])
        self.x3_hat = Interval2Vector([0,0], [0,0])
        
        self.v1_hat = Interval2Vector([0,0], [0,0])
        self.v2_hat = Interval2Vector([0,0], [0,0])
        self.v3_hat = Interval2Vector([0,0], [0,0])
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
    def update_vel(self, data):
        self.true_vel = data
    
    def update_vel_fpv1(self, data): 
        self.fpv_1_vel = data 
    
    def update_vel_fpv3(self, data): 
        self.fpv_3_vel = data 

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

    def inflate(self, t, tlast,rate, drone, flag): 
        
        if (t-tlast) < 1/rate or flag: 
            if drone == 1: 
                self.x1_hat = self.x1_hat + self.v1_hat * (t-tlast)
                self.x1_hat.inflate(0.05)
            elif drone == 2: 
                self.x2_hat = self.x2_hat + self.v2_hat * (t-tlast)
                self.x2_hat.inflate(0.05)
            elif drone == 3: 
                self.x3_hat = self.x3_hat + self.v3_hat * (t-tlast)
                self.x3_hat.inflate(0.05)
        else: 
            if drone == 1: 
                self.x1_hat = Interval2Vector([self.fpv_1_loc.pose.position.x,self.fpv_1_loc.pose.position.x], [self.fpv_1_loc.pose.position.y,self.fpv_1_loc.pose.position.y])
                self.v1_hat = Interval2Vector([self.fpv_1_vel.twist.linear.x,self.fpv_1_vel.twist.linear.x], [self.fpv_1_vel.twist.linear.y,self.fpv_1_vel.twist.linear.y])
            elif drone == 2: 
                self.x2_hat = Interval2Vector([self.gps.pose.position.x, self.gps.pose.position.x], [self.gps.pose.position.y ,self.gps.pose.position.y])
                self.v2_hat = Interval2Vector([self.true_vel.twist.linear.x, self.true_vel.twist.linear.x], [self.true_vel.twist.linear.y, self.true_vel.twist.linear.y])
            elif drone == 3: 
                self.x3_hat = Interval2Vector([self.fpv_3_loc.pose.position.x, self.fpv_3_loc.pose.position.x], [self.fpv_3_loc.pose.position.y,self.fpv_3_loc.pose.position.y])
                self.v3_hat = Interval2Vector([self.fpv_3_vel.twist.linear.x, self.fpv_3_vel.twist.linear.x ], [self.fpv_3_vel.twist.linear.y, self.fpv_3_vel.twist.linear.y])
                
    def run(self, d): 
        rospy.loginfo("Run mission")

        timeout = 10  # (int) seconds
        loop_freq = 25 # Hz
        location_rate = 10
        rate = rospy.Rate(loop_freq)
        crossed = False
        
        height = 2
        rate2 = rospy.Rate(2)
        # for ardupilot takeoff command is required for flying
        last_req = rospy.Time.now()

        r1_star_x, r1_star_y, r2_star_x, r2_star_y, r3_star_x, r3_star_y = formation(1)
        

        x10 = np.array([[0.0], [0.0]])
        x20 = np.array([[0], [0 ]])
        x30 = np.array([[0], [0]])
        
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
        eps = 2
        
        while not(bool_start):
            self.att.pose.position.y = x20[1,0]
            self.att.pose.position.x = x20[0,0]
            self.att.pose.position.z = height
            self.att_setpoint_pub.publish(self.att)
            print("Waiting for other drones")
            x1[0, 0] = self.fpv_1_loc.pose.position.x
            x1[1, 0] = self.fpv_1_loc.pose.position.y
            if np.abs(self.att.pose.position.x - x20[0,0]) < eps and np.abs(self.fpv_1_loc.pose.position.x -x10[0,0]) < eps:
                #and norm(x3 - x30) :
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
        
        time.sleep(4)
        t0 = time.time()  # mission time
        t = time.time() - t0 
        tlast = 0
        r = 6+np.abs(r1_star_x[1,0])
        count = 1
        wn = 0.5
        theta = 0.0
        kp = 0.85
        disconnected = False
        while 0.0 <= theta <= 10 * np.pi:
            t = time.time() - t0 

            self.inflate(t, tlast, location_rate, 1)
            self.inflate(t, tlast, location_rate, 2)
            self.inflate(t, tlast, location_rate, 3)

            x1 = self.x1_hat.mid()
            theta2 = np.arctan2(x1[1, 0], x1[0,0])
            x,dx,y,dy = circle_trajectory(r, theta)
            
            total_uncertainty = self.x1_hat.uncertainty() + self.x2_hat.uncertainty() + self.x3_hat.uncertainty() 
            r1_star_x, r1_star_y, r2_star_x, r2_star_y, r3_star_x, r3_star_y = formation(3.5 + 0.33 *total_uncertainty)
            x_true = x1[0,0] + (r1_star_x[1,0] * np.cos(theta2)) - r1_star_y[1,0] * np.sin(theta2)
            y_true = x1[1,0] + (r1_star_x[1,0] * np.sin(theta2)) + r1_star_y[1,0] * np.cos(theta2) + 5
            
            self.vel.twist.linear.x  = -kp * (self.x2[0,0] - x_true)  
            self.vel.twist.linear.y  = -kp * (self.x2[1,0] - y_true) 
            self.vel.twist.linear.z = 0 
            
            self.att_velocity_pub.publish(self.vel)
            print(theta, self.vel.twist.linear.x, self.vel.twist.linear.y)
            count += 1
            theta = wn * count * 0.01 + np.pi
            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)
            tlast = t
        self.set_mode("LAND", 5)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 90, 0)
        self.set_arm(False, 5)
        self.tearDown()

if __name__ == '__main__':
    rospy.init_node('control_node', anonymous=True)
    control = MavrosOffboardAttctlTest2()
    control.run(0.8)