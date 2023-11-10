#!/usr/bin/env python2
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

# Parameters
lx = 1.
wn = 0.5
Ts = 1 / 50 # 0.02
v_max = 2.
k1 = 1
k2 = 0.5
d0_inf = 0.5
vT = 0.25
height = 2.2


## Trajectory to follow
def x_hat(theta): return lx*np.cos(theta)
def dx_hat(theta): return -lx*np.sin(theta)
def y_hat(theta): return lx*np.sin(theta)
def dy_hat(theta): return lx*wn*np.cos(theta)

def norm(vector):
    vector = vector.flatten()
    return np.sqrt(vector[0]**2 + vector[1]**2)

# Spacing of the pattern
def pattern(sp):
    # Reference formation : rij_ = xi_-xj_
    r11_star = np.array([0, 0]).reshape((2, 1))
    r12_star = np.array([sp * np.cos(pi / 6), sp * (1 + np.sin(pi / 6))]).reshape((2, 1))
    r13_star = np.array([-sp * np.cos(pi / 6), sp * (1 + np.sin(pi / 6))]).reshape((2, 1))
    r14_star = np.array([0, sp]).reshape((2, 1))
    r1_star_x = np.array([r11_star[0], r12_star[0], r13_star[0], r14_star[0]])
    r1_star_y = np.array([r11_star[1], r12_star[1], r13_star[1], r14_star[1]])

    r21_star = -r12_star
    r22_star = np.array([0, 0]).reshape((2, 1))
    r23_star = np.array([-2 * sp * np.cos(pi / 6), 0]).reshape((2, 1))
    r24_star = np.array([-sp * np.cos(pi / 6), -sp * np.sin(pi / 6)]).reshape((2, 1))
    r2_star_x = np.array([r21_star[0], r22_star[0], r23_star[0], r24_star[0]])
    r2_star_y = np.array([r21_star[1], r22_star[1], r23_star[1], r24_star[1]])

    r31_star = -r13_star
    r32_star = -r23_star
    r33_star = np.array([0, 0]).reshape((2, 1))
    r34_star = np.array([sp * np.cos(pi / 6), -sp * np.sin(pi / 6)]).reshape((2, 1))
    r3_star_x = np.array([r31_star[0], r32_star[0], r33_star[0], r34_star[0]])
    r3_star_y = np.array([r31_star[1], r32_star[1], r33_star[1], r34_star[1]])

    return r1_star_x, r1_star_y, r2_star_x, r2_star_y, r3_star_x, r3_star_y


# Pattern djacency matrix
A1 = np.ones((4, 4)) - np.eye(4)

# Function that simulates the intermittent data reception
t_without_data = Ts  # Time since the last measurement received
chance = 5.  # %
def data_is_received():
    global t_without_data
    boolean = random.random() < chance / 100
    boolean = True
    if boolean:
        t_without_data = Ts
    else:
        t_without_data += Ts
    return boolean


room_pat = 2.5# room between pattern spacing and interval max_lenght
room_drone_lenght = 0.5  # room between d0_inf and interval max_lenght


def pattern_spacing(l):
    max = 3
    min = 0.2
    return np.min([np.max([l + room_pat, min + room_pat]), max])


"""
High-gain continuous-discrete time observer
"""

# Observer parameters
θ = 2  # inverserly proportional to the maximum delay of measurement
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 0]])
Δθ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1 / θ, 0],
               [0, 0, 0, 1 / θ]])
K0 = np.array([[2, 0],
               [0, 2],
               [1, 0],
               [0, 1]])
B = np.array([[0, 0],
              [0, 0],
              [1, 0],
              [0, 1]])
C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])


def signe(x):
    if x == 0.:
        return 0.
    elif x < 0:
        return -1.
    else:
        return 1.


# The velocity of the drones is bounded
kb = 0.6  # ~ 1/sqrt(2) diagonal max speed


def bound(u):
    if -v_max * kb <= u <= v_max * kb:
        return u
    else:
        return v_max * signe(u) * kb


def sat(xmin, x):
    return np.max([xmin, x])


def H1(X, X0):
    X = X.flatten()
    X0 = X0.flatten()
    D1 = -2 * X0[2] * (X[0] - X0[0]) - 2 * X0[3] * (X[1] - X0[1])
    d02 = (X[0] - X0[0]) ** 2 + (X[1] - X0[1]) ** 2
    h = -D1 - k1 * (d02 - (d0_inf ** 2))
    return h


def H2(X, X0):
    X = X.flatten()
    X0 = X0.flatten()
    D2 = 0
    d02 = (X[0] - X0[0]) ** 2 + (X[1] - X0[1]) ** 2
    h = -D2 + d0_inf * vT - k2 * (d02 - (d0_inf ** 2))
    return h


def M(X, X0):
    X = X.flatten()
    X0 = X0.flatten()
    M0 = np.array([[2 * (X[0] - X0[0]), 2 * (X[1] - X0[1])],
                   [-(X[1] - X0[1]), X[0] - X0[0]]])
    return M0


class MavrosOffboardAttctlTest2(MavrosHelper):
    def __init__(self, drone_id='drone2'):
        """Test offboard attitude control"""
        super().__init__('drone2')
        self.att = PoseStamped()  # for position
        self.vel = TwistStamped()  # for velocity
        self.gps = PoseStamped()  # gps coordinates
        self.global_gps = PoseStamped()
        self.att_setpoint_pub = rospy.Publisher('/drone2/mavros/setpoint_position/global', PoseStamped, queue_size=10)
        self.att_velocity_pub = rospy.Publisher('/drone2/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        self.takeoff_command = rospy.ServiceProxy("/drone2/mavros/cmd/takeoff", CommandTOL)
        self.arming = rospy.ServiceProxy("/drone2/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/drone2/mavros/set_mode", SetMode)

        def state_cb(msg):
            global current_state
            current_state = msg

        self.state_sub = rospy.Subscriber("/drone2/mavros/state", State, callback=state_cb)
        self.gps_sub = rospy.Subscriber('/drone2/mavros/local_position/pose', PoseStamped, self.update_gps)
        self.global_gps_sub = rospy.Subscriber('/drone2/mavros/global_position/local', PoseStamped, self.update_global_gps)
        self.fpv_2_loc = rospy.Subscriber('/drone2/mavros/local_position/pose', PoseStamped, self.update_fpv_2)

        # Subscribe to the positions of the other drones
        self.fpv_1_loc =rospy.Subscriber('/drone1/mavros/global_position/local', PoseStamped, self.update_fpv_1)
        #self.fpv_2_loc = rospy.Subscriber('/vrpn_client_node/FPV_2/pose', PoseStamped, self.update_fpv_2)
        self.fpv_3_loc = rospy.Subscriber('/drone3/mavros/global_position/local', PoseStamped, self.update_fpv_3)

        """
        Subscribe to the positions of other drones 
        """

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
            rospy.Subscriber('/drone2/mavros/local_position/pose', PoseStamped, self.update_gps)  # update the gps cord

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def run(self):

        rospy.loginfo("Run mission")

        timeout = 10  # (int) seconds
        loop_freq = 50  # Hz
        rate = rospy.Rate(loop_freq)
        crossed = False
        r = 1.4
        count = 1
        wn = 0.5
        theta = 0
        kp = 1.5
        rate2 = rospy.Rate(2)
        # for ardupilot takeoff command is required for flying
        last_req = rospy.Time.now()


        # State Variables
        x10 = np.array([[0.], [0.5]])
        x20 = np.array([[-0.4], [-0.2]])
        x30 = np.array([[0.4], [-0.2]])

        x1 = np.zeros((2, 1))
        x2 = np.zeros((2, 1))
        x3 = np.zeros((2, 1))
        x4 = np.array([[x_hat(0.)], [y_hat(0.)]])

        '''
        while not self.state.armed:
            print("Waiting to be armed")
            time.sleep(4)

        while self.gps.pose.position.z <= height - 1:
            print(self.gps.pose.position.z)
            print("take off to height:", height)
            time.sleep(4)
        '''
        self.set_mode("GUIDED", 5)
        time.sleep(5)

        # FPV_2_checker = True
        #
        # while FPV_2_checker:
        #
        #     if self.fpv_2_loc.pose.position.z != height:
        #         print("Waiting for FPV 2 to reach", height, "m")
        #     else:
        #         FPV_2_checker = False

        # setting drone at initial position
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
        while not (bool_start):
            self.att.pose.position.y = x20[1, 0]
            self.att.pose.position.x = x20[0, 0]
            self.att.pose.position.z = height
            self.att_setpoint_pub.publish(self.att)
            print("Waiting for other drones")
            # x1[0, 0] = self.fpv_1_loc.pose.position.x
            # x1[1, 0] = self.fpv_1_loc.pose.position.y
            x2[0, 0] = self.fpv_2_loc.position.x
            x2[1, 0] = self.fpv_2_loc.position.y
            x3[0, 0] = self.fpv_3_loc.position.x
            x3[1, 0] = self.fpv_3_loc.position.y
            if  norm(x2 - x20) < eps and norm(x3 - x30): #norm(x1 - x10) < eps and
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

        # Wait the other drones to be ready
        """
        rospy.wait_for_service(uav2 + "/vrpn_client_node/FPV_2/pose")
        rospy.wait_for_service(uav3 + "/vrpn_client_node/FPV_2/pose")
        """

        """
        High-gain observor
        """
        # Estimated vector position and speed
        p1_hat_est = x1.copy()
        p2_hat_est = x2.copy()
        p3_hat_est = x3.copy()
        x1_hat_est = np.vstack((p1_hat_est, np.zeros((2, 1))))
        x2_hat_est = np.vstack((p2_hat_est, np.zeros((2, 1))))
        x3_hat_est = np.vstack((p3_hat_est, np.zeros((2, 1))))

        # Measurements
        y1 = np.zeros((2, 1))  # measurement of pj only
        y2 = np.zeros((2, 1))  # measurement of pj only
        y3 = np.zeros((2, 1))  # measurement of pj only

        """
        Interval estimation of the other drones (x1's point of view)
        """
        x1int = IntervalVector([[x1[0, 0], x1[0, 0]], [x1[1, 0], x1[1, 0]]])
        x2int = IntervalVector([[x2[0, 0], x2[0, 0]], [x2[1, 0], x2[1, 0]]])
        x3int = IntervalVector([[x3[0, 0], x3[0, 0]], [x3[1, 0], x3[1, 0]]])
        c1 = np.array(x1int.mid()).reshape((2, 1))
        c2 = np.array(x2int.mid()).reshape((2, 1))
        c3 = np.array(x3int.mid()).reshape((2, 1))
        speed = IntervalVector([[-v_max, v_max], [-v_max, v_max]])  # bounded speed

        # Init variables
        spacing = 1.
        t_last = 0.  # Last data received
        t0 = time.time()  # mission time

        while 0.0 <= theta <= 14 * pi:
            t = time.time() - t0
            t_last2 = time.time() - t0
            # At every instant, if no data is received the interval estimation inflates
            x1int = x1int + speed * Ts
            x3int = x3int + speed * Ts

            # x2 knows at every instant where he is
            p2_hat_est = x2_hat_est[0:2, 0].reshape((2, 1))
            y2 = x2
            x2int = IntervalVector([[y2[0, 0], y2[0, 0]], [y2[1, 0], y2[1, 0]]]).inflate(0.2)
            c2 = np.array(x2int.mid()).reshape((2, 1))
            l2 = x2int.max_diam() / 2

            # xi(t)
            x1[0, 0] = self.fpv_1_loc.pose.position.x
            x1[1, 0] = self.fpv_1_loc.pose.position.y
            #x1[0, 0] = self.fpv_3_loc.pose.position.x
            #x1[1, 0] = self.fpv_3_loc.pose.position.y
            x2[0, 0] = x_hat(theta)
            x2[1, 0] = y_hat(theta) + spacing
            x3[0, 0] = self.fpv_3_loc.pose.position.x
            x3[1, 0] = self.fpv_3_loc.pose.position.y
            # x3[0, 0] = x_hat(theta) + spacing * np.cos(pi / 6)
            # x3[1, 0] = y_hat(theta) - spacing * np.sin(pi / 6)

            # Measurement received
            if data_is_received():  # not(lost_connection()) :
                # last data received
                t_last = t

                # pij(kij(t))
                y1 = x1
                y3 = x3

                # pij_hat(kij(t))
                p1_hat_est = x1_hat_est[0:2, 0].reshape((2, 1))
                p3_hat_est = x3_hat_est[0:2, 0].reshape((2, 1))

                # Intervals
                x1int = IntervalVector([[y1[0, 0], y1[0, 0]], [y1[1, 0], y1[1, 0]]]).inflate(0.2)
                x3int = IntervalVector([[y3[0, 0], y3[0, 0]], [y3[1, 0], y3[1, 0]]]).inflate(0.2)
                c1 = np.array(x1int.mid()).reshape((2, 1))
                c3 = np.array(x3int.mid()).reshape((2, 1))

            # Update of the estimation
            x1_hat_est_dot = A @ x1_hat_est - θ * np.linalg.inv(Δθ) @ K0 @ (np.exp(-2 * θ * (t - t_last)) * (p1_hat_est - y1))
            x2_hat_est_dot = A @ x2_hat_est - θ * np.linalg.inv(Δθ) @ K0 @ (np.exp(-2 * θ * (t - t_last2)) * (p2_hat_est - y2))
            x3_hat_est_dot = A @ x3_hat_est - θ * np.linalg.inv(Δθ) @ K0 @ (np.exp(-2 * θ * (t - t_last)) * (p3_hat_est - y3))
            x1_hat_est = x1_hat_est + Ts * x1_hat_est_dot
            x2_hat_est = x2_hat_est + Ts * x2_hat_est_dot
            x3_hat_est = x3_hat_est + Ts * x3_hat_est_dot

            # Time-varying pattern
            l1 = x1int.max_diam() / 2
            l3 = x3int.max_diam() / 2
            new_spacing = pattern_spacing(l3)
            # The triangle must not shrink to fast
            if new_spacing > spacing:
                spacing = new_spacing
            else:
                spacing += -0.05 / 2

            # Formation flying
            r1_star_x, r1_star_y, r2_star_x, r2_star_y, r3_star_x, r3_star_y = pattern(spacing)
            X = np.array([x1_hat_est[0, 0], x2_hat_est[0, 0], x3_hat_est[0, 0], x4[0, -1]]).reshape((4, 1))
            Y = np.array([x1_hat_est[1, 0], x2_hat_est[1, 0], x3_hat_est[1, 0], x4[1, -1]]).reshape((4, 1))
            ux_2 = -kp * A1[1, :] @ (x2_hat_est[0, 0] - X - r2_star_x) + dx_hat(t)
            uy_2 = -kp * A1[1, :] @ (x2_hat_est[1, 0] - Y - r2_star_y) + dy_hat(t)

            """
            Collision avoidance
            """
            d01_, d02_, d03_ = [], [], []
            ## Drone 2
            X2 = np.vstack((x2[:, -1], x2_hat_est[2:4, 0])).reshape((4, 1))  # x2 knows where he is
            # /drone 1
            M0 = M(X2, c1)
            satx = (M0 @ np.array([[ux_2[0]], [uy_2[0]]]))[0, 0]
            saty = (M0 @ np.array([[ux_2[0]], [uy_2[0]]]))[1, 0]
            h1 = H1(X2, x1_hat_est)
            h2 = H2(X2, x1_hat_est)
            SAT = np.array([[sat(h1, satx)],
                            [sat(h2, saty)]])
            u2_sat = np.linalg.inv(M0) @ SAT
            ux_2, uy_2 = u2_sat[0, 0], u2_sat[1, 0]

            # /drone 3
            d = np.sqrt((x3[0, -1] - x2[0, -1]) ** 2 + (x3[1, -1] - x2[1, -1]) ** 2)
            d02_.append(d)
            d03_.append(d)
            M0 = M(X2, c3)
            satx = (M0 @ np.array([[ux_2], [uy_2]]))[0, 0]
            saty = (M0 @ np.array([[ux_2], [uy_2]]))[1, 0]
            h1 = H1(X2, x3_hat_est)
            h2 = H2(X2, x3_hat_est)
            SAT = np.array([[sat(h1, satx)],
                            [sat(h2, saty)]])
            u2_sat = np.linalg.inv(M0) @ SAT
            ux_2, uy_2 = u2_sat[0, 0], u2_sat[1, 0]
            #d02.append(np.min(d02_))

            # Bound the speed
            ux_2 = bound(ux_2)
            uy_2 = bound(uy_2)
            x1_est_x = (x1_hat_est[0,0] + x1_hat_est[1,0]) * 0.5 
            x1_est_y = (x1_hat_est[2,0] + x1_hat_est[3,0]) * 0.5
            pos_x2 = np.array([x1_est_x, x1_est_y]).reshape((2,1)) + np.array([spacing * np.cos(pi / 6), spacing * (1 + np.sin(pi / 6))]).reshape((2, 1))
            # Sending commands
            self.vel.twist.linear.x = ux_2
            self.vel.twist.linear.y = uy_2
            self.att.pose.position.x = pos_x2[0,0]
            self.att.pose.position.y = pos_x2[1,0] 
            self.att.pose.position.z = 2.5
            self.att_setpoint_pub.publish(self.att)
            #time.sleep(0.1)
            count += 1
            wn += 0.001
            theta = wn * count * 0.01
            print(theta)
            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

            # Update the leader
            x4 = np.array([[x_hat(theta)], [y_hat(theta)]])

        self.set_mode("LAND", 5)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 90, 0)
        self.set_arm(False, 5)
        self.tearDown()


if __name__ == '__main__':
    rospy.init_node('control_node2', anonymous=True)
    control = MavrosOffboardAttctlTest2()
    control.run()