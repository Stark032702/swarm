import numpy as np
import matplotlib.pyplot as plt
from codac import *
from tqdm import tqdm
from scipy.signal import place_poles
from numpy.linalg import inv
import time
import random

N = 4    # Number of robots
Ts = 0.1 # Sampling period
to = 0.2
tau = Interval(0,to) # Delay
noise = 0.05 # Noise of the interval measurements


## We define N mobile robots
# for i between 1 and N
# xi = [Pos_xi ; Pos_yi]
# xidot = [ux_i ; uy_i] (so there are two control inputs)

# Dislay
fig = plt.figure(1)
ax1 = fig.add_subplot(211)

"""
High-gain continuous-discrete time observer
"""
# Random noise for the initial estimation
eps = 0.2
nu = 1

# Observer parameters
θ = 1
A = np.array([[0,0,1,0],
              [0,0,0,1],
              [0,0,0,0],
              [0,0,0,0]])
Δθ = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1/θ,0],
               [0,0,0,1/θ]])
K0 = np.array([[2,0],
               [0,2],
               [1,0],
               [0,1]])
B = np.array([[0,0],
              [0,0],
              [1,0],
              [0,1]])
C = np.array([[1,0,0,0],
              [0,1,0,0]])

# Function that simulates the delay for the classical observer
def data_is_received():
    boolean = random.choice([True, True, False])
    return boolean


def without_reference():
    """
    Part I - DISPLACEMENT based formation without any reference trajectory
    to follow (we 'just' reach a consensus)
     """

    # Initialization
    x1 = np.array([[-6], [-4]])  # cyan
    x2 = np.array([[-5], [4]])  # green
    x3 = np.array([[0], [4]])  # blue
    x4 = np.array([[8], [3]])  # red

    ## High-gain continuous-discrete time observer
    # Estimated vector position and speed
    p1_hat_est = x1.copy() + eps*np.random.normal(0, nu, size=(2, 1))
    p2_hat_est = x2.copy() + eps*np.random.normal(0, nu, size=(2, 1))
    p3_hat_est = x3.copy() + eps*np.random.normal(0, nu, size=(2, 1))
    p4_hat_est = x4.copy() + eps*np.random.normal(0, nu, size=(2, 1))
    v1_hat_est = np.zeros((2, 1)) + eps*np.random.normal(0, nu, size=(2, 1))
    v2_hat_est = np.zeros((2, 1)) + eps*np.random.normal(0, nu, size=(2, 1))
    v3_hat_est = np.zeros((2, 1)) + eps*np.random.normal(0, nu, size=(2, 1))
    v4_hat_est = np.zeros((2, 1)) + eps*np.random.normal(0, nu, size=(2, 1))
    estimator1 = p1_hat_est.copy()
    estimator2 = p2_hat_est.copy()
    estimator3 = p3_hat_est.copy()
    estimator4 = p4_hat_est.copy()

    x1_hat_est = np.vstack((p1_hat_est, v1_hat_est))
    x2_hat_est = np.vstack((p2_hat_est, v2_hat_est))
    x3_hat_est = np.vstack((p3_hat_est, v3_hat_est))
    x4_hat_est = np.vstack((p4_hat_est, v4_hat_est))

    # Estimated measurement
    y1 = np.zeros((2, 1))  # measurement of pj only
    y2 = np.zeros((2, 1))  # measurement of pj only
    y3 = np.zeros((2, 1))  # measurement of pj only
    y4 = np.zeros((2, 1))  # measurement of pj only

    # Intervals
    x10 = IntervalVector([[-6, -6], [-4, -4]]).inflate(0.1)
    x20 = IntervalVector([[-5, -5], [4, 4]]).inflate(0.1)
    x30 = IntervalVector([[0, 0], [4, 4]]).inflate(0.1)
    x40 = IntervalVector([[8, 8], [3, 3]]).inflate(0.1)

    interval_estimator1 = np.array([[x10[0].mid()],[x10[1].mid()]])
    interval_estimator2 = np.array([[x20[0].mid()],[x20[1].mid()]])
    interval_estimator3 = np.array([[x30[0].mid()],[x30[1].mid()]])
    interval_estimator4 = np.array([[x40[0].mid()],[x40[1].mid()]])

    # Trajectories
    tx1 = TrajectoryVector(2)
    tx1.set([x1[0, 0], x1[1, 0]], 0.)
    tx2 = TrajectoryVector(2)
    tx2.set([x2[0, 0], x2[1, 0]], 0.)
    tx3 = TrajectoryVector(2)
    tx3.set([x3[0, 0], x3[1, 0]], 0.)
    tx4 = TrajectoryVector(2)
    tx4.set([x4[0, 0], x4[1, 0]], 0.)

    # Reference formation :   rij_ = xi_-xj_
    r11_star = np.array([[0], [0]])
    r12_star = -np.array([[1], [-1]])
    r13_star = -np.array([[0], [-1]])
    r14_star = -np.array([[1], [0]])
    r1_star_x = np.array([r11_star[0], r12_star[0], r13_star[0], r14_star[0]])
    r1_star_y = np.array([r11_star[1], r12_star[1], r13_star[1], r14_star[1]])

    r21_star = -r12_star
    r22_star = np.array([[0], [0]])
    r23_star = -np.array([[-1], [0]])
    r24_star = -np.array([[0], [1]])
    r2_star_x = np.array([r21_star[0], r22_star[0], r23_star[0], r24_star[0]])
    r2_star_y = np.array([r21_star[1], r22_star[1], r23_star[1], r24_star[1]])

    r31_star = -r13_star
    r32_star = -r23_star
    r33_star = np.array([[0], [0]])
    r34_star = -np.array([[1], [1]])
    r3_star_x = np.array([r31_star[0], r32_star[0], r33_star[0], r34_star[0]])
    r3_star_y = np.array([r31_star[1], r32_star[1], r33_star[1], r34_star[1]])

    r41_star = -r14_star
    r42_star = -r24_star
    r43_star = -r34_star
    r44_star = np.array([[0], [0]])
    r4_star_x = np.array([r41_star[0], r42_star[0], r43_star[0], r44_star[0]])
    r4_star_y = np.array([r41_star[1], r42_star[1], r43_star[1], r44_star[1]])

    # Adjacency matrix
    A1 = np.ones((4, 4)) - np.eye(4)

    # Laplacian matrix and its second eigenvalue Lambda_2
    L1 = -A1 + 4 * np.eye(4)

    tf = 500 * Ts
    tdomain = Interval(0, tf)

    # Bounded speed
    v = 3
    speed = TubeVector(tdomain, Ts, IntervalVector([[-v, v], [-v, v]]))

    # Tubes estimated by x1
    x2_hat = IntervalVector([[-5, -5], [4, 4]]).inflate(0.01)
    x3_hat = IntervalVector([[0, 0], [4, 4]]).inflate(0.01)
    x4_hat = IntervalVector([[8, 8], [3, 3]]).inflate(0.01)

    # Measurements
    y2_tube = TubeVector(tdomain, Ts, IntervalVector(2))
    y3_tube = TubeVector(tdomain, Ts, IntervalVector(2))
    y4_tube = TubeVector(tdomain, Ts, IntervalVector(2))

    # Estimation
    x1_tube = TubeVector(tdomain, Ts, IntervalVector(2))
    x2_tube = TubeVector(tdomain, Ts, IntervalVector(2))
    x3_tube = TubeVector(tdomain, Ts, IntervalVector(2))
    x4_tube = TubeVector(tdomain, Ts, IntervalVector(2))

    # Tuning gain
    kp = 0.1

    t_last = 0.
    for t in tqdm(np.arange(Ts,50+Ts,Ts),ncols=100):
        tk = Interval(t-Ts/4,t+Ts/4)

        X = np.array([x1[0, -1], x2[0, -1], x3[0, -1], x4[0, -1]]).reshape((4,1))
        Y = np.array([x1[1, -1], x2[1, -1], x3[1, -1], x4[1, -1]]).reshape((4,1))

        # Control law : ui = -kp*Σ(aij*((xi-xj) - rij_) = -kp*Σ(aij*εij)
        ux_1 = -kp * A1[0, :] @(x1[0,-1] - X - r1_star_x)
        uy_1 = -kp * A1[0, :] @(x1[1,-1] - Y - r1_star_y)

        ux_2 = -kp * A1[1, :] @(x2[0,-1] - X - r2_star_x)
        uy_2 = -kp * A1[1, :] @(x2[1,-1] - Y - r2_star_y)

        ux_3 = -kp * A1[2, :] @(x3[0, -1] - X - r3_star_x)
        uy_3 = -kp * A1[2, :] @(x3[1, -1] - Y - r3_star_y)

        ux_4 = -kp * A1[3, :] @(x4[0, -1] - X - r4_star_x)
        uy_4 = -kp * A1[3, :] @(x4[1, -1] - Y - r4_star_y)


        # Trajectory
        x1 = np.hstack((x1,x1[:,-1].reshape(2,1)+Ts*np.vstack((ux_1,uy_1))))
        x2 = np.hstack((x2,x2[:,-1].reshape(2,1)+Ts*np.vstack((ux_2,uy_2))))
        x3 = np.hstack((x3,x3[:,-1].reshape(2,1)+Ts*np.vstack((ux_3,uy_3))))
        x4 = np.hstack((x4,x4[:,-1].reshape(2,1)+Ts*np.vstack((ux_4,uy_4))))

        ## High-gain continuous-discrete time observer
        # Measurement received
        if data_is_received():
            t_last = t

            # pij(kij(t)) / Noised measurement
            y1 = x1[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            y2 = x2[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            y3 = x3[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            y4 = x4[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)

            # pij_hat(kij(t))
            p1_hat_est = x1_hat_est[0:2, 0].reshape((2, 1))
            p2_hat_est = x2_hat_est[0:2, 0].reshape((2, 1))
            p3_hat_est = x3_hat_est[0:2, 0].reshape((2, 1))
            p4_hat_est = x4_hat_est[0:2, 0].reshape((2, 1))

        # Update of the estimation
        x1_hat_est_dot = A @ x1_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p1_hat_est - y1))
        x2_hat_est_dot = A @ x2_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p2_hat_est - y2))
        x3_hat_est_dot = A @ x3_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p3_hat_est - y3))
        x4_hat_est_dot = A @ x4_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p4_hat_est - y4))
        x1_hat_est = x1_hat_est + Ts * x1_hat_est_dot
        x2_hat_est = x2_hat_est + Ts * x2_hat_est_dot
        x3_hat_est = x3_hat_est + Ts * x3_hat_est_dot
        x4_hat_est = x4_hat_est + Ts * x4_hat_est_dot
        estimator1 = np.hstack((estimator1,x1_hat_est[0:2,0].reshape((2,1))))
        estimator2 = np.hstack((estimator2, x2_hat_est[0:2, 0].reshape((2, 1))))
        estimator3 = np.hstack((estimator3, x3_hat_est[0:2, 0].reshape((2, 1))))
        estimator4 = np.hstack((estimator4, x4_hat_est[0:2, 0].reshape((2, 1))))

        # Trajectory codac
        tx1.set([x1[0, -1], x1[1, -1]], t)
        tx2.set([x2[0, -1], x2[1, -1]], t)
        tx3.set([x3[0, -1], x3[1, -1]], t)
        tx4.set([x4[0, -1], x4[1, -1]], t)

        # Adding noise to measurements
        rx1 = random.uniform(-noise, noise)
        rx2 = random.uniform(-noise, noise)
        rx3 = random.uniform(-noise, noise)
        rx4 = random.uniform(-noise, noise)
        ry1 = random.uniform(-noise, noise)
        ry2 = random.uniform(-noise, noise)
        ry3 = random.uniform(-noise, noise)
        ry4 = random.uniform(-noise, noise)

        # Interval estimation made by x1
        x1hat = IntervalVector([[x1[0, -1], x1[0, -1]], [x1[1, -1], x1[1, -1]]]).inflate(0.01) + IntervalVector([[rx1,rx1],[ry1,ry1]])
        x2hat = IntervalVector([[x2[0, -1], x2[0, -1]], [x2[1, -1], x2[1, -1]]]).inflate(0.01) + IntervalVector([[rx2,rx2],[ry2,ry2]])
        x3hat = IntervalVector([[x3[0, -1], x3[0, -1]], [x3[1, -1], x3[1, -1]]]).inflate(0.01) + IntervalVector([[rx3,rx3],[ry3,ry3]])
        x4hat = IntervalVector([[x4[0, -1], x4[0, -1]], [x4[1, -1], x4[1, -1]]]).inflate(0.01) + IntervalVector([[rx4,rx4],[ry4,ry4]])

        ## Contractors
        # Measurements
        ctc.eval.contract(tk, x1hat, x1_tube, speed)
        ctc.eval.contract(tk, x2hat, y2_tube, speed)
        ctc.eval.contract(tk, x3hat, y3_tube, speed)
        ctc.eval.contract(tk, x4hat, y4_tube, speed)
        # Delay
        ctc.delay.contract(tau,x2_tube,y2_tube)
        ctc.delay.contract(tau, x3_tube, y3_tube)
        ctc.delay.contract(tau, x4_tube, y4_tube)

        # Mean values of intervals
        interval_estimator2 = np.hstack((interval_estimator2,np.array([[x2_tube(t)[0].mid()],[x2_tube(t)[1].mid()]])))
        interval_estimator3 = np.hstack((interval_estimator3,np.array([[x3_tube(t)[0].mid()],[x3_tube(t)[1].mid()]])))
        interval_estimator4 = np.hstack((interval_estimator4,np.array([[x4_tube(t)[0].mid()],[x4_tube(t)[1].mid()]])))

    # Deleting the last value because not contracted (delay)
    interval_estimator2 = interval_estimator2[:, 0:-10]
    interval_estimator3 = interval_estimator3[:, 0:-10]
    interval_estimator4 = interval_estimator4[:, 0:-10]

    x1f = IntervalVector([[x1[0, -1],x1[0, -1]], [x1[1, -1],x1[1, -1]]]).inflate(0.1)
    x2f = IntervalVector([[x2[0, -1],x2[0, -1]], [x2[1, -1],x2[1, -1]]]).inflate(0.1)
    x3f = IntervalVector([[x3[0, -1],x3[0, -1]], [x3[1, -1],x3[1, -1]]]).inflate(0.1)
    x4f = IntervalVector([[x4[0, -1],x4[0, -1]], [x4[1, -1],x4[1, -1]]]).inflate(0.1)


    ax1.plot(x1[0,:], x1[1,:], 'c', linewidth=2)
    ax1.plot(x2[0,:], x2[1,:], 'g', linewidth=2)
    ax1.plot(x3[0,:], x3[1,:], 'b', linewidth=2)
    ax1.plot(x4[0,:], x4[1,:], 'r', linewidth=2)

    # Estimations
    ax1.plot(estimator1[0, :], estimator1[1, :], 'olive', linewidth=2)
    line,= ax1.plot(estimator2[0, :], estimator2[1, :], 'olive', linewidth=2,label="Continuous-discrete time estimation")
    ax1.plot(estimator3[0, :], estimator3[1, :], 'olive', linewidth=2)
    ax1.plot(estimator4[0, :], estimator4[1, :], 'olive', linewidth=2)

    # Interval Estimation
    line_int, = ax1.plot(interval_estimator2[0, :], interval_estimator2[1, :], 'black',linestyle='dotted', linewidth=1.5,label="Interval estimation")
    ax1.plot(interval_estimator3[0, :], interval_estimator3[1, :], 'black',linestyle='dotted', linewidth=1.5)
    ax1.plot(interval_estimator4[0, :], interval_estimator4[1, :], 'black',linestyle='dotted', linewidth=1.5)

    ax1.grid(True)
    ax1.plot([x1[0,0], x4[0,0], x2[0,0], x3[0,0], x1[0,0]],
             [x1[1,0], x4[1,0], x2[1,0], x3[1,0], x1[1,0]], 'o', linewidth=2)
    ax1.plot([x1[0,-1], x4[0,-1], x2[0,-1], x3[0,-1], x1[0,-1]],
             [x1[1,-1], x4[1,-1], x2[1,-1], x3[1,-1], x1[1,-1]], 'o', linewidth=2)

    ax1.set_title('Without reference')
    ax1.axis([-6.5, 8.5, -4.5, 4.8])
    ax1.legend(handles=[line,line_int])
    #plt.show()

    ## Vibes
    beginDrawing()
    fig_map_without = VIBesFigMap("Without reference")
    fig_map_without.set_properties(100, 100, 600, 200)
    fig_map_without.axis_limits(-6,8,-4,4, True)
    # Tubes
    fig_map_without.add_tube(x1_tube,"x1", 0, 1)
    fig_map_without.add_tube(x2_tube,"x2",0,1)
    fig_map_without.add_tube(x3_tube,"x3",0,1)
    fig_map_without.add_tube(x4_tube,"x4",0,1)
    # Trajectories
    fig_map_without.add_trajectory(tx1,"*x1",0,1,color = "#00FFFF",vehicle_display = False)
    fig_map_without.add_trajectory(tx2,"*x2",0,1,color = "green",vehicle_display = False)
    fig_map_without.add_trajectory(tx3,"*x3",0,1,color = "blue",vehicle_display = False)
    fig_map_without.add_trajectory(tx4,"*x4",0,1,color = "red",vehicle_display = False)
    # Start & end
    fig_map_without.draw_box(x10,"blue[blue]")
    fig_map_without.draw_box(x20,"blue[blue]")
    fig_map_without.draw_box(x30,"blue[blue]")
    fig_map_without.draw_box(x40,"blue[blue]")
    fig_map_without.draw_box(x1f,"orange[orange]")
    fig_map_without.draw_box(x2f,"orange[orange]")
    fig_map_without.draw_box(x3f,"orange[orange]")
    fig_map_without.draw_box(x4f,"orange[orange]")
    fig_map_without.show(0.1)

def with_reference():
    """
    Part II - DISPLACEMENT based formation with a virtual leader which follows a trajectory
    """
    # initialization
    x1 = np.array([-6, -4]).reshape((2, 1))  # cyan
    x2 = np.array([-5, 4]).reshape((2, 1))  # green
    x3 = np.array([0, 4]).reshape((2, 1))  # blue
    x4 = np.array([8, 3]).reshape((2, 1))  # red
    x5 = np.array([0, 0]).reshape((2, 1))

    # Estimated vector position and speed
    p1_hat_est = x1.copy() + eps * np.random.normal(0, nu, size=(2, 1))
    p2_hat_est = x2.copy() + eps * np.random.normal(0, nu, size=(2, 1))
    p3_hat_est = x3.copy() + eps * np.random.normal(0, nu, size=(2, 1))
    p4_hat_est = x4.copy() + eps * np.random.normal(0, nu, size=(2, 1))
    v1_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    v2_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    v3_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    v4_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    estimator1 = p1_hat_est.copy()
    estimator2 = p2_hat_est.copy()
    estimator3 = p3_hat_est.copy()
    estimator4 = p4_hat_est.copy()

    x1_hat_est = np.vstack((p1_hat_est, v1_hat_est))
    x2_hat_est = np.vstack((p2_hat_est, v2_hat_est))
    x3_hat_est = np.vstack((p3_hat_est, v3_hat_est))
    x4_hat_est = np.vstack((p4_hat_est, v4_hat_est))

    # Measurement
    y1 = np.zeros((2, 1))  # measurement of pj only
    y2 = np.zeros((2, 1))  # measurement of pj only
    y3 = np.zeros((2, 1))  # measurement of pj only
    y4 = np.zeros((2, 1))  # measurement of pj only

    tf = 100.
    tdomain = Interval(0, tf)

    # Intervals
    x10 = IntervalVector([[-6, -6], [-4, -4]]).inflate(0.1)
    x20 = IntervalVector([[-5, -5], [4, 4]]).inflate(0.1)
    x30 = IntervalVector([[0, 0], [4, 4]]).inflate(0.1)
    x40 = IntervalVector([[8, 8], [3, 3]]).inflate(0.1)

    interval_estimator1 = np.array([[x10[0].mid()], [x10[1].mid()]])
    interval_estimator2 = np.array([[x20[0].mid()], [x20[1].mid()]])
    interval_estimator3 = np.array([[x30[0].mid()], [x30[1].mid()]])
    interval_estimator4 = np.array([[x40[0].mid()], [x40[1].mid()]])

    # Measurements
    y2_tube = TubeVector(tdomain, Ts, IntervalVector(2))
    y3_tube = TubeVector(tdomain, Ts, IntervalVector(2))
    y4_tube = TubeVector(tdomain, Ts, IntervalVector(2))

    # Estimations
    x1_tube = TubeVector(tdomain, Ts, IntervalVector(2))
    x2_tube = TubeVector(tdomain, Ts, IntervalVector(2))
    x3_tube = TubeVector(tdomain, Ts, IntervalVector(2))
    x4_tube = TubeVector(tdomain, Ts, IntervalVector(2))

    # Bounded speed
    v = 5
    speed = TubeVector(tdomain, Ts, IntervalVector([[-v, v], [-v, v]]))

    # Trajectories
    tx1 = TrajectoryVector(2)
    tx1.set([x1[0, 0], x1[1, 0]], 0.)
    tx2 = TrajectoryVector(2)
    tx2.set([x2[0, 0], x2[1, 0]], 0.)
    tx3 = TrajectoryVector(2)
    tx3.set([x3[0, 0], x3[1, 0]], 0.)
    tx4 = TrajectoryVector(2)
    tx4.set([x4[0, 0], x4[1, 0]], 0.)

    # Reference formation : rij_ = xi_-xj_
    r11_star = np.array([0, 0]).reshape((2, 1))
    r12_star = -np.array([1, -1]).reshape((2, 1))
    r13_star = -np.array([0, -1]).reshape((2, 1))
    r14_star = -np.array([1, 0]).reshape((2, 1))
    r15_star = -np.array([0.5, -0.5]).reshape((2, 1))
    r1_star_x = np.array([r11_star[0], r12_star[0], r13_star[0], r14_star[0], r15_star[0]])
    r1_star_y = np.array([r11_star[1], r12_star[1], r13_star[1], r14_star[1], r15_star[1]])

    r21_star = -r12_star
    r22_star = np.array([0, 0]).reshape((2, 1))
    r23_star = -np.array([-1, 0]).reshape((2, 1))
    r24_star = -np.array([0, 1]).reshape((2, 1))
    r25_star = -np.array([-0.5, 0.5]).reshape((2, 1))
    r2_star_x = np.array([r21_star[0], r22_star[0], r23_star[0], r24_star[0], r25_star[0]])
    r2_star_y = np.array([r21_star[1], r22_star[1], r23_star[1], r24_star[1], r25_star[1]])

    r31_star = -r13_star
    r32_star = -r23_star
    r33_star = np.array([0, 0]).reshape((2, 1))
    r34_star = -np.array([1, 1]).reshape((2, 1))
    r35_star = -np.array([0.5, 0.5]).reshape((2, 1))
    r3_star_x = np.array([r31_star[0], r32_star[0], r33_star[0], r34_star[0], r35_star[0]])
    r3_star_y = np.array([r31_star[1], r32_star[1], r33_star[1], r34_star[1], r35_star[1]])

    r41_star = -r14_star
    r42_star = -r24_star
    r43_star = -r34_star
    r44_star = np.array([0, 0]).reshape((2, 1))
    r45_star = -np.array([-0.5, -0.5]).reshape((2, 1))
    r4_star_x = np.array([r41_star[0], r42_star[0], r43_star[0], r44_star[0], r45_star[0]])
    r4_star_y = np.array([r41_star[1], r42_star[1], r43_star[1], r44_star[1], r45_star[1]])

    # Adjacency matrix
    A1 = np.ones((5, 5)) - np.eye(5)

    # Laplacian matrix and its second eigenvalue Lambda_2
    L1 = -A1 + 5 * np.eye(5)

    # Tuning gains
    kp = 0.1
    kleader = 0.01

    # Last data received
    t_last = 0.

    # Leader constant velocity
    u_leader = kleader * np.array([[1], [-0.6]])
    for t in tqdm(np.arange(Ts, tf + Ts, Ts), ncols=100):
        tk = Interval(t - Ts / 4, t + Ts / 4)

        X = np.array([x1[0, -1], x2[0, -1], x3[0, -1], x4[0, -1], x5[0, -1]]).reshape((5, 1))
        Y = np.array([x1[1, -1], x2[1, -1], x3[1, -1], x4[1, -1], x5[1, -1]]).reshape((5, 1))

        ux_1 = -kp * A1[0, :] @ (x1[0, -1] - X - r1_star_x)
        uy_1 = -kp * A1[0, :] @ (x1[1, -1] - Y - r1_star_y)

        ux_2 = -kp * A1[1, :] @ (x2[0, -1] - X - r2_star_x)
        uy_2 = -kp * A1[1, :] @ (x2[1, -1] - Y - r2_star_y)

        ux_3 = -kp * A1[2, :] @ (x3[0, -1] - X - r3_star_x)
        uy_3 = -kp * A1[2, :] @ (x3[1, -1] - Y - r3_star_y)

        ux_4 = -kp * A1[3, :] @ (x4[0, -1] - X - r4_star_x)
        uy_4 = -kp * A1[3, :] @ (x4[1, -1] - Y - r4_star_y)

        # Trajectory
        x1 = np.hstack((x1, x1[:, -1].reshape((2, 1)) + Ts * np.vstack((ux_1, uy_1))))
        x2 = np.hstack((x2, x2[:, -1].reshape((2, 1)) + Ts * np.vstack((ux_2, uy_2))))
        x3 = np.hstack((x3, x3[:, -1].reshape((2, 1)) + Ts * np.vstack((ux_3, uy_3))))
        x4 = np.hstack((x4, x4[:, -1].reshape((2, 1)) + Ts * np.vstack((ux_4, uy_4))))
        x5 = np.hstack((x5, x5[:, -1].reshape((2, 1)) + Ts * u_leader))

        ## High-gain continuous-discrete time observer
        # Measurement received
        if data_is_received():
            # last data received
            t_last = t

            # pij(kij(t)) / Noised measurement
            y1 = x1[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            y2 = x2[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            y3 = x3[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            y4 = x4[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)

            # pij_hat(kij(t))
            p1_hat_est = x1_hat_est[0:2, 0].reshape((2, 1))
            p2_hat_est = x2_hat_est[0:2, 0].reshape((2, 1))
            p3_hat_est = x3_hat_est[0:2, 0].reshape((2, 1))
            p4_hat_est = x4_hat_est[0:2, 0].reshape((2, 1))

        # Update of the estimation
        x1_hat_est_dot = A @ x1_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p1_hat_est - y1))
        x2_hat_est_dot = A @ x2_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p2_hat_est - y2))
        x3_hat_est_dot = A @ x3_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p3_hat_est - y3))
        x4_hat_est_dot = A @ x4_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p4_hat_est - y4))
        x1_hat_est = x1_hat_est + Ts * x1_hat_est_dot
        x2_hat_est = x2_hat_est + Ts * x2_hat_est_dot
        x3_hat_est = x3_hat_est + Ts * x3_hat_est_dot
        x4_hat_est = x4_hat_est + Ts * x4_hat_est_dot
        estimator1 = np.hstack((estimator1, x1_hat_est[0:2, 0].reshape((2, 1))))
        estimator2 = np.hstack((estimator2, x2_hat_est[0:2, 0].reshape((2, 1))))
        estimator3 = np.hstack((estimator3, x3_hat_est[0:2, 0].reshape((2, 1))))
        estimator4 = np.hstack((estimator4, x4_hat_est[0:2, 0].reshape((2, 1))))

        # Trajectory codac
        tx1.set([x1[0, -1], x1[1, -1]], t)
        tx2.set([x2[0, -1], x2[1, -1]], t)
        tx3.set([x3[0, -1], x3[1, -1]], t)
        tx4.set([x4[0, -1], x4[1, -1]], t)

        # Adding noise to measurements
        rx1 = random.uniform(-noise, noise)
        rx2 = random.uniform(-noise, noise)
        rx3 = random.uniform(-noise, noise)
        rx4 = random.uniform(-noise, noise)
        ry1 = random.uniform(-noise, noise)
        ry2 = random.uniform(-noise, noise)
        ry3 = random.uniform(-noise, noise)
        ry4 = random.uniform(-noise, noise)

        # Interval estimation made by x1
        x1hat = IntervalVector([[x1[0, -1], x1[0, -1]], [x1[1, -1], x1[1, -1]]]).inflate(0.01) + IntervalVector([[rx1, rx1], [ry1, ry1]])
        x2hat = IntervalVector([[x2[0, -1], x2[0, -1]], [x2[1, -1], x2[1, -1]]]).inflate(0.01) + IntervalVector([[rx2, rx2], [ry2, ry2]])
        x3hat = IntervalVector([[x3[0, -1], x3[0, -1]], [x3[1, -1], x3[1, -1]]]).inflate(0.01) + IntervalVector([[rx3, rx3], [ry3, ry3]])
        x4hat = IntervalVector([[x4[0, -1], x4[0, -1]], [x4[1, -1], x4[1, -1]]]).inflate(0.01) + IntervalVector([[rx4, rx4], [ry4, ry4]])

        ## Contractors
        # Measurements
        ctc.eval.contract(tk, x1hat, x1_tube, speed)
        ctc.eval.contract(tk, x2hat, y2_tube, speed)
        ctc.eval.contract(tk, x3hat, y3_tube, speed)
        ctc.eval.contract(tk, x4hat, y4_tube, speed)
        # Delay
        ctc.delay.contract(tau, x2_tube, y2_tube)
        ctc.delay.contract(tau, x3_tube, y3_tube)
        ctc.delay.contract(tau, x4_tube, y4_tube)

        # Mean values of intervals
        interval_estimator2 = np.hstack((interval_estimator2, np.array([[x2_tube(t)[0].mid()], [x2_tube(t)[1].mid()]])))
        interval_estimator3 = np.hstack((interval_estimator3, np.array([[x3_tube(t)[0].mid()], [x3_tube(t)[1].mid()]])))
        interval_estimator4 = np.hstack((interval_estimator4, np.array([[x4_tube(t)[0].mid()], [x4_tube(t)[1].mid()]])))

    # Deleting the last value because not contracted (delay)
    interval_estimator2 = interval_estimator2[:, 0:-10]
    interval_estimator3 = interval_estimator3[:, 0:-10]
    interval_estimator4 = interval_estimator4[:, 0:-10]

    x1f = IntervalVector([[x1[0, -1], x1[0, -1]], [x1[1, -1], x1[1, -1]]]).inflate(0.1)
    x2f = IntervalVector([[x2[0, -1], x2[0, -1]], [x2[1, -1], x2[1, -1]]]).inflate(0.1)
    x3f = IntervalVector([[x3[0, -1], x3[0, -1]], [x3[1, -1], x3[1, -1]]]).inflate(0.1)
    x4f = IntervalVector([[x4[0, -1], x4[0, -1]], [x4[1, -1], x4[1, -1]]]).inflate(0.1)

    ## Vibes
    beginDrawing()
    fig_map_with = VIBesFigMap("With reference")
    fig_map_with.set_properties(100, 330, 600, 200)
    fig_map_with.axis_limits(-6, 8, -4, 4, True)
    # Tubes
    fig_map_with.add_tube(x1_tube, "x1", 0, 1)
    fig_map_with.add_tube(x2_tube, "x2", 0, 1)
    fig_map_with.add_tube(x3_tube, "x3", 0, 1)
    fig_map_with.add_tube(x4_tube, "x4", 0, 1)
    # Trajectories
    fig_map_with.add_trajectory(tx1, "*x1", 0, 1, color="#00FFFF", vehicle_display=False)
    fig_map_with.add_trajectory(tx2, "*x2", 0, 1, color="green", vehicle_display=False)
    fig_map_with.add_trajectory(tx3, "*x3", 0, 1, color="blue", vehicle_display=False)
    fig_map_with.add_trajectory(tx4, "*x4", 0, 1, color="red", vehicle_display=False)
    # Start & end
    fig_map_with.draw_box(x10, "blue[blue]")
    fig_map_with.draw_box(x20, "blue[blue]")
    fig_map_with.draw_box(x30, "blue[blue]")
    fig_map_with.draw_box(x40, "blue[blue]")
    fig_map_with.draw_box(x1f, "orange[orange]")
    fig_map_with.draw_box(x2f, "orange[orange]")
    fig_map_with.draw_box(x3f, "orange[orange]")
    fig_map_with.draw_box(x4f, "orange[orange]")

    fig_map_with.show(0.1)

    ax2 = fig.add_subplot(212)
    ax2.plot(x1[0, :], x1[1, :], 'c', linewidth=2)
    ax2.plot(x2[0, :], x2[1, :], 'g', linewidth=2)
    ax2.plot(x3[0, :], x3[1, :], 'b', linewidth=2)
    ax2.plot(x4[0, :], x4[1, :], 'r', linewidth=2)

    # Estimations
    ax2.plot(estimator1[0, :], estimator1[1, :], 'olive', linewidth=2)
    line,= ax2.plot(estimator2[0, :], estimator2[1, :], 'olive', linewidth=2,label="Continuous-discrete time estimation")
    ax2.plot(estimator3[0, :], estimator3[1, :], 'olive', linewidth=2)
    ax2.plot(estimator4[0, :], estimator4[1, :], 'olive', linewidth=2)

    # Interval Estimation
    line_int, = ax2.plot(interval_estimator2[0, :], interval_estimator2[1, :], 'black', linestyle='dotted', linewidth=1.5, label="Interval estimation")
    ax2.plot(interval_estimator3[0, :], interval_estimator3[1, :], 'black', linestyle='dotted', linewidth=1.5)
    ax2.plot(interval_estimator4[0, :], interval_estimator4[1, :], 'black', linestyle='dotted', linewidth=1.5)

    ax2.grid(True)
    ax2.plot([x1[0, 0], x4[0, 0], x2[0, 0], x3[0, 0], x1[0, 0]], [x1[1, 0], x4[1, 0], x2[1, 0], x3[1, 0], x1[1, 0]],'o', linewidth=2)
    ax2.plot([x1[0, -1], x4[0, -1], x2[0, -1], x3[0, -1], x1[0, -1]],[x1[1, -1], x4[1, -1], x2[1, -1], x3[1, -1], x1[1, -1]], 'o', linewidth=2)
    ax2.set_title('With reference')
    ax2.axis([-6.5, 8.5, -10, 4.8])
    ax2.legend(handles=[line,line_int])
    plt.show()

without_reference()
with_reference()
