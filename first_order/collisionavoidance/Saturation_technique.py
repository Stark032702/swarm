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

# Spacing of the patern
sp = 2

"""
High-gain continuous-discrete time observer
"""
# Random noise for the initial estimation
eps = 0.2
nu = 1

# Observer parameters
θ = 2                    # inverserly proportional to the maximum delay of measurement
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
    boolean = random.choice([True, True,True,True,True, False])
    return True #boolean


"""
Collision avoidance - Saturation functions
"""
# Tuning parameters
k1 = 1
k2 = 0.5


d0_inf = 1.
vT = 0.25

def sat(xmin,x):
    return np.max([xmin,x])

def H1(X,X0):
    X = X.flatten()
    X0 = X0.flatten()
    D1 = -2*X[2]*(X[0]-X0[0])- 2*X[3]*(X[1]-X0[1])
    d02 = (X[0]-X0[0])**2 + (X[1]-X0[1])**2
    d0_dot = np.sqrt((X[2]-X0[2])**2 + (X[3]-X0[3])**2)
    h = -D1  - k1*(d02-(d0_inf**2))
    #print(h)
    return h

def H2(X,X0):
    X = X.flatten()
    X0 = X0.flatten()
    D2 = 0
    d02 = (X[0] - X0[0]) ** 2 + (X[1] - X0[1]) ** 2
    d0_dot = np.sqrt((X[2] - X0[2]) ** 2 + (X[3] - X0[3]) ** 2)
    ΦT = -(X[1]-X0[1])*X[2] + (X[0]-X0[0])*X[3]
    h = -D2  - -d0_inf*vT - k2*(d02-(d0_inf**2))
    #print(h)
    return h

def M(X,X0):
    X = X.flatten()
    X0 = X0.flatten()
    M0 = np.array([[2*(X[0]-X0[0]),2*(X[1]-X0[1])],
                  [-(X[1]-X0[1]),X[0]-X0[0]]])
    return M0


# Dislay 1 - 2D Trajectories
fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# Display 2 - 1D Trajectories
fig_mult, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))
Xt_dev = []
Yt_dev = []
Xt = []
Yt = []
D0t = []
d01,d02,d03,d04 = [],[],[],[]



def without_collision_avoidance():
    """
        Part I - DISPLACEMENT based formation with a virtual leader which follows a trajectory
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
    r12_star = -np.array([sp, -sp]).reshape((2, 1))
    r13_star = -np.array([0, -sp]).reshape((2, 1))
    r14_star = -np.array([sp, 0]).reshape((2, 1))
    r15_star = -np.array([sp / 2, -sp / 2]).reshape((2, 1))
    r1_star_x = np.array([r11_star[0], r12_star[0], r13_star[0], r14_star[0], r15_star[0]])
    r1_star_y = np.array([r11_star[1], r12_star[1], r13_star[1], r14_star[1], r15_star[1]])

    r21_star = -r12_star
    r22_star = np.array([0, 0]).reshape((2, 1))
    r23_star = -np.array([-sp, 0]).reshape((2, 1))
    r24_star = -np.array([0, sp]).reshape((2, 1))
    r25_star = -np.array([-sp / 2, sp / 2]).reshape((2, 1))
    r2_star_x = np.array([r21_star[0], r22_star[0], r23_star[0], r24_star[0], r25_star[0]])
    r2_star_y = np.array([r21_star[1], r22_star[1], r23_star[1], r24_star[1], r25_star[1]])

    r31_star = -r13_star
    r32_star = -r23_star
    r33_star = np.array([0, 0]).reshape((2, 1))
    r34_star = -np.array([sp, sp]).reshape((2, 1))
    r35_star = -np.array([sp / 2, sp / 2]).reshape((2, 1))
    r3_star_x = np.array([r31_star[0], r32_star[0], r33_star[0], r34_star[0], r35_star[0]])
    r3_star_y = np.array([r31_star[1], r32_star[1], r33_star[1], r34_star[1], r35_star[1]])

    r41_star = -r14_star
    r42_star = -r24_star
    r43_star = -r34_star
    r44_star = np.array([0, 0]).reshape((2, 1))
    r45_star = -np.array([-sp / 2, -sp / 2]).reshape((2, 1))
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
        x1hat = IntervalVector([[x1[0, -1], x1[0, -1]], [x1[1, -1], x1[1, -1]]]).inflate(0.01) + IntervalVector(
            [[rx1, rx1], [ry1, ry1]])
        x2hat = IntervalVector([[x2[0, -1], x2[0, -1]], [x2[1, -1], x2[1, -1]]]).inflate(0.01) + IntervalVector(
            [[rx2, rx2], [ry2, ry2]])
        x3hat = IntervalVector([[x3[0, -1], x3[0, -1]], [x3[1, -1], x3[1, -1]]]).inflate(0.01) + IntervalVector(
            [[rx3, rx3], [ry3, ry3]])
        x4hat = IntervalVector([[x4[0, -1], x4[0, -1]], [x4[1, -1], x4[1, -1]]]).inflate(0.01) + IntervalVector(
            [[rx4, rx4], [ry4, ry4]])

        ## Contractors
        # # Measurements
        # ctc.eval.contract(tk, x1hat, x1_tube, speed)
        # ctc.eval.contract(tk, x2hat, y2_tube, speed)
        # ctc.eval.contract(tk, x3hat, y3_tube, speed)
        # ctc.eval.contract(tk, x4hat, y4_tube, speed)
        # # Delay
        # ctc.delay.contract(tau, x2_tube, y2_tube)
        # ctc.delay.contract(tau, x3_tube, y3_tube)
        # ctc.delay.contract(tau, x4_tube, y4_tube)

        # Mean values of intervals
        interval_estimator2 = np.hstack((interval_estimator2, np.array([[x2_tube(t)[0].mid()], [x2_tube(t)[1].mid()]])))
        interval_estimator3 = np.hstack((interval_estimator3, np.array([[x3_tube(t)[0].mid()], [x3_tube(t)[1].mid()]])))
        interval_estimator4 = np.hstack((interval_estimator4, np.array([[x4_tube(t)[0].mid()], [x4_tube(t)[1].mid()]])))

    Xt.append(x1[0, :].copy()); Yt.append(x1[1, :].copy())
    Xt.append(x2[0, :].copy()); Yt.append(x2[1, :].copy())
    Xt.append(x3[0, :].copy()); Yt.append(x3[1, :].copy())
    Xt.append(x4[0, :].copy()); Yt.append(x4[1, :].copy())


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
    fig_map_with = VIBesFigMap("Without Collision avoidance")
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


    ax1.plot(x1[0, :], x1[1, :], 'c', linewidth=2)
    ax1.plot(x2[0, :], x2[1, :], 'g', linewidth=2)
    ax1.plot(x3[0, :], x3[1, :], 'b', linewidth=2)
    ax1.plot(x4[0, :], x4[1, :], 'r', linewidth=2)

    # Estimations
    # ax1.plot(estimator1[0, :], estimator1[1, :], 'olive', linewidth=1.5)
    # line, = ax1.plot(estimator2[0, :], estimator2[1, :], 'olive', linewidth=1.5, label="Continuous-discrete time estimation")
    # ax1.plot(estimator3[0, :], estimator3[1, :], 'olive', linewidth=1.5)
    # ax1.plot(estimator4[0, :], estimator4[1, :], 'olive', linewidth=1.5)

    # Interval Estimation
    # line_int, = ax1.plot(interval_estimator2[0, :], interval_estimator2[1, :], 'black', linestyle='dotted',
    #                      linewidth=1.5, label="Interval estimation")
    # ax1.plot(interval_estimator3[0, :], interval_estimator3[1, :], 'black', linestyle='dotted', linewidth=1.5)
    # ax1.plot(interval_estimator4[0, :], interval_estimator4[1, :], 'black', linestyle='dotted', linewidth=1.5)

    ax1.grid(True)
    ax1.plot([x1[0, 0], x4[0, 0], x2[0, 0], x3[0, 0], x1[0, 0]], [x1[1, 0], x4[1, 0], x2[1, 0], x3[1, 0], x1[1, 0]],
             'o', linewidth=2)
    ax1.plot([x1[0, -1], x4[0, -1], x2[0, -1], x3[0, -1], x1[0, -1]],
             [x1[1, -1], x4[1, -1], x2[1, -1], x3[1, -1], x1[1, -1]], 'o', linewidth=2)
    ax1.set_title('Without collision avoidance')
    ax1.axis([-6.5, 8.5, -10, 4.8])
    #ax1.legend(handles=[line])

def with_collision_avoidance():
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

    ## Intervals
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
    r12_star = -np.array([sp, -sp]).reshape((2, 1))
    r13_star = -np.array([0, -sp]).reshape((2, 1))
    r14_star = -np.array([sp, 0]).reshape((2, 1))
    r15_star = -np.array([sp / 2, -sp / 2]).reshape((2, 1))
    r1_star_x = np.array([r11_star[0], r12_star[0], r13_star[0], r14_star[0], r15_star[0]])
    r1_star_y = np.array([r11_star[1], r12_star[1], r13_star[1], r14_star[1], r15_star[1]])

    r21_star = -r12_star
    r22_star = np.array([0, 0]).reshape((2, 1))
    r23_star = -np.array([-sp, 0]).reshape((2, 1))
    r24_star = -np.array([0, sp]).reshape((2, 1))
    r25_star = -np.array([-sp / 2, sp / 2]).reshape((2, 1))
    r2_star_x = np.array([r21_star[0], r22_star[0], r23_star[0], r24_star[0], r25_star[0]])
    r2_star_y = np.array([r21_star[1], r22_star[1], r23_star[1], r24_star[1], r25_star[1]])

    r31_star = -r13_star
    r32_star = -r23_star
    r33_star = np.array([0, 0]).reshape((2, 1))
    r34_star = -np.array([sp, sp]).reshape((2, 1))
    r35_star = -np.array([sp / 2, sp / 2]).reshape((2, 1))
    r3_star_x = np.array([r31_star[0], r32_star[0], r33_star[0], r34_star[0], r35_star[0]])
    r3_star_y = np.array([r31_star[1], r32_star[1], r33_star[1], r34_star[1], r35_star[1]])

    r41_star = -r14_star
    r42_star = -r24_star
    r43_star = -r34_star
    r44_star = np.array([0, 0]).reshape((2, 1))
    r45_star = -np.array([-sp / 2, -sp / 2]).reshape((2, 1))
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

    # Minimum distance between drones
    d02_inf = 1

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

        ### Collision avoidance
        d01_,d02_,d03_,d04_= [],[],[],[]
        ## Drone 1
        # /drone 2
        d = sqrt((x1[0,-1]-x2[0,-1])**2+(x1[1,-1]-x2[1,-1])**2)
        d01_.append(d)
        d02_.append(d)
        M0 = M(x1_hat_est,x2_hat_est)
        satx = (M0@np.array([[ux_1[0]],[uy_1[0]]]))[0,0]
        saty = (M0@np.array([[ux_1[0]],[uy_1[0]]]))[1,0]
        h1 = H1(x1_hat_est,x2_hat_est)
        h2 = H2(x1_hat_est,x2_hat_est)
        SAT = np.array([[sat(h1,satx)],
                        [sat(h2,saty)]])
        u1_sat = inv(M0)@SAT
        ux_1,uy_1 = u1_sat[0,0],u1_sat[1,0]

        # /drone 3
        d = sqrt((x1[0, -1] - x3[0, -1]) ** 2 + (x1[1, -1] - x3[1, -1]) ** 2)
        d01_.append(d)
        d03_.append(d)
        M0 = M(x1_hat_est, x3_hat_est)
        satx = (M0 @ np.array([[ux_1], [uy_1]]))[0, 0]
        saty = (M0 @ np.array([[ux_1], [uy_1]]))[1, 0]
        h1 = H1(x1_hat_est, x3_hat_est)
        h2 = H2(x1_hat_est, x3_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u1_sat = inv(M0) @ SAT
        ux_1, uy_1 = u1_sat[0, 0], u1_sat[1, 0]

        # /drone 4
        d = sqrt((x1[0, -1] - x4[0, -1]) ** 2 + (x1[1, -1] - x4[1, -1]) ** 2)
        d01_.append(d)
        d04_.append(d)
        M0 = M(x1_hat_est, x4_hat_est)
        satx = (M0 @ np.array([[ux_1], [uy_1]]))[0, 0]
        saty = (M0 @ np.array([[ux_1], [uy_1]]))[1, 0]
        h1 = H1(x1_hat_est, x4_hat_est)
        h2 = H2(x1_hat_est, x4_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u1_sat = inv(M0) @ SAT
        ux_1, uy_1 = u1_sat[0, 0], u1_sat[1, 0]
        d01.append(np.min(d01_))

        ## Drone 2
        # /drone 1
        M0 = M(x2_hat_est,x1_hat_est)
        satx = (M0 @ np.array([[ux_2[0]], [uy_2[0]]]))[0, 0]
        saty = (M0 @ np.array([[ux_2[0]], [uy_2[0]]]))[1, 0]
        h1 = H1(x2_hat_est, x1_hat_est)
        h2 = H2(x2_hat_est, x1_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u2_sat = inv(M0) @ SAT
        ux_2, uy_2 = u2_sat[0, 0], u2_sat[1, 0]

        # /drone 3
        d = sqrt((x3[0, -1] - x2[0, -1]) ** 2 + (x3[1, -1] - x2[1, -1]) ** 2)
        d02_.append(d)
        d03_.append(d)
        M0 = M(x2_hat_est, x3_hat_est)
        satx = (M0 @ np.array([[ux_2], [uy_2]]))[0, 0]
        saty = (M0 @ np.array([[ux_2], [uy_2]]))[1, 0]
        h1 = H1(x2_hat_est, x3_hat_est)
        h2 = H2(x2_hat_est, x3_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u2_sat = inv(M0) @ SAT
        ux_2, uy_2 = u2_sat[0, 0], u2_sat[1, 0]

        # /drone 4
        d = sqrt((x4[0, -1] - x2[0, -1]) ** 2 + (x4[1, -1] - x2[1, -1]) ** 2)
        d02_.append(d)
        d04_.append(d)
        M0 = M(x2_hat_est, x4_hat_est)
        satx = (M0 @ np.array([[ux_2], [uy_2]]))[0, 0]
        saty = (M0 @ np.array([[ux_2], [uy_2]]))[1, 0]
        h1 = H1(x2_hat_est, x4_hat_est)
        h2 = H2(x2_hat_est, x4_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u2_sat = inv(M0) @ SAT
        ux_2, uy_2 = u2_sat[0, 0], u2_sat[1, 0] # saturated output
        d02.append(np.min(d02_))

        ## Drone 3
        # /drone 1
        M0 = M(x3_hat_est, x1_hat_est)
        satx = (M0 @ np.array([[ux_3[0]], [uy_3[0]]]))[0, 0]
        saty = (M0 @ np.array([[ux_3[0]], [uy_3[0]]]))[1, 0]
        h1 = H1(x3_hat_est, x1_hat_est)
        h2 = H2(x3_hat_est, x1_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u3_sat = inv(M0) @ SAT
        ux_3, uy_3 = u3_sat[0, 0], u3_sat[1, 0]

        # /drone 2
        M0 = M(x3_hat_est, x2_hat_est)
        satx = (M0 @ np.array([[ux_3], [uy_3]]))[0, 0]
        saty = (M0 @ np.array([[ux_3], [uy_3]]))[1, 0]
        h1 = H1(x3_hat_est, x2_hat_est)
        h2 = H2(x3_hat_est, x2_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u3_sat = inv(M0) @ SAT
        ux_3, uy_3 = u3_sat[0, 0], u3_sat[1, 0]

        # /drone 4
        d = sqrt((x4[0, -1] - x3[0, -1]) ** 2 + (x4[1, -1] - x3[1, -1]) ** 2)
        d03_.append(d)
        d04_.append(d)
        M0 = M(x3_hat_est, x4_hat_est)
        satx = (M0 @ np.array([[ux_3], [uy_3]]))[0, 0]
        saty = (M0 @ np.array([[ux_3], [uy_3]]))[1, 0]
        h1 = H1(x3_hat_est, x4_hat_est)
        h2 = H2(x3_hat_est, x4_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u3_sat = inv(M0) @ SAT
        ux_3, uy_3 = u3_sat[0, 0], u3_sat[1, 0]
        d03.append(np.min(d03_))

        ## Drone 4
        # /drone 1
        M0 = M(x4_hat_est, x1_hat_est)
        satx = (M0 @ np.array([[ux_4[0]], [uy_4[0]]]))[0, 0]
        saty = (M0 @ np.array([[ux_4[0]], [uy_4[0]]]))[1, 0]
        h1 = H1(x4_hat_est, x1_hat_est)
        h2 = H2(x4_hat_est, x1_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u4_sat = inv(M0) @ SAT
        ux_4, uy_4 = u4_sat[0, 0], u4_sat[1, 0]

        # /drone 2
        M0 = M(x4_hat_est, x2_hat_est)
        satx = (M0 @ np.array([[ux_4], [uy_4]]))[0, 0]
        saty = (M0 @ np.array([[ux_4], [uy_4]]))[1, 0]
        h1 = H1(x4_hat_est, x2_hat_est)
        h2 = H2(x4_hat_est, x2_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u4_sat = inv(M0) @ SAT
        ux_4, uy_4 = u4_sat[0, 0], u4_sat[1, 0]

        # /drone 3
        M0 = M(x4_hat_est, x3_hat_est)
        satx = (M0 @ np.array([[ux_4], [uy_4]]))[0, 0]
        saty = (M0 @ np.array([[ux_4], [uy_4]]))[1, 0]
        h1 = H1(x4_hat_est, x3_hat_est)
        h2 = H2(x4_hat_est, x3_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u4_sat = inv(M0) @ SAT
        ux_4, uy_4 = u4_sat[0, 0], u4_sat[1, 0]
        d04.append(np.min(d04_))




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
        # ctc.eval.contract(tk, x1hat, x1_tube, speed)
        # ctc.eval.contract(tk, x2hat, y2_tube, speed)
        # ctc.eval.contract(tk, x3hat, y3_tube, speed)
        # ctc.eval.contract(tk, x4hat, y4_tube, speed)
        # # Delay
        # ctc.delay.contract(tau, x2_tube, y2_tube)
        # ctc.delay.contract(tau, x3_tube, y3_tube)
        # ctc.delay.contract(tau, x4_tube, y4_tube)

        # Mean values of intervals
        interval_estimator2 = np.hstack((interval_estimator2, np.array([[x2_tube(t)[0].mid()], [x2_tube(t)[1].mid()]])))
        interval_estimator3 = np.hstack((interval_estimator3, np.array([[x3_tube(t)[0].mid()], [x3_tube(t)[1].mid()]])))
        interval_estimator4 = np.hstack((interval_estimator4, np.array([[x4_tube(t)[0].mid()], [x4_tube(t)[1].mid()]])))

    # Display 2
    Xt_dev.append(x1[0, :].copy()); Yt_dev.append(x1[1, :].copy())
    Xt_dev.append(x2[0, :].copy()); Yt_dev.append(x2[1, :].copy())
    Xt_dev.append(x3[0, :].copy()); Yt_dev.append(x3[1, :].copy())
    Xt_dev.append(x4[0, :].copy()); Yt_dev.append(x4[1, :].copy())

    # Deleting the last value because not contracted (delay)
    interval_estimator2 = interval_estimator2[:, 0:-10]
    interval_estimator3 = interval_estimator3[:, 0:-10]
    interval_estimator4 = interval_estimator4[:, 0:-10]

    # Final interval
    x1f = IntervalVector([[x1[0, -1], x1[0, -1]], [x1[1, -1], x1[1, -1]]]).inflate(0.1)
    x2f = IntervalVector([[x2[0, -1], x2[0, -1]], [x2[1, -1], x2[1, -1]]]).inflate(0.1)
    x3f = IntervalVector([[x3[0, -1], x3[0, -1]], [x3[1, -1], x3[1, -1]]]).inflate(0.1)
    x4f = IntervalVector([[x4[0, -1], x4[0, -1]], [x4[1, -1], x4[1, -1]]]).inflate(0.1)

    ## Vibes
    beginDrawing()
    fig_map_with = VIBesFigMap("With collision avoidance")
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


    ax2.plot(x1[0, :], x1[1, :], 'c', linewidth=2)
    ax2.plot(x2[0, :], x2[1, :], 'g', linewidth=2)
    ax2.plot(x3[0, :], x3[1, :], 'b', linewidth=2)
    ax2.plot(x4[0, :], x4[1, :], 'r', linewidth=2)

    # Estimations
    # ax2.plot(estimator1[0, :], estimator1[1, :], 'olive', linewidth=1.5)
    # line,= ax2.plot(estimator2[0, :], estimator2[1, :], 'olive', linewidth=1.5,label="Continuous-discrete time estimation")
    # ax2.plot(estimator3[0, :], estimator3[1, :], 'olive', linewidth=1.5)
    # ax2.plot(estimator4[0, :], estimator4[1, :], 'olive', linewidth=1.5)

    # Interval Estimation
    # line_int, = ax2.plot(interval_estimator2[0, :], interval_estimator2[1, :], 'black', linestyle='dotted', linewidth=1.5, label="Interval estimation")
    # ax2.plot(interval_estimator3[0, :], interval_estimator3[1, :], 'black', linestyle='dotted', linewidth=1.5)
    # ax2.plot(interval_estimator4[0, :], interval_estimator4[1, :], 'black', linestyle='dotted', linewidth=1.5)

    ax2.grid(True)
    ax2.plot([x1[0, 0], x4[0, 0], x2[0, 0], x3[0, 0], x1[0, 0]], [x1[1, 0], x4[1, 0], x2[1, 0], x3[1, 0], x1[1, 0]],'o', linewidth=2)
    ax2.plot([x1[0, -1], x4[0, -1], x2[0, -1], x3[0, -1], x1[0, -1]],[x1[1, -1], x4[1, -1], x2[1, -1], x3[1, -1], x1[1, -1]], 'o', linewidth=2)
    ax2.set_title('With collision avoidance')
    ax2.axis([-6.5, 8.5, -10, 4.8])
    #ax2.legend(handles=[line])  # line_int])


    # Display 2
    t = np.linspace(0,tf,len(Xt[0]))
    D0t = [d01,d02,d03,d04]
    for i in range(3):
        for j in range(4):
            ax = axes.flat[4*i+j]
            if i==0:
                ax.set_title('MAV n°'+str(j+1))
                line1,= ax.plot(t, Xt_dev[j], color='red',linestyle='dashed',label="OIST")
                line2,= ax.plot(t, Xt[j], color='blue',label="without corrector")
                if j==0: ax.set_ylabel('X [m]')
                ax.legend(handles=[line1,line2])
            if i==1:
                line1, = ax.plot(t, Yt_dev[j], color='red',linestyle='dashed',label="OIST")
                line2, = ax.plot(t, Yt[j], color='blue',label="without corrector")
                if j == 0: ax.set_ylabel('Y [m]')
                ax.legend(handles=[line1, line2])
            if i==2:
                ax.set_ylim(0, 5)
                ax.set_xlabel("Time [s]")
                if j == 0: ax.set_ylabel('Distance to closest drone [m]')
                ax.plot(t[0:-1],D0t[j],color='darkblue')
                ax.axhspan(ymin=ax.get_ylim()[0], ymax=d0_inf, facecolor='gray', alpha=0.5)

    fig_mult.tight_layout()
    fig_mult.suptitle("Trajectories of the drones", fontweight='bold', y=1.)


    plt.show()



without_collision_avoidance()
with_collision_avoidance()
