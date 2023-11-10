import numpy as np
import matplotlib.pyplot as plt
from codac import *
from tqdm import tqdm
from scipy.signal import place_poles
from numpy.linalg import inv
from numpy import pi
import time
import random
from matplotlib.patches import Circle

## Mission parameters
N = 3    # Number of robots
tf = 30.
Ts = 0.02 # Sampling period
# Tuning gains for pattern following
kp = 1.
# Trajectory to follow
w = 0.1
lx = 7
ly = 10
# Collision avoidance parameters
k1 = 1
k2 = 2
k3 = 1
k4 = 2
d0_inf = 1.
vT = 0.25
acc_max = 15.
speed_max = 2.
k_acc_avoid = 1.

# Spacing of the patern
sp = 1
def pattern(sp):
    # Reference formation : rij_ = xi_-xj_
    r11_star = np.array([0, 0]).reshape((2, 1))
    r12_star = np.array([sp * cos(pi / 6), sp * (1 + sin(pi / 6))]).reshape((2, 1))
    r13_star = np.array([-sp * cos(pi / 6), sp * (1 + sin(pi / 6))]).reshape((2, 1))
    r14_star = np.array([0, sp]).reshape((2, 1))
    r1_star_x = np.array([r11_star[0], r12_star[0], r13_star[0], r14_star[0]])
    r1_star_y = np.array([r11_star[1], r12_star[1], r13_star[1], r14_star[1]])

    r21_star = -r12_star
    r22_star = np.array([0, 0]).reshape((2, 1))
    r23_star = np.array([-2 * sp * cos(pi / 6), 0]).reshape((2, 1))
    r24_star = np.array([-sp * cos(pi / 6), -sp * sin(pi / 6)]).reshape((2, 1))
    r2_star_x = np.array([r21_star[0], r22_star[0], r23_star[0], r24_star[0]])
    r2_star_y = np.array([r21_star[1], r22_star[1], r23_star[1], r24_star[1]])

    r31_star = -r13_star
    r32_star = -r23_star
    r33_star = np.array([0, 0]).reshape((2, 1))
    r34_star = np.array([sp * cos(pi / 6), -sp * sin(pi / 6)]).reshape((2, 1))

    r3_star_x = np.array([r31_star[0], r32_star[0], r33_star[0], r34_star[0]])
    r3_star_y = np.array([r31_star[1], r32_star[1], r33_star[1], r34_star[1]])

    return r1_star_x, r1_star_y, r2_star_x, r2_star_y, r3_star_x, r3_star_y


def x_hat(t): return lx*cos(w*t)
def dx_hat(t): return -lx*w*sin(w*t)
def ddx_hat(t): return -lx*(w**2)*cos(w*t)
def y_hat(t): return lx*sin(w*t)#ly*sin(2*w*t)
def dy_hat(t): return lx*w*cos(w*t)#ly*w*cos(2*w*t)
def ddy_hat(t): return -lx*(w**2)*sin(w*t)


speed_max_ = speed_max * 0.5
def bound(u,v): # each drone bounds its speed
    if v < -speed_max_ and u < 0 :
        return 0.
    if v > speed_max_ and u > 0 :
        return 0.
    else :
        return u
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
t_without_data = Ts # Time since the last measurement received
data_is_not_received = False
chance = 4.  # %
def data_is_received():
    global t_without_data, data_is_not_received
    boolean = random.random() < chance/100
    boolean = True
    if boolean:
        t_without_data = Ts
        data_is_not_received = False
    else :
        t_without_data+=Ts
        data_is_not_received = True
    return boolean



"""
Collision avoidance - Saturation functions
"""

def sat(xmin,x):
    return np.max([xmin,x])

d0_inf_ = 1.3*d0_inf
def H1(X,X0):       # X = [x, y, vx, vy, ax, ay]
    global data_is_not_received
    X = X.flatten()
    X0 = X0.flatten()
    D1 = 2*(X[2]-X0[2])**2 + 2*(X[3]-X0[3])**2 - 2*(X[0]-X0[0])*X0[4] + 2*(X[1]-X0[1])*X0[5]
    d02 = (X[0]-X0[0])**2 + (X[1]-X0[1])**2
    d02_dot = (X[2]-X0[2])*2 + (X[3]-X0[3])*2
    if data_is_not_received:
        dd0_inf = speed_max
    else :
        dd0_inf = 0
    plus = 2*(dd0_inf**2 + 0) + k1*k2*2*dd0_inf*d0_inf_
    h = -D1  - (k1 + k2) * d02_dot - k1*k2*(d02 - (d0_inf_**2)) + plus
    return h

def H2(X,X0):   # X = [x, y, vx, vy, ax, ay]
    global data_is_not_received
    X = X.flatten()
    X0 = X0.flatten()
    D2 = -(X[3]-X0[3])*X[2] + (X[2]-X0[2])*X[3]
    d02 = (X[0] - X0[0]) ** 2 + (X[1] - X0[1]) ** 2
    d02_dot = (X[2]-X0[2])*2 + (X[3]-X0[3])*2
    phi_orth = -(X[1]-X0[1])*X[2] + (X[0]-X0[0])*X[3]
    if data_is_not_received:
        dd0_inf = speed_max
    else :
        dd0_inf = 0
    plus = 2*dd0_inf*d0_inf_*vT + k3*2*dd0_inf*d0_inf_
    h = -D2  -k3 * d02_dot -k4*(phi_orth - d0_inf_*vT) - k4*k3 * (d02 - (d0_inf_**2)) + plus
    return h

def M(X,X0):   # X = [x, y, vx, vy, ax, ay]
    X = X.flatten()
    X0 = X0.flatten()
    M0 = np.array([[2*(X[0]-X0[0]),2*(X[1]-X0[1])],
                  [-(X[1]-X0[1]),X[0]-X0[0]]])
    return M0

room_pat = 1.5               # room between pattern spacing and interval max_lenght
room_drone_lenght = 0.5      # room between d0_inf and interval max_lenght
def pattern_spacing(l2,l3):
    max = 10
    min = 0.1
    return np.min([np.max([l2 + room_pat, min+room_pat]),max])

def norm(vector):
    return sqrt(vector[0,0]**2 + vector[1,0]**2)

def acc_avoid(x,x0):
    ur = (x[0:2,0].reshape((2,1)) - x0[0:2,0].reshape((2,1)))/norm(x[0:2,0].reshape((2,1)) - x0[0:2,0].reshape((2,1)))
    return k_acc_avoid * ur



# Dislay- 2D Trajectories
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax_l = 12
xmin,xmax = -ax_l,ax_l
ymin,ymax = -ax_l,ax_l

d01,d02,d03,d04 = [],[],[],[]


def clear(ax):
    ax.clear()
    ax.xmin = xmin
    ax.xmax = xmax
    ax.ymin = ymin
    ax.ymax = ymax

def mission():
    global ax,vT,d0_inf,x1,x2,x3
    """
    DISPLACEMENT based formation with a virtual leader which follows a trajectory
    The pattern inflates when there is delay in communication
    """
    kdisp = 0
    # initialization
    x1 = np.array([1, 3, 0, 0]).reshape((4, 1))  # cyan
    x2 = np.array([-5, 2, 0, 0]).reshape((4, 1))  # green
    x3 = np.array([4, -3, 0, 0]).reshape((4, 1))  # blue
    x4 = np.array([x_hat(0), y_hat(0), dx_hat(0), dy_hat(0)]).reshape((4, 1))

    # Estimated vector position and speed
    p1_hat_est = x1.copy()[0:2].reshape((2,1)) + eps * np.random.normal(0, nu, size=(2, 1))
    p2_hat_est = x2.copy()[0:2].reshape((2,1)) + eps * np.random.normal(0, nu, size=(2, 1))
    p3_hat_est = x3.copy()[0:2].reshape((2,1)) + eps * np.random.normal(0, nu, size=(2, 1))
    v1_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    v2_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    v3_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    a1_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    a2_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    a3_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))

    estimator1 = p1_hat_est.copy()
    estimator2 = p2_hat_est.copy()
    estimator3 = p3_hat_est.copy()

    x1_hat_est = np.vstack((p1_hat_est, v1_hat_est, a1_hat_est))
    x2_hat_est = np.vstack((p2_hat_est, v2_hat_est, a2_hat_est))
    x3_hat_est = np.vstack((p3_hat_est, v3_hat_est, a3_hat_est))

    # Measurement
    y1 = np.zeros((2, 1))  # measurement of pj only
    y2 = np.zeros((2, 1))  # measurement of pj only
    y3 = np.zeros((2, 1))  # measurement of pj only
    yv1 = np.zeros((2, 1))  # measurement of pj only
    yv2 = np.zeros((2, 1))  # measurement of pj only
    yv3 = np.zeros((2, 1))  # measurement of pj only

    # Display
    ax1t, ay1t = [], []
    ax2t, ay2t = [], []
    ax3t, ay3t = [], []
    vx1t, vy1t = [], []
    vx2t, vy2t = [], []
    vx3t, vy3t = [], []

    # Adjacency matrix
    A1 = np.ones((4, 4)) - np.eye(4)

    # Laplacian matrix and its second eigenvalue Lambda_2
    L1 = -A1 + 4 * np.eye(4)

    # Last data received
    t_last = 0.

    # Trajectories
    tx1 = np.array([[],[]])
    tx2 = np.array([[], []])
    tx3 = np.array([[], []])
    d0_min = []

    """
    Interval estimation of the other drones (x1's point of view)
    """
    x1int = IntervalVector([[x1[0, 0], x1[0, 0]], [x1[1, 0], x1[1, 0]]])
    x2int = IntervalVector([[x2[0, 0], x2[0, 0]], [x2[1, 0], x2[1, 0]]])
    x3int = IntervalVector([[x3[0, 0], x3[0, 0]], [x3[1, 0], x3[1, 0]]])
    vx1int = IntervalVector([[x1[2, 0], x1[2, 0]], [x1[3, 0], x1[3, 0]]])
    vx2int = IntervalVector([[x2[2, 0], x2[2, 0]], [x2[3, 0], x2[3, 0]]])
    vx3int = IntervalVector([[x3[2, 0], x3[2, 0]], [x3[3, 0], x3[3, 0]]])
    c1 = np.array(x1int.mid()).reshape((2, 1))
    c2 = np.array(x2int.mid()).reshape((2, 1))
    c3 = np.array(x3int.mid()).reshape((2, 1))
    speed = IntervalVector([[-speed_max,speed_max],[-speed_max,speed_max]])    # bounded speed
    acc = IntervalVector([[-acc_max,acc_max],[-acc_max,acc_max]])

    kdisp = 0
    spacing = 0.                                   # initialisation of the pattern spacing
    for t in np.arange(Ts, tf + Ts, Ts):

        # At every instant, if no data is received the interval estimation inflates
        vx1int = (x1int + Ts * acc)
        vx2int = (x2int + Ts * acc)
        vx3int = (x3int + Ts * acc)
        x1int = x1int + Ts * speed
        x2int = x2int + Ts * speed
        x3int = x3int + Ts * speed

        """
        High-gain continuous-discrete time observer
        """
        # Measurement received
        if data_is_received():
            # last data received
            t_last = t

            # pij(kij(t)) / Noised measurement
            y1 = x1[0:2, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            y2 = x2[0:2, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            y3 = x3[0:2, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            yv1 = x1[2:4, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            yv2 = x2[2:4, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            yv3 = x3[2:4, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)

            # pij_hat(kij(t))
            p1_hat_est = x1_hat_est[0:2, 0].reshape((2, 1))
            p2_hat_est = x2_hat_est[0:2, 0].reshape((2, 1))
            p3_hat_est = x3_hat_est[0:2, 0].reshape((2, 1))

            # Intervals
            x1int = IntervalVector([[y1[0, 0], y1[0, 0]], [y1[1, 0], y1[1, 0]]]).inflate(0.2)
            x2int = IntervalVector([[y2[0, 0], y2[0, 0]], [y2[1, 0], y2[1, 0]]]).inflate(0.2)
            x3int = IntervalVector([[y3[0, 0], y3[0, 0]], [y3[1, 0], y3[1, 0]]]).inflate(0.2)
            vx1int = IntervalVector([[yv1[0, 0], yv1[0, 0]], [yv1[1, 0], yv1[1, 0]]]).inflate(0.02)
            vx2int = IntervalVector([[yv2[0, 0], yv2[0, 0]], [yv2[1, 0], yv2[1, 0]]]).inflate(0.02)
            vx3int = IntervalVector([[yv3[0, 0], yv3[0, 0]], [yv3[1, 0], yv3[1, 0]]]).inflate(0.02)

            c1 = np.array(x1int.mid()).reshape((2, 1))
            c2 = np.array(x2int.mid()).reshape((2, 1))
            c3 = np.array(x3int.mid()).reshape((2, 1))


        # Update of the estimation
        # x1_hat_est_dot = A @ x1_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p1_hat_est - y1))
        # x2_hat_est_dot = A @ x2_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p2_hat_est - y2))
        # x3_hat_est_dot = A @ x3_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p3_hat_est - y3))
        #
        # x1_hat_est = x1_hat_est + Ts * x1_hat_est_dot
        # x2_hat_est = x2_hat_est + Ts * x2_hat_est_dot
        # x3_hat_est = x3_hat_est + Ts * x3_hat_est_dot
        #
        # estimator1 = np.hstack((estimator1, x1_hat_est[0:2, 0].reshape((2, 1))))
        # estimator2 = np.hstack((estimator2, x2_hat_est[0:2, 0].reshape((2, 1))))
        # estimator3 = np.hstack((estimator3, x3_hat_est[0:2, 0].reshape((2, 1))))


        # Time-varying pattern
        l2 = x2int.max_diam()/2
        l3 = x3int.max_diam()/2
        new_spacing = pattern_spacing(l2,l3)
        # The triangle must not shrink to fast
        if new_spacing > spacing:
            spacing = new_spacing
        else :
            spacing += -0.05*2

        spacing = 1.5

        d0_min.append(l2)

        r1_star_x, r1_star_y, r2_star_x, r2_star_y, r3_star_x, r3_star_y = pattern(spacing)

        X = np.array([x1[0, 0], x2[0, 0], x3[0, 0], x4[0, 0]]).reshape((4, 1))
        dX = np.array([x1[2, 0], x2[2, 0], x3[2, 0], x4[2, 0]]).reshape((4, 1))
        Y = np.array([x1[1, 0], x2[1, 0], x3[1, 0], x4[1, 0]]).reshape((4, 1))
        dY = np.array([x1[3, 0], x2[3, 0], x3[3, 0], x4[3, 0]]).reshape((4, 1))

        ux_1 = -kp * A1[0, :] @ (x1[0, 0] - X - r1_star_x + 2*(x1[2, 0] - dX)) + ddx_hat(t)
        uy_1 = -kp * A1[0, :] @ (x1[1, 0] - Y - r1_star_y + 2*(x1[3, 0] - dY)) + ddy_hat(t)
        ux_2 = -kp * A1[1, :] @ (x2[0, 0] - X - r2_star_x + 2*(x2[2, 0] - dX)) + ddx_hat(t)
        uy_2 = -kp * A1[1, :] @ (x2[1, 0] - Y - r2_star_y + 2*(x2[3, 0] - dY)) + ddy_hat(t)
        ux_3 = -kp * A1[2, :] @ (x3[0, 0] - X - r3_star_x + 2*(x3[2, 0] - dX)) + ddx_hat(t)
        uy_3 = -kp * A1[2, :] @ (x3[1, 0] - Y - r3_star_y + 2*(x3[3, 0] - dY)) + ddy_hat(t)


        """ 
        Collision avoidance 
        """
        d01_,d02_,d03_= [],[],[]
        ## Drone 1
        X1 = np.vstack( (x1[:, 0].reshape((4,1)), np.array([ux_1,uy_1]) ) ).reshape((6, 1))  # x1 knows where he is
        X2 = np.vstack((x2[:, 0].reshape((4,1)), np.array([ux_2,uy_2]) )).reshape((6, 1))
        X3 = np.vstack((x3[:, 0].reshape((4,1)), np.array([ux_3,uy_3]) )).reshape((6, 1))
        # /drone 2
        d = sqrt((x1[0,0]-x2[0,0])**2+(x1[1,0]-x2[1,0])**2)
        d01_.append(d)
        d02_.append(d)
        M0 = M(X1,X2)
        satx = (M0@np.array([[ux_1[0]],[uy_1[0]]]))[0,0]
        saty = (M0@np.array([[ux_1[0]],[uy_1[0]]]))[1,0]
        h1 = H1(X1,X2)
        h2 = H2(X1,X2)
        SAT = np.array([[sat(h1,satx)],
                        [sat(h2,saty)]])
        u1_sat = inv(M0)@SAT
        ux_1,uy_1 = u1_sat[0,0],u1_sat[1,0]

        # /drone 3
        d = sqrt((x1[0, -1] - x3[0, -1]) ** 2 + (x1[1, -1] - x3[1, -1]) ** 2)
        d01_.append(d)
        d03_.append(d)
        M0 = M(X1, X3)
        satx = (M0 @ np.array([[ux_1], [uy_1]]))[0, 0]
        saty = (M0 @ np.array([[ux_1], [uy_1]]))[1, 0]
        h1 = H1(X1, X3)
        h2 = H2(X1, x3)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u1_sat = inv(M0) @ SAT
        ux_1, uy_1 = u1_sat[0, 0], u1_sat[1, 0]
        d01.append(np.min(d01_))

        ## Drone 2
        X1 = np.vstack((x1[:, 0].reshape((4, 1)), np.array([[ux_1], [uy_1]]))).reshape((6, 1))  # x1 knows where he is
        X2 = np.vstack((x2[:, 0].reshape((4, 1)), np.array([ux_2, uy_2]))).reshape((6, 1))
        X3 = np.vstack((x3[:, 0].reshape((4, 1)), np.array([ux_3, uy_3]))).reshape((6, 1))
        # /drone 1
        M0 = M(X2,X1)
        satx = (M0 @ np.array([[ux_2[0]], [uy_2[0]]]))[0, 0]
        saty = (M0 @ np.array([[ux_2[0]], [uy_2[0]]]))[1, 0]
        h1 = H1(X2, X1)
        h2 = H2(X2, X1)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u2_sat = inv(M0) @ SAT
        ux_2, uy_2 = u2_sat[0, 0], u2_sat[1, 0]

        # /drone 3
        d = sqrt((x3[0, -1] - x2[0, -1]) ** 2 + (x3[1, -1] - x2[1, -1]) ** 2)
        d02_.append(d)
        d03_.append(d)
        M0 = M(X2, X3)
        satx = (M0 @ np.array([[ux_2], [uy_2]]))[0, 0]
        saty = (M0 @ np.array([[ux_2], [uy_2]]))[1, 0]
        h1 = H1(X2, X3)
        h2 = H2(X2, X3)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u2_sat = inv(M0) @ SAT
        ux_2, uy_2 = u2_sat[0, 0], u2_sat[1, 0]
        d02.append(np.min(d02_))

        ## Drone 3
        X1 = np.vstack((x1[:, 0].reshape((4, 1)), np.array([[ux_1], [uy_1]]))).reshape((6, 1))  # x1 knows where he is
        X2 = np.vstack((x2[:, 0].reshape((4, 1)), np.array([[ux_2], [uy_2]]))).reshape((6, 1))
        X3 = np.vstack((x3[:, 0].reshape((4, 1)), np.array([ux_3, uy_3]))).reshape((6, 1))

        # /drone 1
        M0 = M(X3, X1)
        satx = (M0 @ np.array([[ux_3[0]], [uy_3[0]]]))[0, 0]
        saty = (M0 @ np.array([[ux_3[0]], [uy_3[0]]]))[1, 0]
        h1 = H1(X3, X1)
        h2 = H2(X3, X1)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u3_sat = inv(M0) @ SAT
        ux_3, uy_3 = u3_sat[0, 0], u3_sat[1, 0]

        # /drone 2
        M0 = M(X3, X2)
        satx = (M0 @ np.array([[ux_3], [uy_3]]))[0, 0]
        saty = (M0 @ np.array([[ux_3], [uy_3]]))[1, 0]
        h1 = H1(X3, X2)
        h2 = H2(X3, X2)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u3_sat = inv(M0) @ SAT
        ux_3, uy_3 = u3_sat[0, 0], u3_sat[1, 0]
        d03.append(np.min(d03_))

        # Bound the velocity of the drones
        ux_1 = bound(ux_1, x1[2, 0])
        uy_1 = bound(uy_1, x1[3, 0])
        ux_2 = bound(ux_2, x2[2, 0])
        uy_2 = bound(uy_2, x2[3, 0])
        ux_3 = bound(ux_3, x3[2, 0])
        uy_3 = bound(uy_3, x3[3, 0])

        # Leader
        x4 = np.array([[x_hat(t)],[y_hat(t)],[dx_hat(t)],[dy_hat(t)]])

        x1 = np.vstack((x1[0:2, 0].reshape((2,1)) + Ts * x1[2:4, 0].reshape((2,1)), x1[2:4, 0].reshape((2,1)) + Ts * np.array([ux_1, uy_1]).reshape((2,1))))
        x2 = np.vstack((x2[0:2, 0].reshape((2,1)) + Ts * x2[2:4, 0].reshape((2,1)), x2[2:4, 0].reshape((2,1)) + Ts * np.array([ux_2, uy_2]).reshape((2,1))))
        x3 = np.vstack((x3[0:2, 0].reshape((2,1)) + Ts * x3[2:4, 0].reshape((2,1)), x3[2:4, 0].reshape((2,1)) + Ts * np.array([ux_3, uy_3]).reshape((2,1))))

        # Trajectory
        tx1 = np.hstack((tx1, x1[0:2, 0].reshape((2, 1)) + Ts * x1[2:4,0].reshape((2, 1)) ))    #+ Ts * np.vstack((ux_1, uy_1))))
        tx2 = np.hstack((tx2, x2[0:2, 0].reshape((2, 1)) + Ts * x2[2:4,0].reshape((2, 1)) ))    #+ Ts * np.vstack((ux_2, uy_2))))
        tx3 = np.hstack((tx3, x3[0:2, 0].reshape((2, 1)) + Ts * x3[2:4,0].reshape((2, 1)) ))    #+ Ts * np.vstack((ux_3, uy_3))))
        ax1t.append(ux_1) ; ay1t.append(uy_1)
        ax2t.append(ux_2) ; ay2t.append(uy_2)
        ax3t.append(ux_3) ; ay3t.append(uy_3)
        vx1t.append(x1[2,0]) ; vy1t.append(x1[3,0])
        vx2t.append(x2[2,0]) ; vy2t.append(x2[3,0])
        vx3t.append(x3[2,0]) ; vy3t.append(x3[3,0])

        # Dynamic display
        if kdisp % 3 == 0:
            clear(ax)
            s = spacing  # pattern_spacing(l2, l3)
            # ax.plot(x4[0, -1], x4[1, -1] + s, 'r', marker='o')
            # ax.plot(x4[0, -1] - s * cos(pi / 6), x4[1, -1] - s * sin(pi / 6), 'r', marker='o')
            # ax.plot(x4[0, -1] + s * cos(pi / 6), x4[1, -1] - s * sin(pi / 6), 'r', marker='o')
            ax.grid(True)
            T = np.arange(0, 2 * pi, 0.01)
            plt.plot(lx * np.cos(T), lx * np.sin(T), color='indianred',linestyle='dashed')
            ax.plot([x1[0, 0]], [x1[1, 0]], 'o', linewidth=2, color="hotpink", label="Drone 1")
            ax.plot([x2[0, 0], x3[0, 0]],[x2[1, 0], x3[1, 0]], 'o', linewidth=2,alpha=0.5, color = "orange",label="Other drones")
            ax.set_title("Triangle patern - Drone1's point of view ")
            ax.axis([-ax_l, ax_l, -ax_l, ax_l])
            ax.text(6,10.5, "Time : t = " + str("{:.2f}".format(np.round(t,2))) + " s")

            # Intervals
            circle2 = Circle(c2, l2, edgecolor='blue', facecolor='none',label= "Interval estimation of the position of other robots")
            circle3 = Circle(c3, l3, edgecolor='blue', facecolor='none')
            #circle1 = Circle(c1, d0_inf, edgecolor='blue', facecolor='none')
            ax.add_patch(circle2)
            ax.add_patch(circle3)
            #ax.add_patch(circle1)

            ax.legend(loc='lower right')
            plt.pause(0.001)
        kdisp += 1


    # Display 2 - 1D Trajectories
    fig_mult, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 9))
    Axt = [ax1t, ax2t, ax3t]
    Ayt = [ay1t, ay2t, ay3t]
    Vxt = [vx1t, vx2t, vx3t]
    Vyt = [vy1t, vy2t, vy3t]
    t = np.linspace(0, tf, len(d01))
    D0t = [d01, d02, d03]
    for i in range(5):
        for j in range(3):
            ax = axes.flat[3 * i + j]
            if i == 0:
                ax.set_title('Drone n°' + str(j + 1))
                ax.plot(t, Axt[j], color='royalblue')
                if j == 0: ax.set_ylabel('ax [m.s-2]')
            if i == 1:
                ax.plot(t, Ayt[j], color='royalblue')
                if j == 0: ax.set_ylabel('ay [m.s-2]')
            if i == 2:
                ax.plot(t, Vxt[j], color='royalblue')
                if j == 0: ax.set_ylabel('vx [m.s-1]')
            if i == 3:
                ax.plot(t, Vyt[j], color='royalblue')
                if j == 0: ax.set_ylabel('vy [m.s-1]')
            if i == 4:
                # Constant data reception
                ax.set_ylim(0, 5)
                ax.set_xlabel("Time [s]")
                if j == 0: ax.set_ylabel('Distance to closest drone [m]')
                ax.plot(t, D0t[j], color='darkblue')
                ax.axhspan(ymin=ax.get_ylim()[0], ymax=d0_inf, facecolor='gray', alpha=0.5)

                # Intermittent data reception
                # ax.set_ylim(0, 10)
                # ax.set_xlabel("Time [s]")
                # if j == 0: ax.set_ylabel('Distance to closest drone [m]')
                # ax.plot(np.linspace(0, tf, len(D0t[j])), D0t[j], color='darkblue')
                # ax.fill_between(t, d0_min, color='gray', alpha=0.5, label='Zone grise')

    fig_mult.tight_layout()
    fig_mult.suptitle("Trajectories of the drones", fontweight='bold', y=1.)

    plt.show()




mission()
