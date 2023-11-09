import numpy as np
from numpy import hstack,vstack,linspace,zeros,array,ones
import matplotlib.pyplot as plt
from codac import *
from tqdm import tqdm
from scipy.signal import place_poles
from numpy.linalg import inv
from numpy import pi
import time
import random
from matplotlib.patches import Circle
from scipy.linalg import expm
from roblib import add1,circle3H,tran3H,ToH,eulerH,draw3H
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


alt = 2
def draw_quadrotor3D(ax, p):
    l = 0.3
    mirror = 1
    R = np.eye(3)
    α = np.zeros((4,1))
    p = np.vstack((p,np.array([[alt]]))).reshape((3,1))
    Ca = hstack((circle3H(0.3 * l), [[0.3 * l, -0.3 * l], [0, 0], [0, 0], [1, 1]]))  # the disc + the blades
    T = tran3H(p[0, 0], p[1, 0], p[2, 0]) @ ToH(R)  # I replaced tran3H(*) to avoid warning
    C0 = T @ tran3H(0, l, 0) @ eulerH(0, 0, α[0]) @ Ca  # we rotate the blades
    C1 = T @ tran3H(-l, 0, 0) @ eulerH(0, 0, -α[1]) @ Ca
    C2 = T @ tran3H(0, -l, 0) @ eulerH(0, 0, α[2]) @ Ca
    C3 = T @ tran3H(l, 0, 0) @ eulerH(0, 0, -α[3]) @ Ca
    M = T @ add1(array([[l, -l, 0, 0, 0], [0, 0, 0, l, -l], [0, 0, 0, 0, 0]]))
    draw3H(ax, M, 'grey', True, mirror)  # body
    draw3H(ax, C0, 'green', True, mirror)
    draw3H(ax, C1, 'black', True, mirror)
    draw3H(ax, C2, 'red', True, mirror)
    draw3H(ax, C3, 'blue', True, mirror)


def rectangle(ax,intvec):
    lz = 0.2
    rx = intvec[0]
    ry = intvec[1]
    vertices = np.array([
        [rx.ub(), ry.ub(), alt+lz/2],
        [rx.lb(), ry.ub(), alt+lz/2],
        [rx.lb(), ry.lb(), alt+lz/2],
        [rx.ub(), ry.lb(), alt+lz/2],
        [rx.ub(), ry.ub(), alt-lz/2],
        [rx.lb(), ry.ub(), alt-lz/2],
        [rx.lb(), ry.lb(), alt-lz/2],
        [rx.ub(), ry.lb(), alt-lz/2]
    ])

    # Cube faces
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Face avant
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Face supérieure
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Face droite
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Face arrière
        [vertices[3], vertices[0], vertices[4], vertices[7]],  # Face inférieure
        [vertices[4], vertices[5], vertices[6], vertices[7]]  # Face gauche
    ]

    # Affichage des faces du cube avec transparence
    for face in faces:
        x = list([v[0] for v in face])
        y = list([v[1] for v in face])
        z = list([v[2] for v in face])
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts,alpha =0.5))



## We define N mobile robots
# for i between 1 and N
# xi = [Pos_xi ; Pos_yi]
# xidot = [ux_i ; uy_i] (so there are two control inputs)


N = 3    # Number of robots
tf = 200.
Ts = 0.1 # Sampling period
to = 0.2
tau = Interval(0,to) # Delay
noise = 0.05 # Noise of the interval measurements

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

# Tuning gains for pattern following
kp = 0.3
kleader = 0.15

## Trajectory to follow
w = 0.1
lx = 5
ly = 2
def x_hat(t): return 0 # lx*cos(w*t)
def dx_hat(t): return 0 # -lx*w*sin(w*t)
def y_hat(t): return 0 # # ly*sin(2*w*t)
def dy_hat(t): return 0 # ly*2*w*cos(w*t)
def control(X,t):
    x = X[0,0]
    y = X[1,0]
    ux = 3 *(x_hat(t) - x) + dx_hat(t)
    uy = 3 *(y_hat(t) - y) + dy_hat(t)
    return np.array([[ux],[uy]])

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
    global t_without_data
    boolean = random.choice([True,False, False])
    if boolean:
        t_without_data = Ts
    else :
        t_without_data+=Ts
    return boolean


# Time since the last measurement received
t_without_data = Ts


"""
Collision avoidance - Saturation functions
"""
# Parameters
k1 = 1
k2 = 0.5
d0_inf = 0.35
vT = 0.25

def sat(xmin,x):
    return np.max([xmin,x])

def H1(X,X0):
    X = X.flatten()
    X0 = X0.flatten()
    D1 = -2*X0[2]*(X[0]-X0[0])- 2*X0[3]*(X[1]-X0[1])
    d02 = (X[0]-X0[0])**2 + (X[1]-X0[1])**2
    h = -D1  - k1*(d02-(d0_inf**2))
    return h

def H2(X,X0):
    X = X.flatten()
    X0 = X0.flatten()
    D2 = 0
    d02 = (X[0] - X0[0]) ** 2 + (X[1] - X0[1]) ** 2
    h = -D2  - -d0_inf*vT - k2*(d02-(d0_inf**2))
    return h

def M(X,X0):
    X = X.flatten()
    X0 = X0.flatten()
    M0 = np.array([[2*(X[0]-X0[0]),2*(X[1]-X0[1])],
                  [-(X[1]-X0[1]),X[0]-X0[0]]])
    return M0

room_pat = 1.5              # room between pattern spacing and interval max_lenght
room_drone_lenght = 0.5      # room between d0_inf and interval max_lenght
def pattern_spacing(l2,l3):
    max = 10
    min = 1
    return np.min([np.max([l2 + room_pat, min+room_pat]),max])

def minimum_distance():
    return 1.


# Dislay- 3D Trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xmin,xmax = -10, 10
ymin, ymax = -10, 10
zmin, zmax = 0, 5
ax.set_xlim3d(xmin, xmax)
ax.set_ylim3d(ymin, ymax)
ax.set_zlim3d(zmin, zmax)

d01,d02,d03,d04 = [],[],[],[]


def clear(ax):
    ax.clear()
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)

def triangle():
    global d0_inf
    """
    DISPLACEMENT based formation with a virtual leader which follows a trajectory
    The pattern inflates when there is delay in communication
    """
    kdisp = 0
    # initialization
    x1 = np.array([-6, -4]).reshape((2, 1))  # cyan
    x2 = np.array([-5, 4]).reshape((2, 1))  # green
    x3 = np.array([0, 4]).reshape((2, 1))  # blue
    x4 = np.array([x_hat(0), y_hat(0)]).reshape((2, 1))


    # Estimated vector position and speed
    p1_hat_est = x1.copy() + eps * np.random.normal(0, nu, size=(2, 1))
    p2_hat_est = x2.copy() + eps * np.random.normal(0, nu, size=(2, 1))
    p3_hat_est = x3.copy() + eps * np.random.normal(0, nu, size=(2, 1))
    v1_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    v2_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))
    v3_hat_est = np.zeros((2, 1)) + eps * np.random.normal(0, nu, size=(2, 1))

    estimator1 = p1_hat_est.copy()
    estimator2 = p2_hat_est.copy()
    estimator3 = p3_hat_est.copy()

    x1_hat_est = np.vstack((p1_hat_est, v1_hat_est))
    x2_hat_est = np.vstack((p2_hat_est, v2_hat_est))
    x3_hat_est = np.vstack((p3_hat_est, v3_hat_est))

    # Measurement
    y1 = np.zeros((2, 1))  # measurement of pj only
    y2 = np.zeros((2, 1))  # measurement of pj only
    y3 = np.zeros((2, 1))  # measurement of pj only


    # Adjacency matrix
    A1 = np.ones((4, 4)) - np.eye(4)

    # Laplacian matrix and its second eigenvalue Lambda_2
    L1 = -A1 + 4 * np.eye(4)

    # Last data received
    t_last = 0.

    """
    Interval estimation of the other drones (x1's point of view)
    """
    x1int = IntervalVector([[x1[0, 0], x1[0, 0]], [x1[1, 0], x1[1, 0]]])
    x1sp = IntervalVector([[0., 0.], [0., 0.]])
    x2int = IntervalVector([[x2[0, 0], x2[0, 0]], [x2[1, 0], x2[1, 0]]])
    x2sp = IntervalVector([[0., 0.], [0., 0.]])
    x3int = IntervalVector([[x3[0, 0], x3[0, 0]], [x3[1, 0], x3[1, 0]]])
    x3sp = IntervalVector([[0., 0.], [0., 0.]])
    ux_2, uy_2 = 0., 0.
    speed = IntervalVector([[-3., 3.], [-3., 3.]])  # bounded speed ~ 15km/h
    ka = 5
    acc = IntervalVector([[-ka, ka], [-ka, ka]])  # bounded acceleration ~

    spacing = 0.                                   # initialisation of the pattern spacing
    ux_1, uy_1 = 0., 0.
    ux_3, uy_3 = 0., 0.
    for t in tqdm(np.arange(Ts, tf + Ts, Ts), ncols=100):

        # At every instant, if no data is received the interval estimation inflates
        x1int = x1int + Ts * x1sp
        x1sp = x1sp + Ts * acc
        x2int = x2int + Ts * x2sp
        x2sp = x2sp + Ts * acc
        x3int = x3int + Ts * x3sp
        x3sp = x3sp + Ts * acc

        """
        High-gain continuous-discrete time observer
        """
        # Measurement received
        if data_is_received():
            # last data received
            t_last = t

            # pij(kij(t)) / Noised measurement
            y1 = x1[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            y2 = x2[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            y3 = x3[:, -1].reshape((2, 1)) + random.uniform(-0.01, 0.01)
            yv1 = np.array([[ux_1],[uy_1]]) + random.uniform(-0.01, 0.01)
            yv2 = np.array([[ux_2],[uy_2]]) + random.uniform(-0.01, 0.01)
            yv3 = np.array([[ux_3],[uy_3]]) + random.uniform(-0.01, 0.01)

            # pij_hat(kij(t))
            p1_hat_est = x1_hat_est[0:2, 0].reshape((2, 1))
            p2_hat_est = x2_hat_est[0:2, 0].reshape((2, 1))
            p3_hat_est = x3_hat_est[0:2, 0].reshape((2, 1))

            # Intervals
            x1int = IntervalVector([[y1[0, 0], y1[0, 0]], [y1[1, 0], y1[1, 0]]]).inflate(0.35)
            x1sp = IntervalVector([[yv1[0, 0], yv1[0, 0]], [yv1[1, 0], yv1[1, 0]]]).inflate(0.2)
            x2int = IntervalVector([[y2[0, 0], y2[0, 0]], [y2[1, 0], y2[1, 0]]]).inflate(0.35)
            x2sp = IntervalVector([[yv2[0, 0], yv2[0, 0]], [yv2[1, 0], yv2[1, 0]]]).inflate(0.2)
            x3int = IntervalVector([[y3[0, 0], y3[0, 0]], [y3[1, 0], y3[1, 0]]]).inflate(0.35)
            x3sp = IntervalVector([[yv3[0, 0], yv3[0, 0]], [yv3[1, 0], yv3[1, 0]]]).inflate(0.2)


        # Update of the estimation
        x1_hat_est_dot = A @ x1_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p1_hat_est - y1))
        x2_hat_est_dot = A @ x2_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p2_hat_est - y2))
        x3_hat_est_dot = A @ x3_hat_est - θ * inv(Δθ) @ K0 @ (exp(-2 * θ * (t - t_last)) * (p3_hat_est - y3))

        x1_hat_est = x1_hat_est + Ts * x1_hat_est_dot
        x2_hat_est = x2_hat_est + Ts * x2_hat_est_dot
        x3_hat_est = x3_hat_est + Ts * x3_hat_est_dot

        estimator1 = np.hstack((estimator1, x1_hat_est[0:2, 0].reshape((2, 1))))
        estimator2 = np.hstack((estimator2, x2_hat_est[0:2, 0].reshape((2, 1))))
        estimator3 = np.hstack((estimator3, x3_hat_est[0:2, 0].reshape((2, 1))))


        # Time-varying pattern
        l2 = x2int.max_diam()/2
        l3 = x3int.max_diam()/2
        new_spacing = pattern_spacing(l2,l3)
        # The triangle must not shrink to fast
        if new_spacing > spacing:
            spacing = new_spacing
        else :
            spacing += -0.05
        d0_inf = spacing - room_pat

        r1_star_x, r1_star_y, r2_star_x, r2_star_y, r3_star_x, r3_star_y = pattern(spacing)

        X = np.array([x1_hat_est[0, 0], x2_hat_est[0, 0], x3_hat_est[0, 0], x4[0, -1]]).reshape((4, 1))
        Y = np.array([x1_hat_est[1, 0], x2_hat_est[1, 0], x3_hat_est[1, 0], x4[1, -1]]).reshape((4, 1))

        ux_1 = -kp * A1[0, :] @ (x1_hat_est[0, 0] - X - r1_star_x)
        uy_1 = -kp * A1[0, :] @ (x1_hat_est[1, 0] - Y - r1_star_y)

        ux_2 = -kp * A1[1, :] @ (x2_hat_est[0, 0] - X - r2_star_x)
        uy_2 = -kp * A1[1, :] @ (x2_hat_est[1, 0] - Y - r2_star_y)

        ux_3 = -kp * A1[2, :] @ (x3_hat_est[0, 0] - X - r3_star_x)
        uy_3 = -kp * A1[2, :] @ (x3_hat_est[1, 0] - Y - r3_star_y)


        """ 
        Collision avoidance 
        """
        d01_,d02_,d03_= [],[],[]
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
        d03.append(np.min(d03_))

        # Leader controller
        u_leader = kleader * control(x4[:, -1].reshape((2, 1)),t)

        # Trajectory
        x1 = np.hstack((x1, x1[:, -1].reshape((2, 1)) + Ts * np.vstack((ux_1, uy_1))))
        x2 = np.hstack((x2, x2[:, -1].reshape((2, 1)) + Ts * np.vstack((ux_2, uy_2))))
        x3 = np.hstack((x3, x3[:, -1].reshape((2, 1)) + Ts * np.vstack((ux_3, uy_3))))
        x4 = np.hstack((x4, x4[:, -1].reshape((2, 1)) + Ts * u_leader))

        if kdisp%1==0:
            clear(ax)
            # ax.plot(x1[0, :], x1[1, :], 'c', linewidth=2)
            # ax.plot(x2[0, :], x2[1, :], 'g', linewidth=2)
            # ax.plot(x3[0, :], x3[1, :], 'b', linewidth=2)
            #ax.plot(x4[0, -1], x4[1, -1], 'r',marker = 'o')
            #ax.plot(x_hat(t),y_hat(t),'b',marker = 'o')

            # Estimations
            #ax.plot(estimator1[0, -1], estimator1[1, -1], 'olive', linewidth=1,marker='X')
            #ax.plot(estimator2[0, -1], estimator2[1, -1], 'olive', linewidth=1,marker='X',label="Continuous-discrete time estimation")
            #ax.plot(estimator3[0, -1], estimator3[1, -1], 'olive', linewidth=1,marker='X')


            ax.grid(True)
            draw_quadrotor3D(ax, x1[:, -1].reshape((2,1)))
            draw_quadrotor3D(ax, x2[:, -1].reshape((2,1)))
            draw_quadrotor3D(ax, x3[:, -1].reshape((2,1)))

            ax.set_title("Triangle patern - Robot1's point of view ")
            ax.text(0,0,7.5, "Time : t = " + str(np.round(t,4)) + " s")
            ax.text(0,0,7, "Time since last measurement : tau = " + str(np.round(t_without_data, 4)) + " s")

            # Intervals
            c2 = np.array(x2int.mid()).reshape((2, 1))
            c3 = np.array(x3int.mid()).reshape((2, 1))
            circle2 = Circle(c2, l2, edgecolor='blue', facecolor='none',label= "Interval estimation of the position of other robots")
            circle3 = Circle(c3, l3, edgecolor='blue', facecolor='none')
            # ax.add_patch(circle2)
            # ax.add_patch(circle3)
            rectangle(ax, x2int)
            rectangle(ax, x3int)

            #ax.legend(loc='lower right')
            plt.pause(0.05)


        kdisp+=1



triangle()