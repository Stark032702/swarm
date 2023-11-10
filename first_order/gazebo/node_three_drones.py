import numpy as np
import matplotlib.pyplot as plt
from codac import *
from numpy.linalg import inv
from numpy import pi
import time
import random
from matplotlib.patches import Circle
import rospy
import rospkg
from geometry_msgs.msg import Vector3, PoseStamped, Quaternion
from mrs_msgs.msg import VelocityReferenceStamped
from mrs_msgs.msg import VelocityReference, LkfStates
from mrs_msgs.srv import Vec4
from mrs_msgs.srv import VelocityReferenceStampedSrv
from std_msgs.msg import Header, Time, String, Bool
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler

# Name of the UAVs from the simulation
uav1 = "/uav9"
uav2 = "/uav21"
uav3 = "/uav27"

# initialization of the positions
x10 = np.array([0., 5.]).reshape((2, 1))  # cyan
x20 = np.array([-3., -0.5]).reshape((2, 1))  # green
x30 = np.array([3., -0.5]).reshape((2, 1))  # blue

x1 = np.array([0., 5.]).reshape((2, 1))  # cyan
x2 = np.array([-3., -0.5]).reshape((2, 1))  # green
x3 = np.array([3., 0.5]).reshape((2, 1))  # blue

## Python params ##
# Simulation
pyplot_display = False
python_dynamic = False
tf = 90.    # Duration of the mission
Ts = 0.05   # Sampling period
# Gains of pattern following command
kp = 0.35       # Tuning gains for pattern following
kleader = 0.45  # Tuning gain of the leader
# Trajectory to follow
height = 1.65   # fly height of the drones
w = 0.05        # angular speed on the circle
lx = 8          # radius of the circle
ly = 10
v_max = 1.5
# Loss of connection
chance = 1              # % chance of loosing connection every instant
t_max_loss = 2.         # maximum duration of a connection loss
t_without_loss = 20.    # minimum of time between 2 loss of connections
# Diplay
n_display = 1 # numbers of lines of dynamic display in the terminal

"""
## Install codac ##
pip3 install --upgrade pip
pip3 install codac
"""

"""
Dans le terminal

video :
ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :1 -c:v libx264 -qp 0 -preset ultrafast capture.mkv
(echo $DISPLAY    -i :0.0 into -i :1)

screenshot :
scrot -s interval_estimation.png
"""




"""
    ######################      ROS FUNCTIONS       ######################
"""

# Display in the terminal
def delete_previous_lines(n):
    print('\033[{}F'.format(n) + '\033[K' * n, end='')

# Update the state of the drones
# uav 1
def callback_posx1(data): # x
    global x1
    x1[0,0] = data.pos
def callback_posy1(data): # y
    global x1
    x1[1,0] = data.pos
# uav 2
def callback_posx2(data): # x
    global x2
    x2[0,0] = data.pos
def callback_posy2(data): # y
    global x2
    x2[1,0] = data.pos
# uav 3
def callback_posx3(data): # x
    global x3
    x3[0,0] = data.pos
def callback_posy3(data): # y
    global x3
    x3[1,0] = data.pos

# Ros initilization
def init():
    # Intialize the client node
    rospy.init_node('node_control')

    # Subscibe to position of the drones
    rospy.Subscriber(uav1 + "/odometry/lkf_states_x", LkfStates, callback_posx1)
    rospy.Subscriber(uav1 + "/odometry/lkf_states_y", LkfStates, callback_posy1)
    rospy.Subscriber(uav2 + "/odometry/lkf_states_x", LkfStates, callback_posx2)
    rospy.Subscriber(uav2 + "/odometry/lkf_states_y", LkfStates, callback_posy2)
    rospy.Subscriber(uav3 + "/odometry/lkf_states_x", LkfStates, callback_posx3)
    rospy.Subscriber(uav3 + "/odometry/lkf_states_y", LkfStates, callback_posy3)

    # Wait until the services are available
    rospy.wait_for_service(uav1 + "/control_manager/velocity_reference")
    rospy.wait_for_service(uav2 + "/control_manager/velocity_reference")
    rospy.wait_for_service(uav3 + "/control_manager/velocity_reference")

    print("[INFO] : Connection established between Gazebo and guidance")

# Beginning of the mission
def starting():
    global x10,x20,x30
    print("Preparing for starting ")
    rospy.wait_for_service(uav1 + "/control_manager/goto")
    rospy.wait_for_service(uav2 + "/control_manager/goto")
    rospy.wait_for_service(uav3 + "/control_manager/goto")

    # Drone 1
    try:
        velocity_proxy = rospy.ServiceProxy(uav1 + "/control_manager/goto", Vec4)
        request = [x10[0, 0], x10[1, 0], height, 0.0]
        response = velocity_proxy(request)
    except rospy.ServiceException as e:
        rospy.logerr('Error calling service : %s' % str(e))
    # Drone 2
    try:
        velocity_proxy = rospy.ServiceProxy(uav2 + "/control_manager/goto", Vec4)
        request = [x20[0, 0], x20[1, 0], height, 0.0]
        response = velocity_proxy(request)
    except rospy.ServiceException as e:
        rospy.logerr('Error calling service : %s' % str(e))
    # Drone 3
    try:
        velocity_proxy = rospy.ServiceProxy(uav3 + "/control_manager/goto", Vec4)
        request = [x30[0, 0], x30[1, 0], height, 0.0]
        response = velocity_proxy(request)
    except rospy.ServiceException as e:
        rospy.logerr('Error calling service : %s' % str(e))

    time.sleep(17)

    # Disable MRS collision avoidance
    rqst = False
    rospy.wait_for_service(uav1 + "/control_manager/mpc_tracker/collision_avoidance")
    rospy.wait_for_service(uav2 + "/control_manager/mpc_tracker/collision_avoidance")
    rospy.wait_for_service(uav3 + "/control_manager/mpc_tracker/collision_avoidance")
    # Drone 1
    try:
        coll_proxy = rospy.ServiceProxy(uav1 + "/control_manager/mpc_tracker/collision_avoidance", SetBool)
        response = coll_proxy(rqst)
    except rospy.ServiceException as e:
        rospy.logerr('Error calling service : %s' % str(e))
    # Drone 2
    try:
        coll_proxy = rospy.ServiceProxy(uav2 + "/control_manager/mpc_tracker/collision_avoidance", SetBool)
        response = coll_proxy(rqst)
    except rospy.ServiceException as e:
        rospy.logerr('Error calling service : %s' % str(e))
    # Drone 3
    try:
        coll_proxy = rospy.ServiceProxy(uav3 + "/control_manager/mpc_tracker/collision_avoidance", SetBool)
        response = coll_proxy(rqst)
    except rospy.ServiceException as e:
        rospy.logerr('Error calling service : %s' % str(e))

# Send commands to a drone
def set_velocity(uav_name, x, y, z=0.):
    try:
        # Create proxies for the services
        velocity_proxy = rospy.ServiceProxy(uav_name + "/control_manager/velocity_reference", VelocityReferenceStampedSrv)

        # Create and complete the message
        h = Header()
        ref = VelocityReference()
        speed = Vector3()
        speed.x = float(x)
        speed.y = float(y)
        speed.z = float(z)
        ref.velocity = speed
        ref.altitude = 0.0
        ref.heading = 0.0
        ref.heading_rate = 0.0
        ref.use_altitude = False
        ref.use_heading = False
        ref.use_heading_rate = False
        request = VelocityReferenceStamped()
        request.header = h
        request.reference = ref

        # Call the service
        response = velocity_proxy(request)

    except rospy.ServiceException as e:
        rospy.logerr('Error calling service : %s' % str(e))

# Visualization on Rviz
supp = 0.20
marker_pub1 = rospy.Publisher(uav1 + "/visualization_marker/drone", Marker, queue_size = 0)
marker_pub2 = rospy.Publisher(uav2 + "/visualization_marker/drone", Marker, queue_size = 0)
marker_pub3 = rospy.Publisher(uav3 + "/visualization_marker/drone", Marker, queue_size = 0)
marker_pub_box1 = rospy.Publisher(uav1 + "/visualization_marker/box", Marker, queue_size = 0)
marker_pub_box2 = rospy.Publisher(uav2 + "/visualization_marker/box", Marker, queue_size = 0)
marker_pub_box3 = rospy.Publisher(uav3 + "/visualization_marker/box", Marker, queue_size = 0)
def rviz(l2,l3,c1,c2,c3):
    global x1,x2,x3
    quat = Quaternion(*quaternion_from_euler(pi/2, 0., 0.))
    # drones
    marker1 = Marker() # drone 1
    marker2 = Marker() # drone 2
    marker3 = Marker() # drone 3
    # intervals
    marker_box1 = Marker()  # box 1
    marker_box2 = Marker()  # box 2
    marker_box3 = Marker()  # box 3

    # Drone 1
    marker1.header.frame_id = "map"
    marker1.header.stamp = rospy.Time.now()
    marker1.type = marker1.MESH_RESOURCE # set shape
    marker1.id = 0
    # scale of the marker
    marker1.scale.x = 1.0
    marker1.scale.y = 1.0
    marker1.scale.z = 1.0
    # color
    marker1.color.r = 1.0
    marker1.color.g = 0.2
    marker1.color.b = 0.0
    marker1.color.a = 1.0
    # pose of the marker
    marker1.pose.position.x = x1[0,0]
    marker1.pose.position.y = x1[1,0]
    marker1.pose.position.z = height
    marker1.pose.orientation = quat
    marker1.mesh_resource = "package://control/meshs/quadcopter.stl"

    # Intervals
    marker_box1.header.frame_id = "map"
    marker_box1.header.stamp = rospy.Time.now()
    # set shape
    marker_box1.type = 1
    marker_box1.id = 0
    # scale of the marker
    marker_box1.scale.x = l2 + supp/2
    marker_box1.scale.y = l2 + supp/2
    marker_box1.scale.z = 0.2
    # color
    marker_box1.color.r = 0.3
    marker_box1.color.g = 0.4
    marker_box1.color.b = 0.9
    marker_box1.color.a = 0.6
    # pose of the marker
    marker_box1.pose.position.x = c1[0, 0]
    marker_box1.pose.position.y = c1[1, 0]
    marker_box1.pose.position.z = height
    marker_box1.pose.orientation.x = 0.0
    marker_box1.pose.orientation.y = 0.0
    marker_box1.pose.orientation.z = 0.0
    marker_box1.pose.orientation.w = 1.0

    # Drone 2
    marker2.header.frame_id = "map"
    marker2.header.stamp = rospy.Time.now()
    # set shape
    marker2.type = marker2.MESH_RESOURCE
    marker2.id = 0
    # scale of the marker
    marker2.scale.x = 1.0
    marker2.scale.y = 1.0
    marker2.scale.z = 1.0
    # color
    marker2.color.r = 1.
    marker2.color.g = 0.2
    marker2.color.b = 0.
    marker2.color.a = 1.0
    # pose of the marker
    marker2.pose.position.x = x2[0, 0]
    marker2.pose.position.y = x2[1, 0]
    marker2.pose.position.z = height
    marker2.pose.orientation = quat
    marker2.mesh_resource = "package://control/meshs/quadcopter.stl"

    # Intervals
    marker_box2.header.frame_id = "map"
    marker_box2.header.stamp = rospy.Time.now()
    # set shape
    marker_box2.type = 1
    marker_box2.id = 0
    # scale of the marker
    marker_box2.scale.x = l2 + supp/2
    marker_box2.scale.y = l2 + supp/2
    marker_box2.scale.z = 0.2
    # color
    marker_box2.color.r = 0.3
    marker_box2.color.g = 0.4
    marker_box2.color.b = 0.9
    marker_box2.color.a = 0.6
    # pose of the marker
    marker_box2.pose.position.x = c2[0, 0]
    marker_box2.pose.position.y = c2[1, 0]
    marker_box2.pose.position.z = height
    marker_box2.pose.orientation.x = 0.0
    marker_box2.pose.orientation.y = 0.0
    marker_box2.pose.orientation.z = 0.0
    marker_box2.pose.orientation.w = 1.0

    # Drone 3
    marker3.header.frame_id = "map"
    marker3.header.stamp = rospy.Time.now()
    # set shape
    marker3.type = marker3.MESH_RESOURCE
    marker3.id = 0
    # scale of the marker
    marker3.scale.x = 1.0
    marker3.scale.y = 1.0
    marker3.scale.z = 1.0
    # color
    marker3.color.r = 1.
    marker3.color.g = 0.2
    marker3.color.b = 0.
    marker3.color.a = 1.0
    # pose of the marker
    marker3.pose.position.x = x3[0, 0]
    marker3.pose.position.y = x3[1, 0]
    marker3.pose.position.z = height
    marker3.pose.orientation = quat
    marker3.mesh_resource = "package://control/meshs/quadcopter.stl"

    # Intervals
    marker_box3.header.frame_id = "map"
    marker_box3.header.stamp = rospy.Time.now()
    # set shape
    marker_box3.type = 1
    marker_box3.id = 0
    # scale of the marker
    marker_box3.scale.x = l3 + supp/2
    marker_box3.scale.y = l3 + supp/2
    marker_box3.scale.z = 0.2
    # color
    marker_box3.color.r = 0.3
    marker_box3.color.g = 0.4
    marker_box3.color.b = 0.9
    marker_box3.color.a = 0.6
    # pose of the marker
    marker_box3.pose.position.x = c3[0, 0]
    marker_box3.pose.position.y = c3[1, 0]
    marker_box3.pose.position.z = height
    marker_box3.pose.orientation.x = 0.0
    marker_box3.pose.orientation.y = 0.0
    marker_box3.pose.orientation.z = 0.0
    marker_box3.pose.orientation.w = 1.0

    # Publish
    marker_pub1.publish(marker1)
    marker_pub2.publish(marker2)
    marker_pub3.publish(marker3)
    marker_pub_box1.publish(marker_box1)
    marker_pub_box2.publish(marker_box2)
    marker_pub_box3.publish(marker_box3)

"""
    ######################      PYTHON FUNCTIONS       ######################
"""

## Trajectory to follow
def x_hat(t): return lx*cos(w*t)
def dx_hat(t): return -lx*w*sin(w*t)
def y_hat(t): return lx*sin(w*t) #ly*sin(2*w*t)
def dy_hat(t): return lx*w*cos(w*t) #ly*w*cos(2*w*t)
def control(X,t):
    x = X[0,0]
    y = X[1,0]
    ux = 5 *(x_hat(t) - x) + dx_hat(t)
    uy = 5 *(y_hat(t) - y) + dy_hat(t)
    return np.array([[ux],[uy]])

# Spacing of the pattern
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

# Function that simulates a lost of connection
co_is_lost = False
t0_co_is_lost = 0.
t0_without_co_loss = time.time()
def lost_connection():
    global co_is_lost,timer,t0_co_is_lost,t0_without_co_loss,t_without_data
    if co_is_lost :
        t_without_data += Ts
    else :
        t_without_data = Ts
    if not(co_is_lost) and time.time() - t0_without_co_loss > t_without_loss:
        loose = random.random() < chance/100
        if loose :
            co_is_lost = True
            t0_co_is_lost = time.time()
    elif co_is_lost and time.time() - t0_co_is_lost > t_max_loss  :
        co_is_lost = False
        t0_without_co_loss = time.time()
    # display
    delete_previous_lines(1)
    print("Drones are connected :",not(co_is_lost))

    return co_is_lost

# Function that simulates the delay for the classical observer
t_without_data = Ts # Time since the last measurement received
def data_is_received():
    global t_without_data
    boolean = random.choice([True,False,False,False,False,False])
    #boolean = True
    if boolean:
        t_without_data = Ts
    else :
        t_without_data+=Ts
    return boolean

room_pat = 1.                # room between pattern spacing and interval max_lenght
room_drone_lenght = 0.5      # room between d0_inf and interval max_lenght
def pattern_spacing(l2,l3):
    max = 8
    min = 0.2
    return np.min([np.max([l2 + room_pat, min+room_pat]),max])

# limit the speed
kb = 0.6     # ~ 1/sqrt(2) diagonal max speed
def bound(u):
    if -v_max*kb <= u <= v_max*kb:
        return u
    else :
        return v_max * signe(u) * kb

test_x1 = True
test_x2 = True
test_x3 = True
def is_in_interval(x,int_x):
    if not(int_x[0].contains(x[0,0]) and int_x[1].contains(x[1,0])):
        return False


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


"""
Collision avoidance - Saturation functions
"""
# Parameters
k1 = 1
k2 = 0.5
d0_inf = 0.35
vT = 0.25

def signe(x):
    if x ==0.:
        return 0.
    elif x < 0:
        return -1.
    else :
        return 1.

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
    h = -D2  + d0_inf*vT - k2*(d02-(d0_inf**2))
    return h

def M(X,X0):
    X = X.flatten()
    X0 = X0.flatten()
    M0 = np.array([[2*(X[0]-X0[0]),2*(X[1]-X0[1])],
                  [-(X[1]-X0[1]),X[0]-X0[0]]])
    return M0


# Final display - 2D trajectories
fig_f = plt.figure(2)
ax_f = fig_f.add_subplot(111)
ax_f.set_title('Trajectories of the drones')
l_ax = 13
ax_f.axis([-l_ax, l_ax, -l_ax, l_ax])
traj_x1 = np.array([[],[]])
traj_x2 = np.array([[],[]])
traj_x3 = np.array([[],[]])
upper_born_x1_x = []
lower_born_x1_x = []
upper_born_x1_y = []
lower_born_x1_y = []
upper_born_x2_x = []
lower_born_x2_x = []
upper_born_x2_y = []
lower_born_x2_y = []
upper_born_x3_x = []
lower_born_x3_x = []
upper_born_x3_y = []
lower_born_x3_y = []

# Dynamic display - 2D motion
# fig = plt.figure(1)
# ax = fig.add_subplot(111)
# l_ax = 12
# xmin,xmax = -l_ax,l_ax
# ymin,ymax = -l_ax,l_ax

d01,d02,d03,d04 = [],[],[],[]
def clear(ax):
    ax.clear()
    ax.xmin = xmin
    ax.xmax = xmax
    ax.ymin = ymin
    ax.ymax = ymax

x4 = np.array([x_hat(0), y_hat(0)]).reshape((2, 1))
def mission():
    global d0_inf
    global x1,x2,x3,x4
    global traj_x1,traj_x2,traj_x3
    """
    DISPLACEMENT based formation with a virtual leader which follows a trajectory
    The pattern inflates when there is delay in communication
    """
    print("Beginning of the mission")
    for i in range(n_display):
        print(" ")

    # Estimated vector position and speed
    p1_hat_est = x1.copy() + eps * np.random.normal(0, nu, size=(2, 1))
    p2_hat_est = x2.copy() + eps * np.random.normal(0, nu, size=(2, 1))
    p3_hat_est = x3.copy() + eps * np.random.normal(0, nu, size=(2, 1))

    estimator1 = p1_hat_est.copy()
    estimator2 = p2_hat_est.copy()
    estimator3 = p3_hat_est.copy()

    x1_hat_est = np.vstack((p1_hat_est, np.zeros((2, 1))))
    x2_hat_est = np.vstack((p2_hat_est, np.zeros((2, 1))))
    x3_hat_est = np.vstack((p3_hat_est, np.zeros((2, 1))))

    # Measurement
    y1 = np.zeros((2, 1))  # measurement of pj only
    y2 = np.zeros((2, 1))  # measurement of pj only
    y3 = np.zeros((2, 1))  # measurement of pj only

    # Adjacency matrix
    A1 = np.ones((4, 4)) - np.eye(4)



    """
    Interval estimation of the other drones (x1's point of view)
    """
    x1int = IntervalVector([[x1[0, 0], x1[0, 0]], [x1[1, 0], x1[1, 0]]])
    x2int = IntervalVector([[x2[0, 0], x2[0, 0]], [x2[1, 0], x2[1, 0]]])
    x3int = IntervalVector([[x3[0, 0], x3[0, 0]], [x3[1, 0], x3[1, 0]]])
    c1 = np.array(x1int.mid()).reshape((2, 1))
    c2 = np.array(x2int.mid()).reshape((2, 1))
    c3 = np.array(x3int.mid()).reshape((2, 1))
    speed = IntervalVector([[-v_max,v_max],[-v_max,v_max]]).inflate(0.52)    # bounded speed

    spacing = 2.                                   # initialisation of the pattern spacing
    dt_loop = Ts
    t_last = 0. # Last data received
    t0 = time.time()      # mission time
    t = time.time()-t0
    t_mission = []
    while t < tf :
        t0_loop = time.time()
        t = time.time() - t0

        # At every instant, if no data is received the interval estimation inflates
        x1int = x1int + dt_loop * speed
        x2int = x2int + dt_loop * speed
        x3int = x3int + dt_loop * speed

        # Measurement received
        if data_is_received(): #not(lost_connection()) : #
            # last data received
            t_last = t

            # pij(kij(t)) / Noised measurement
            y1 = x1 + random.uniform(-0.001, 0.001)
            y2 = x2 + random.uniform(-0.001, 0.001)
            y3 = x3 + random.uniform(-0.001, 0.001)

            # pij_hat(kij(t))
            p1_hat_est = x1_hat_est[0:2, 0].reshape((2, 1))
            p2_hat_est = x2_hat_est[0:2, 0].reshape((2, 1))
            p3_hat_est = x3_hat_est[0:2, 0].reshape((2, 1))

            # Intervals
            x1int = IntervalVector([[y1[0, 0], y1[0, 0]], [y1[1, 0], y1[1, 0]]]).inflate(0.2)
            x2int = IntervalVector([[y2[0, 0], y2[0, 0]], [y2[1, 0], y2[1, 0]]]).inflate(0.2)
            x3int = IntervalVector([[y3[0, 0], y3[0, 0]], [y3[1, 0], y3[1, 0]]]).inflate(0.2)
            c1 = np.array(x1int.mid()).reshape((2, 1))
            c2 = np.array(x2int.mid()).reshape((2, 1))
            c3 = np.array(x3int.mid()).reshape((2, 1))

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
            spacing += -0.05/3
        d0_inf = 0.5


        # Formation flying
        r1_star_x, r1_star_y, r2_star_x, r2_star_y, r3_star_x, r3_star_y = pattern(spacing)
        X = np.array([x1_hat_est[0, 0], x2_hat_est[0, 0], x3_hat_est[0, 0], x4[0, -1]]).reshape((4, 1))
        Y = np.array([x1_hat_est[1, 0], x2_hat_est[1, 0], x3_hat_est[1, 0], x4[1, -1]]).reshape((4, 1))
        ux_1 = -kp * A1[0, :] @ (x1_hat_est[0, 0] - X - r1_star_x) + dx_hat(t)
        uy_1 = -kp * A1[0, :] @ (x1_hat_est[1, 0] - Y - r1_star_y) + dy_hat(t)
        ux_2 = -kp * A1[1, :] @ (x2_hat_est[0, 0] - X - r2_star_x) + dx_hat(t)
        uy_2 = -kp * A1[1, :] @ (x2_hat_est[1, 0] - Y - r2_star_y) + dy_hat(t)
        ux_3 = -kp * A1[2, :] @ (x3_hat_est[0, 0] - X - r3_star_x) + dx_hat(t)
        uy_3 = -kp * A1[2, :] @ (x3_hat_est[1, 0] - Y - r3_star_y) + dy_hat(t)

        """
        Collision avoidance
        """
        d01_,d02_,d03_= [],[],[]
        ## Drone 1
        X1 = np.vstack((x1[:, -1], x1_hat_est[2:4, 0])).reshape((4, 1))  # x1 knows where he is
        # /drone 2
        d = sqrt((x1[0,0]-x2[0,0])**2+(x1[1,0]-x2[1,0])**2)
        d01_.append(d)
        d02_.append(d)
        M0 = M(x1[:, -1].reshape((2, 1)),c2)
        satx = (M0@np.array([[ux_1[0]],[uy_1[0]]]))[0,0]
        saty = (M0@np.array([[ux_1[0]],[uy_1[0]]]))[1,0]
        h1 = H1(X1,x2_hat_est)
        h2 = H2(X1,x2_hat_est)
        SAT = np.array([[sat(h1,satx)],
                        [sat(h2,saty)]])
        u1_sat = inv(M0)@SAT
        ux_1,uy_1 = u1_sat[0,0],u1_sat[1,0]

        # /drone 3
        d = sqrt((x1[0, -1] - x3[0, -1]) ** 2 + (x1[1, -1] - x3[1, -1]) ** 2)
        d01_.append(d)
        d03_.append(d)
        M0 = M(x1[:,-1], c3)
        satx = (M0 @ np.array([[ux_1], [uy_1]]))[0, 0]
        saty = (M0 @ np.array([[ux_1], [uy_1]]))[1, 0]
        h1 = H1(X1, x3_hat_est)
        h2 = H2(X1, x3_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u1_sat = inv(M0) @ SAT
        ux_1, uy_1 = u1_sat[0, 0], u1_sat[1, 0]
        d01.append(np.min(d01_))

        ## Drone 2
        X2 = np.vstack((x2[:, -1], x2_hat_est[2:4, 0])).reshape((4, 1))  # x2 knows where he is
        # /drone 1
        M0 = M(X2,c1)
        satx = (M0 @ np.array([[ux_2[0]], [uy_2[0]]]))[0, 0]
        saty = (M0 @ np.array([[ux_2[0]], [uy_2[0]]]))[1, 0]
        h1 = H1(X2, x1_hat_est)
        h2 = H2(X2, x1_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u2_sat = inv(M0) @ SAT
        ux_2, uy_2 = u2_sat[0, 0], u2_sat[1, 0]

        # /drone 3
        d = sqrt((x3[0, -1] - x2[0, -1]) ** 2 + (x3[1, -1] - x2[1, -1]) ** 2)
        d02_.append(d)
        d03_.append(d)
        M0 = M(X2, c3)
        satx = (M0 @ np.array([[ux_2], [uy_2]]))[0, 0]
        saty = (M0 @ np.array([[ux_2], [uy_2]]))[1, 0]
        h1 = H1(X2, x3_hat_est)
        h2 = H2(X2, x3_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u2_sat = inv(M0) @ SAT
        ux_2, uy_2 = u2_sat[0, 0], u2_sat[1, 0]
        d02.append(np.min(d02_))

        ## Drone 3
        X3 = np.vstack((x3[:, -1], x3_hat_est[2:4, 0])).reshape((4, 1))  # x3 knows where he is
        # /drone 1
        M0 = M(X3, c1)
        satx = (M0 @ np.array([[ux_3[0]], [uy_3[0]]]))[0, 0]
        saty = (M0 @ np.array([[ux_3[0]], [uy_3[0]]]))[1, 0]
        h1 = H1(X3, x1_hat_est)
        h2 = H2(X3, x1_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u3_sat = inv(M0) @ SAT
        ux_3, uy_3 = u3_sat[0, 0], u3_sat[1, 0]

        # /drone 2
        M0 = M(X3, x2_hat_est)
        satx = (M0 @ np.array([[ux_3], [uy_3]]))[0, 0]
        saty = (M0 @ np.array([[ux_3], [uy_3]]))[1, 0]
        h1 = H1(X3, x2_hat_est)
        h2 = H2(X3, x2_hat_est)
        SAT = np.array([[sat(h1, satx)],
                        [sat(h2, saty)]])
        u3_sat = inv(M0) @ SAT
        ux_3, uy_3 = u3_sat[0, 0], u3_sat[1, 0]
        d03.append(np.min(d03_))

        # Leader controller
        u_leader = kleader * control(x4[:, -1].reshape((2, 1)),t)

        # Bound the speed
        ux_1 = bound(ux_1)
        uy_1 = bound(uy_1)
        ux_2 = bound(ux_2)
        uy_2 = bound(uy_2)
        ux_3 = bound(ux_3)
        uy_3 = bound(uy_3)

        # Python dynamic
        if python_dynamic:
            x1 = x1 + Ts * np.vstack((ux_1, uy_1))
            x2 = x2 + Ts * np.vstack((ux_2, uy_2))
            x3 = x3 + Ts * np.vstack((ux_3, uy_3))
        x4 = x4 + Ts * u_leader

        # Sending the commands to the Gazebo simulation
        if not(python_dynamic):
            set_velocity(uav1, ux_1, uy_1)
            set_velocity(uav2, ux_2, uy_2)
            set_velocity(uav3, ux_3, uy_3)


        dt_loop = time.time() - t0_loop

        if not(pyplot_display):
            time.sleep(max(Ts - dt_loop, 0.000001))

        if pyplot_display:
            clear(ax)
            s = spacing
            ax.plot(x4[0, -1], x4[1, -1] + s, 'r', marker='o')
            ax.plot(x4[0, -1] -s * cos(pi / 6),x4[1, -1] -s * sin(pi / 6), 'r', marker='o')
            ax.plot(x4[0, -1] + s * cos(pi / 6),x4[1, -1] -s * sin(pi / 6), 'r', marker='o')

            ax.grid(True)
            T = np.arange(0, 2 * pi, 0.01)
            abs = [x_hat(ti) for ti in T]
            ord = [y_hat(ti) for ti in T]
            ax.plot(abs, ord, color='indianred',linestyle='dashed')
            ax.plot([x1[0, -1]], [x1[1, -1]], 'o', linewidth=2, color="hotpink", label="Robot 1")
            ax.plot([x2[0, -1], x3[0, -1]],[x2[1, -1], x3[1, -1]], 'o', linewidth=2,alpha=0.5, color = "orange",label="Other robots")
            ax.set_title("Triangle patern - Robot1's point of view ")
            ax.axis([-l_ax, l_ax, -l_ax, l_ax])
            ax.text(4.5,7, "Time : t = " + str(np.round(t,4)) + " s")
            ax.text(-1.5, 6, "Time since last measurement : tau = " + str(np.round(t_without_data, 4)) + " s")

            # Intervals
            circle2 = Circle(c2, l2, edgecolor='blue', facecolor='none',label= "Interval estimation of the position of other robots")
            circle3 = Circle(c3, l3, edgecolor='blue', facecolor='none')
            ax.add_patch(circle2)
            ax.add_patch(circle3)

            ax.legend(loc='lower right')
            plt.pause(max(Ts - dt_loop,0.00001))

        # Dynamic display in the terminal
        delete_previous_lines(n_display)
        print("Time since last measurement :", np.round(10 * t_without_data) / 10)


        ## Rviz
        rviz(l2,l3,c1,c2,c3)

        ## Record
        # Trajectories
        t_mission.append(time.time()-t0)
        traj_x1 = np.hstack((traj_x1, x1))
        traj_x2 = np.hstack((traj_x2, x2))
        traj_x3 = np.hstack((traj_x3, x3))
        # Intervals
        upper_born_x1_x.append(x1int[0].ub())
        lower_born_x1_x.append(x1int[0].lb())
        upper_born_x1_y.append(x1int[1].ub())
        lower_born_x1_y.append(x1int[1].lb())
        upper_born_x2_x.append(x2int[0].ub())
        lower_born_x2_x.append(x2int[0].lb())
        upper_born_x2_y.append(x2int[1].ub())
        lower_born_x2_y.append(x2int[1].lb())
        upper_born_x3_x.append(x3int[0].ub())
        lower_born_x3_x.append(x3int[0].lb())
        upper_born_x3_y.append(x3int[1].ub())
        lower_born_x3_y.append(x3int[1].lb())


    print("End of the mission")

    ###  Final display ###
    # Trajectories
    ax_f.plot(traj_x1[0, :], traj_x1[1, :], 'c', linewidth=2)
    ax_f.plot(traj_x2[0, :], traj_x2[1, :], 'g', linewidth=2)
    ax_f.plot(traj_x3[0, :], traj_x3[1, :], 'b', linewidth=2)
    # Departure
    ax_f.plot(traj_x1[0, 0], traj_x1[1, 0], 'blue', marker = "x")
    ax_f.plot(traj_x2[0, 0], traj_x2[1, 0], 'blue', marker = "x")
    ax_f.plot(traj_x3[0, 0], traj_x3[1, 0], 'blue', marker = "x")
    # Arrival
    ax_f.plot(traj_x1[0, -1], traj_x1[1, -1], 'orange', marker = "o")
    ax_f.plot(traj_x2[0, -1], traj_x2[1, -1], 'orange', marker = "o")
    ax_f.plot(traj_x3[0, -1], traj_x3[1, -1], 'orange', marker = "o")
    #ax_f.plot(x4[0, 0], x4[1, 0], 'red', marker="o", label = 'reference point (center of the formation)')
    T = np.arange(0, 2 * pi, 0.01)
    abs = [x_hat(ti/w) for ti in T]  # lx * np.cos(T) #
    ord = [y_hat(ti/w) for ti in T]  # lx * np.sin(T) #
    ax_f.plot(abs, ord, color='indianred', linestyle='dashed', label ='Trajectory to follow')
    ax_f.legend(loc='lower right')
    # Derivatives
    fig_mult, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 9))
    dev_traj_x1_x = np.gradient(traj_x1[0, :], t_mission)
    dev_traj_x1_y = np.gradient(traj_x1[1, :], t_mission)
    dev_traj_x2_x = np.gradient(traj_x2[0, :], t_mission)
    dev_traj_x2_y = np.gradient(traj_x2[1, :], t_mission)
    dev_traj_x3_x = np.gradient(traj_x3[0, :], t_mission)
    dev_traj_x3_y = np.gradient(traj_x3[1, :], t_mission)
    dev_sec_traj_x1_x = np.gradient(dev_traj_x1_x, t_mission)
    dev_sec_traj_x1_y = np.gradient(dev_traj_x1_y, t_mission)
    dev_sec_traj_x2_x = np.gradient(dev_traj_x2_x, t_mission)
    dev_sec_traj_x2_y = np.gradient(dev_traj_x2_y, t_mission)
    dev_sec_traj_x3_x = np.gradient(dev_traj_x3_x, t_mission)
    dev_sec_traj_x3_y = np.gradient(dev_traj_x3_y, t_mission)
    dev_x = [dev_traj_x1_x,dev_traj_x2_x,dev_traj_x3_x]
    dev_y = [dev_traj_x1_y,dev_traj_x2_y,dev_traj_x3_y]
    dev_sec_x = [dev_sec_traj_x1_x,dev_sec_traj_x2_x,dev_sec_traj_x3_x]
    dev_sec_y = [dev_sec_traj_x1_y, dev_sec_traj_x2_y, dev_sec_traj_x3_y]
    for i in range(4):
        for j in range(3):
            ax = axes.flat[3 * i + j]
            if i == 0:
                ax.set_title('MAV n°' + str(j + 1))
                ax.plot(t_mission,dev_x[j], color = 'blue')
                if j == 0: ax.set_ylabel('vx [m.s-1]')
            if i == 1:
                ax.plot(t_mission,dev_y[j], color = 'blue')
                if j == 0: ax.set_ylabel('vy [m.s-1]')
            if i == 2:
                ax.plot(t_mission,dev_sec_x[j], color = 'blue')
                if j == 0: ax.set_ylabel('ax [m.s-2]')
            if i == 3:
                ax.plot(t_mission,dev_sec_y[j], color = 'blue')
                if j == 0: ax.set_ylabel('ay [m.s-2]')
                ax.set_xlabel("Time [s]")
    fig_mult.tight_layout()
    fig_mult.suptitle("Acceleration of the drones", fontweight='bold', y=1.)

    # Interval Estimation
    fig_tube1, axes_tube1 = plt.subplots(nrows=2, ncols=1, figsize=(7, 15))
    axt1_x1 = axes_tube1.flat[0]
    axt2_x1 = axes_tube1.flat[1]
    fig_tube2, axes_tube2 = plt.subplots(nrows=2, ncols=1, figsize=(7, 15))
    axt1_x2 = axes_tube2.flat[0]
    axt2_x2 = axes_tube2.flat[1]
    fig_tube3, axes_tube3 = plt.subplots(nrows=2, ncols=1, figsize=(7, 15))
    axt1_x3 = axes_tube3.flat[0]
    axt2_x3 = axes_tube3.flat[1]
    for i in range(2):
        for j in range(3):
            if i == 0:
                if j == 0:
                    axt1_x1.set_title('Interval estimation of Drone n°1')
                    axt1_x1.set_ylabel('x [m]')
                    axt1_x1.plot(t_mission,traj_x1[0], color = 'blue', label = "Real position")
                    axt1_x1.plot(t_mission, upper_born_x1_x, color='red',label = "Interval estimation")
                    axt1_x1.plot(t_mission, lower_born_x1_x, color='red')
                elif j == 1:
                    axt1_x2.set_title('Interval estimation of Drone n°2')
                    axt1_x2.set_ylabel('x [m]')
                    axt1_x2.plot(t_mission, traj_x2[0], color='blue', label = "Real position")
                    axt1_x2.plot(t_mission, upper_born_x2_x, color='red', label="Interval estimation")
                    axt1_x2.plot(t_mission, lower_born_x2_x, color='red')
                else:
                    axt1_x3.set_title('Interval estimation of Drone n°3')
                    axt1_x3.set_ylabel('x [m]')
                    axt1_x3.plot(t_mission, traj_x3[0], color='blue', label = "Real position")
                    axt1_x3.plot(t_mission, upper_born_x3_x, color='red', label="Interval estimation")
                    axt1_x3.plot(t_mission, lower_born_x3_x, color='red')
            if i == 1:
                if j == 0:
                    axt2_x1.set_ylabel('y [m]')
                    axt2_x1.plot(t_mission, traj_x1[1], color='blue', label = "Real position")
                    axt2_x1.plot(t_mission, upper_born_x1_y, color='red', label="Interval estimation")
                    axt2_x1.plot(t_mission, lower_born_x1_y, color='red')
                    axt2_x1.set_xlabel("Time [s]")
                elif j == 1:
                    axt2_x2.set_ylabel('y [m]')
                    axt2_x2.plot(t_mission, traj_x2[1], color='blue', label = "Real position")
                    axt2_x2.plot(t_mission, upper_born_x2_y, color='red', label="Interval estimation")
                    axt2_x2.plot(t_mission, lower_born_x2_y, color='red')
                    axt2_x2.set_xlabel("Time [s]")
                else:
                    axt2_x3.set_ylabel('y [m]')
                    axt2_x3.plot(t_mission, traj_x3[1], color='blue', label = "Real position")
                    axt2_x3.plot(t_mission, upper_born_x3_y, color='red', label="Interval estimation")
                    axt2_x3.plot(t_mission, lower_born_x3_y, color='red')
                    axt2_x3.set_xlabel("Time [s]")

    fig_tube1.tight_layout()
    fig_tube2.tight_layout()
    fig_tube3.tight_layout()
    #plt.show()


init()
starting()
mission()
