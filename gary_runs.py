import h5py
import sys
import os
import time
import pdb
#import seaborn as sns

import matplotlib
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import multiprocessing as mp
import fnmatch
import argparse
import scipy
from scipy.stats import maxwell
from mpl_toolkits.mplot3d import Axes3D
# import particle_orbits.taylor_field_tools
# import particle_orbits.orbit_statistics
# from particle_orbits import orbit_statistics


#import plotly.graph_objects as go
#import plotly.express as px
#from plotly.subplots import make_subplots

#from plotly.subplots import make_subplots
import concurrent.futures
#import plotly.graph_objects as go
import numpy as np

from particle_orbits.classes.Fields import getDipoleField, anotherHarris, getHarrisField, E, getUniformField, \
    getSpheromakField, nullField, getWirePotentialZ, getHarrisPotential, getWireField, getDipolePotential, \
    getHarrisCurrent, getDipoleFlux, getSpheromakFlux, getSpeiserEfield1, getSpeiserBField1, getSpeiserBField2, \
    getSpheromakPotential, getHarrisElectricField, get2DSpheromakFlux, getDoubleHarrisField, \
    getDoubleHarrisELectricField

# from particle_orbits.orbit_statistics import plot_U0_histograms

sys.path.insert(0, "./classes")

import Particle as pt
import Fields

# from path_locations import field_data_path, data_dump_path


initial_pos = np.array([1, 0, -3])
initial_vel = np.array([0.3, 0.5, 1])
initial_con = np.array([[0.35, 0.7, 0.82], [1, 0, 0]])
norbits = 1
dt = 0.01


def generate_pts(norbits, initial_pos, initial_vel, dt):
    '''intial position and intial velocity randomized
    use analyitical field instead of discrete data
    npt: number of particles
    '''

    data_dump_path = "/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                    data_dump_path=data_dump_path, write_data=False,
                    silent=False, periodic=True)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=10, z_max=6, z_min=-6)

    p.step(getDoubleHarrisField, int(norbits / dt), getDoubleHarrisELectricField)

    return p


def easy_generate_pts(initial_con):
    '''intial position and intial velocity randomized
    use analyitical field instead of discrete data
    npt: number of particles
    '''
    initial_pos = initial_con[0]
    initial_vel = initial_con[1]
    norbits = 1
    dt = 0.01
    data_dump_path = "/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                    data_dump_path=data_dump_path, write_data=False,
                    silent=False, periodic=True)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=10, z_max=5, z_min=-5)  # this is for Spheromak and Harris Sheet
    p.step(getDoubleHarrisField, int(norbits / dt), getDoubleHarrisELectricField)
    return p


def generate_pts_time(initial_con):
    '''intial position and intial velocity randomized
    use analyitical field instead of discrete data
    npt: number of particles
    '''
    initial_pos = initial_con[0]
    initial_vel = initial_con[1]
    norbits = initial_con[2]
    dt = 0.01
    data_dump_path = "/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                    data_dump_path=data_dump_path, write_data=False,
                    silent=False, periodic=True)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=10, z_max=5, z_min=-5)  # this is for Spheromak and Harris Sheet
    p.step(getDoubleHarrisField, int(norbits / dt), getDoubleHarrisELectricField)
    return p


def generate_pts_E(initial_con):
    '''intial position and intial velocity randomized
    use analyitical field instead of discrete data
    npt: number of particles
    '''
    initial_pos = initial_con[0]
    initial_vel = initial_con[1]
    sigma = initial_con[2]
    norbits = 2
    dt = 0.01
    data_dump_path = "/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                    data_dump_path=data_dump_path, write_data=False,
                    silent=False, periodic=True, sigma=sigma)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=100, z_max=10, z_min=-10)  # this is for Spheromak and Harris Sheet
    p.step(getDoubleHarrisField, int(norbits / dt), getDoubleHarrisELectricField)
    return p


def write_single_position_data(p1, filename, groupname, write_mode='w-'):
    try:
        with h5py.File(filename, write_mode) as hf:
            # create a new group if it doesn't already exist
            try:
                gp = hf.create_group(groupname)
            except ValueError:
                print("Duplicate group name: " + groupname)

            gp.create_dataset('r', data=p1.r[:-1])
            gp.create_dataset('v', data=p1.v[:-1])
            gp.create_dataset('B', data=p1.Bfield[:-1])
            # subtract 1 from the shape for initial condition

            gp.create_dataset('iter', data=[p1.iter])
            gp.create_dataset('outOfBounds', data=[p1.outOfBounds])
            gp.create_dataset('dt', data=[p1.dt])

    except IOError as fileerr:
        # for debugging
        print(fileerr)

        # if the file already exists, ask if you want to overwrite
        print("\nThis data file already exists.")
        user_input = input("Do you wish to overwrite it (o), append to it (a), or cancel data dump (c)? (o/a/c)")

        if user_input == 'o':
            write_single_position_data(p1, filename, groupname, write_mode='w')
            print("Data file will be overwritten.")
        elif user_input == 'a':
            write_single_position_data(p1, filename, groupname, write_mode='a')
            print("Data will be appended to file.")
        else:
            sys.exit("Canceling data dump.\n")
            pass


'''
def easy_generate_pts1(initial_con):

    initial_pos = initial_con[0]
    initial_vel = initial_con[1]
    norbits = 10
    dt = 0.01
    data_dump_path="/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                     data_dump_path=data_dump_path, write_data=False,
                     silent=False)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=120, z_max=90, z_min=-90) #this is for Spheromak and Harris Sheet
    p.step(getSpheromakField, int(norbits/dt))
    #p.step(getHarrisField, int(norbits/dt),getHarrisElectricField) # this is for Harris Sheet
    return p

def easy_generate_pts2(initial_con):
    initial_pos = initial_con[0]
    initial_vel = initial_con[1]
    norbits = 10
    dt = 0.01
    data_dump_path="/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                     data_dump_path=data_dump_path, write_data=False,
                     silent=False)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=120, z_max=90, z_min=-90) #this is for Spheromak and Harris Sheet
    #p.step(getSpheromakField, int(norbits/dt))
    p.step(getHarrisField, int(norbits/dt),getHarrisElectricField) # this is for Harris Sheet
    return p
'''


def plot_traj_wire(p):
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]

    steps = []
    Az = []
    for i in range(len(trajx)):
        Az.append(getWirePotentialZ(np.array([trajx[i], trajy[i], trajz[i]]), 0))
        steps.append(i)

    Az = np.asarray(Az)
    steps = np.asarray(steps)

    vel = p.get_v()
    velx = np.array(vel)[:, 0]
    vely = np.array(vel)[:, 1]
    velz = np.array(vel)[:, 2]

    q = p.get_q()

    m = p.get_m()

    pz_mech = m * velz
    pz_em = q * Az
    pz = pz_mech + pz_em
    px_mech = m * velx
    py_mech = m * vely
    p_mech = (px_mech ** 2 + py_mech ** 2 + pz_mech ** 2) ** (1 / 2)
    p_tot = pz_em + p_mech

    top_z = max(trajz)
    bottom_z = min(trajz)

    print("steps:", len(steps))
    print("pz_mech:", len(pz_mech))
    print("pz_em:", len(pz_em))
    print("pz:", len(pz))

    fig = plt.figure(figsize=plt.figaspect(1 / 4))
    # fig, axs = plt.subplots(2,2)

    ax = fig.add_subplot(141, projection='3d')

    # axs[0,0].quiver(x,y,z,Bx,By,Bz,normalize=True)
    ax.plot3D(trajx, trajy, trajz, 'r')

    wirex = []
    wirey = []
    wirez = []
    for i in range(int(top_z - bottom_z) + 25):
        wirex.append(0)
        wirey.append(0)
        wirez.append(i - 25)
    ax.plot3D(wirex, wirey, wirez, "k")

    z = np.linspace(bottom_z, top_z, 100)
    # radius of 1
    x1 = np.linspace(-1, 1, 100)
    Xc1, Zc1 = np.meshgrid(x1, z)
    Yc1 = np.sqrt(1 - Xc1 ** 2)
    ax.plot_surface(Xc1, Yc1, Zc1, alpha=0.2, color="#9400D3")
    ax.plot_surface(Xc1, -Yc1, Zc1, alpha=0.2, color="#9400D3")
    # radius of 5
    x5 = np.linspace(-5, 5, 100)
    Xc5, Zc5 = np.meshgrid(x5, z)
    Yc5 = np.sqrt(25 - Xc5 ** 2)
    ax.plot_surface(Xc5, Yc5, Zc5, alpha=0.2, color="#483D8B")
    ax.plot_surface(Xc5, -Yc5, Zc5, alpha=0.2, color="#483D8B")
    # radius of 10
    x10 = np.linspace(-10, 10, 100)
    Xc10, Zc10 = np.meshgrid(x10, z)
    Yc10 = np.sqrt(100 - Xc10 ** 2)
    ax.plot_surface(Xc10, Yc10, Zc10, alpha=0.2, color="#0000FF")
    ax.plot_surface(Xc10, -Yc10, Zc10, alpha=0.2, color="#0000FF")

    # ax.view_init(90, 0)
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_title('particle trajectory', fontsize=10)

    ax = fig.add_subplot(142)
    ax.plot(steps, pz_mech)
    ax.set_title('mechanical momentum', fontsize=10)

    ax = fig.add_subplot(143)
    ax.plot(steps, pz_em)
    ax.set_title('magnetic momentum', fontsize=10)

    ax = fig.add_subplot(144)
    ax.plot(steps, pz)
    ax.set_title('canonical momentum', fontsize=10)
    plt.show()


def plot_traj(p):
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]
    # print(traj)

    fig = plt.figure(figsize=plt.figaspect(1 / 2))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(trajx, trajy, trajz, s=0.01)
    # ax.plot3D(trajx, trajy, trajz, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # set axes to equal scale
    # ax.axis("equal")
    ax.set_title('particle trajectory', fontsize=10)
    plt.show()


def plot_orbits(p):
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]

    fig = plt.figure(figsize=plt.figaspect(1 / 2))
    # fig, axs = plt.subplots(2,2)

    ax = fig.add_subplot(121, projection='3d')

    ax.plot3D(trajx, trajy, trajz, 'r')

    ax.set_title('particle trajectory', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


# plot_spheromak_2d(generate_pts(norbits,initial_pos,initial_vel,dt))

def paper_plot_spheromak_2d(p):
    left = 0.2  # the left side of the subplots of the figure
    right = 0.85  # the right side of the subplots of the figure
    bottom = 0.15  # the bottom of the subplots of the figure
    top = 0.85  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 0.1  # the amount of height reserved for white space between subplots
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]
    r = []
    for i in range(len(trajx)):
        r_mag = np.sqrt(trajx[i] ** 2 + trajy[i] ** 2)
        r.append(r_mag)
    r = np.asarray(r)
    fig = plt.figure(num=1, figsize=(3, 3), dpi=200, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    rr, zz = np.linspace(0, 1, 100, endpoint=True), np.linspace(0, 1, 100, endpoint=True)
    RR, ZZ = np.meshgrid(rr, zz)
    values = get2DSpheromakFlux(RR, ZZ)
    ax = plt.subplot(1, 1, 1)  # (num rows, num columns, subplot position)
    plt.plot(r, trajz, color='blue', linewidth=1.5, label='Sin')
    plt.contourf(RR, ZZ, values)
    plt.xlabel('r', fontsize=9)
    plt.ylabel('z', fontsize=9)
    plt.title('Particle Trajectory and Flux Surface', fontsize=9)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.show()


def plotHarrisMomentumAndTraj(p):
    '''intend to plot the three momentum, need to give a particle, p'''
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]

    steps = []
    Ay = []
    for i in range(len(trajx)):
        Ay.append(getHarrisPotential(np.array([trajx[i], trajy[i], trajz[i]]), 0))
        steps.append(i)
    #
    Ay = np.asarray(Ay)
    steps = np.asarray(steps)

    vel = p.get_v()
    vely = np.array(vel)[:, 1]

    q = p.get_q()
    m = p.get_m()

    py_mech = m * vely
    py_em = q * Ay
    py = py_mech + py_em

    print("steps:", len(steps))
    print("py_mech:", len(py_mech))
    print("pz_em:", len(py_em))
    print("pz:", len(py))

    fig = plt.figure(figsize=plt.figaspect(1 / 4))

    ax = fig.add_subplot(141, projection='3d')

    ax.plot3D(trajx, trajy, trajz, 'r')

    ax.set_xlabel('X axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('particle trajectory', fontsize=10)

    ax = fig.add_subplot(142)
    ax.plot(steps, py_mech)
    ax.set_title('mechanical y-momentum', fontsize=10)

    ax = fig.add_subplot(143)
    ax.plot(steps, py_em)
    ax.set_title('magnetic y-momentum', fontsize=10)

    ax = fig.add_subplot(144)
    ax.plot(steps, py)
    ax.set_title('canonical y-momentum', fontsize=10)
    plt.show()


def plot_Spheromak_momentum_traj(p):
    '''intend to plot the three momentum, need to give a particle, p'''
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]

    vel = p.get_v()
    velx = np.array(vel)[:, 0]
    vely = np.array(vel)[:, 1]
    vel_theta = []
    steps = []
    Atheta = []
    for i in range(len(trajx)):
        Atheta.append(getSpheromakPotential(np.array([trajx[i], trajy[i], trajz[i]]), 0))
        steps.append(i)
        normal = [-trajy[i], trajx[i]] / (np.sqrt(trajy[i] ** 2 + trajx[i] ** 2))
        v_planar = [velx[i], vely[i]]
        v_new = np.dot(normal, v_planar)
        vel_theta.append(v_new)

    vel_theta = np.asarray(vel_theta)
    Atheta = np.asarray(Atheta)
    steps = np.asarray(steps)

    q = p.get_q()
    m = p.get_m()

    ptheta_mech = m * vel_theta
    ptheta_em = q * Atheta
    ptheta = ptheta_mech + ptheta_em

    fig = plt.figure(figsize=plt.figaspect(1 / 4))

    ax = fig.add_subplot(141, projection='3d')

    ax.plot3D(trajx, trajy, trajz, 'r')

    ax.set_xlabel('X axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('particle trajectory', fontsize=10)

    ax = fig.add_subplot(142)
    ax.plot(steps, ptheta_mech)
    ax.set_title('mechanical phi-momentum', fontsize=10)

    ax = fig.add_subplot(143)
    ax.plot(steps, ptheta_em)
    ax.set_title('magnetic phi-momentum', fontsize=10)

    ax = fig.add_subplot(144)
    ax.plot(steps, ptheta)
    ax.set_title('totak phi-momentum', fontsize=10)
    plt.show()


def plot_Spheromak_angular_momentum_traj(p):
    '''intend to plot the three momentum, need to give a particle, p'''
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]

    vel = p.get_v()
    velx = np.array(vel)[:, 0]
    vely = np.array(vel)[:, 1]
    print(vel[-1])
    vel_theta_ang = []
    steps = []
    Atheta = []
    for i in range(len(trajx)):
        Atheta.append(getSpheromakPotential(np.array([trajx[i], trajy[i], trajz[i]]), 0))
        steps.append(i)
        r_mag: float = np.sqrt(trajy[i] ** 2 + trajx[i] ** 2)
        normal: float = [-trajy[i], trajx[i]] / r_mag
        v_planar = [velx[i], vely[i]]
        v_new = np.dot(normal, v_planar)
        v_new_ang = v_new * r_mag
        vel_theta_ang.append(v_new_ang)

    vel_theta = np.asarray(vel_theta_ang)
    Atheta = np.asarray(Atheta)
    steps = np.asarray(steps)

    q = p.get_q()
    m = p.get_m()

    ptheta_mech_ang = m * vel_theta
    ptheta_em = q * Atheta
    ptheta = ptheta_mech_ang + ptheta_em

    fig = plt.figure(figsize=plt.figaspect(1 / 4))

    ax = fig.add_subplot(141, projection='3d')

    ax.plot3D(trajx, trajy, trajz, 'r')

    ax.set_xlabel('X axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('particle trajectory', fontsize=10)

    ax = fig.add_subplot(142)
    ax.plot(steps, ptheta_mech_ang)
    ax.set_title('mechanical theta-angular momentum', fontsize=10)

    ax = fig.add_subplot(143)
    ax.plot(steps, ptheta_em)
    ax.set_title('magnetic theta-momentum', fontsize=10)

    ax = fig.add_subplot(144)
    ax.plot(steps, ptheta)
    ax.set_title('canonical theta-momentum', fontsize=10)
    plt.show()
    # plt.savefig("Spheromak=" + str(initial_pos) + " " + str(norbits) + "orbits v=" + str(initial_vel) + ".png")


def getPos(x, y, z):
    return np.asarray([x, y, z])


def generate_velocity(nvel, mu=0, std=1):
    '''nvel: the number of velocity you want to give to a particle
    this function creates an array of velocities. A little different from
    '''
    s = []
    for i in range(0, nvel):
        v = np.random.normal(mu, std, 3)  # 0=center 1=std 3=three component
        s.append([v[0], v[1], v[2]])
    s = np.asarray(s)
    return s


def get_position(width=2, length=2, height=2, scale=2):
    '''
    only works if you want to get data point from a box
    :param width:
    :param length:
    :param height:
    :param scale: the larger the scale, the finer the "mesh" is going to be
    :return: an array of x,y,z positions
    '''
    width, length, height = width * scale, length * scale, height * scale
    initial_pos = list(np.ndindex(width + 1, length + 1, height + 1))
    offx = width / 2
    offy = length / 2
    offz = height / 2
    initial_pos = np.array(initial_pos)
    initial_pos = initial_pos - [offx, offy, offz]
    initial_pos = initial_pos / scale
    initial_pos = np.asarray(initial_pos)
    return initial_pos




def get_average_final_speed(pts):
    speed_f = []
    for pt in pts:
        new_v = pt.get_v()
        speed_f.append(np.sqrt(np.dot(new_v[-1], new_v[-1])))
    speed_f = np.asarray(speed_f)
    return np.average(speed_f)


def get_max_final_speed(pts):
    speed_f = []
    for pt in pts:
        new_v = pt.get_v()
        speed_f.append(np.sqrt(np.dot(new_v[-1], new_v[-1])))
    speed_f = np.asarray(speed_f)
    return np.amax(speed_f)



def confined_init_r(pts):
    confined = []
    for pt in pts:
        if pt.is_out_of_bounds() is False:
            confined.append(pt)
    confined_init_x = []
    confined_init_y = []
    confined_init_z = []
    for i in range(len(confined)):
        new_confined_r = confined[i].get_r()
        new_confined_init_r = new_confined_r[0]
        new_confined_init_x = new_confined_init_r[0]
        new_confined_init_y = new_confined_init_r[1]
        new_confined_init_z = new_confined_init_r[2]
        confined_init_x.append(new_confined_init_x)
        confined_init_y.append(new_confined_init_y)
        confined_init_z.append(new_confined_init_z)
    confined_init_x = np.asarray(confined_init_x)
    confined_init_y = np.asarray(confined_init_y)
    confined_init_z = np.asarray(confined_init_z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(confined_init_x, confined_init_y, confined_init_z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def init_vs_final_speed(pts):
    speed_0 = []
    speed_f = []
    for pt in pts:
        new_v = pt.get_v()
        speed_f.append(np.sqrt(np.dot(new_v[-1], new_v[-1])))
        speed_0.append(np.sqrt(np.dot(new_v[0], new_v[0])))
    speed_f, speed_0 = np.asarray(speed_f), np.asarray(speed_0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(speed_0, speed_f)
    ax.set_xlabel('init speed')
    ax.set_ylabel('final speed')
    ax.set_title('initial speed vs. final speed')
    plt.show()


def lost_final_r(pts):
    lost = []
    for pt in pts:
        if pt.is_out_of_bounds() is True:
            lost.append(pt)
    lost_final_x = []
    lost_final_y = []
    lost_final_z = []
    for i in range(len(lost)):
        new_lost_r = lost[i].get_r()
        new_lost_final_r = new_lost_r[-1]
        new_lost_init_x = new_lost_final_r[0]
        new_lost_init_y = new_lost_final_r[1]
        new_lost_init_z = new_lost_final_r[2]
        lost_final_x.append(new_lost_init_x)
        lost_final_y.append(new_lost_init_y)
        lost_final_z.append(new_lost_init_z)
    lost_final_x = np.asarray(lost_final_x)
    lost_final_y = np.asarray(lost_final_y)
    lost_final_z = np.asarray(lost_final_z)
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(lost_final_x, lost_final_y, lost_final_z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D final position')

    ax = fig.add_subplot(132)
    ax.scatter(lost_final_x, lost_final_y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('xy final position')

    ax = fig.add_subplot(133)
    ax.scatter(lost_final_x, lost_final_z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('yz final position')

    plt.show()


def init_speed_vs_final_distance(pts):
    speed_0 = []
    dist_f = []
    for pt in pts:
        new_v = pt.get_v()
        new_r = pt.get_r()
        dist_f.append(np.sqrt(np.dot(new_r[-1], new_r[-1])))
        speed_0.append(np.sqrt(np.dot(new_v[0], new_v[0])))
    speed_0, dist_f = np.asarray(speed_0), np.asarray(dist_f)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(speed_0, dist_f)
    ax.set_xlabel('init speed')
    ax.set_ylabel('final distance')
    ax.set_title('initial speed vs the final distance')
    plt.show()


def confined_init_r_and_init_v(pts):
    confined = []
    for pt in pts:
        if pt.is_out_of_bounds() is False:
            confined.append(pt)
    confined_init_x = []
    confined_init_y = []
    confined_init_z = []
    confined_init_vx = []
    confined_init_vy = []
    confined_init_vz = []
    for i in range(len(confined)):
        new_confined_r = confined[i].get_r()
        new_confined_init_r = new_confined_r[0]
        new_confined_init_x = new_confined_init_r[0]
        new_confined_init_y = new_confined_init_r[1]
        new_confined_init_z = new_confined_init_r[2]
        confined_init_x.append(new_confined_init_x)
        confined_init_y.append(new_confined_init_y)
        confined_init_z.append(new_confined_init_z)

        new_confined_v = confined[i].get_v()
        new_confined_init_v = new_confined_v[0]
        new_confined_init_vx = new_confined_init_v[0]
        new_confined_init_vy = new_confined_init_v[1]
        new_confined_init_vz = new_confined_init_v[2]
        confined_init_vx.append(new_confined_init_vx)
        confined_init_vy.append(new_confined_init_vy)
        confined_init_vz.append(new_confined_init_vz)

    confined_init_x = np.asarray(confined_init_x)
    confined_init_y = np.asarray(confined_init_y)
    confined_init_z = np.asarray(confined_init_z)
    confined_init_vx = np.asarray(confined_init_vx)
    confined_init_vy = np.asarray(confined_init_vy)
    confined_init_vz = np.asarray(confined_init_vz)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(confined_init_x, confined_init_y, confined_init_z, confined_init_vx, confined_init_vy, confined_init_vz,
              normalize=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('initial position and velocity of confined particles', fontsize=8)
    fig.show()


def print_statistics(pts):
    pos_0 = []
    speed_0 = []
    dist_f = []
    speed_f = []
    confined = []
    num_lost = 0
    for pt in pts:
        new_r = pt.get_r()
        new_pos_f = new_r[-1]
        new_pos_0 = new_r[0]
        if pt.is_out_of_bounds() is True:
            num_lost += 1
        else:
            confined.append(pt)
        new_v = pt.get_v()
        dist_f.append(np.sqrt(np.dot(new_pos_f, new_pos_f)))
        speed_f.append(np.sqrt(np.dot(new_v[-1], new_v[-1])))
        speed_0.append(np.sqrt(np.dot(new_v[0], new_v[0])))
        pos_0.append(new_pos_0)
    speed_0, dist_f, speed_f = np.asarray(speed_0), np.asarray(dist_f), np.asarray(speed_f)
    max_speed = np.amax(speed_f)
    min_speed = np.amin(speed_f)
    max_dist = np.amax(dist_f)
    min_dist = np.amin(dist_f)
    avr_dist = np.average(dist_f)
    avr_speed_0 = np.average(speed_0)
    avr_speed_f = np.average(speed_f)
    print('statistics of the series of particles')
    print(f'the max and min final speed is {max_speed} and {min_speed}')
    print(f'the max and min final distance from origin is {max_dist} and {min_dist}')
    print(f'the average distance from origin is {avr_dist}')
    print(f'the average initial speed is {avr_speed_0}')
    print(f'the average final speed is {avr_speed_f}')
    print(f'number of particle lost is  {num_lost}')
    print(f'percent of particle lost is  {num_lost / len(pts)}')
    print(f'the number of particle is  {len(pts)}')



def true_runs(nvel, width=6, length=6, height=4, scale=1):
    # get_position returns every point in a geometry based on cartesian coordinate. Test it out if you like!
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    # due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    initial_con = []
    # for every position we want to investigate, we generate nvel number of velocities.
    for pos in initial_poss:
        initial_vel = generate_velocity(nvel)
        for vel in initial_vel:
            initial_con.append([pos, vel])
    # how multiprocessing module works
    pool = mp.Pool()
    # print(initial_con)
    pts = pool.map(easy_generate_pts, initial_con)
    print_statistics(pts)
    # lost_final_r(pts)
    # plt.savefig(f'final position of lost particles. nvel = {nvel} dimension = {width}*{length}*{height}*{scale}.png')
    # confined_init_r_and_init_v(pts)
    # plt.savefig(f'quiver with confined initial position and initial velocity. nvel = {nvel} dimension = {width}*{length}*{height}*{scale}.png')
    # histogram_distribution_of_energy(pts)
    plot_speed_distribution_pts(pts)

    # confined_init_r(pts)
    # plt.savefig(f'initial position of confined particles. nvel = {nvel} dimension = {width}*{length}*{height}*{scale}.png')
    # init_vs_final_speed(pts)
    # plt.savefig(f'init vs final speed. nvel = {nvel} dimension = {width}*{length}*{height}*{scale}.png')
    # init_speed_vs_final_distance(pts)
    # plt.savefig(f'init speed vs final distance. nvel = {nvel} dimension = {width}*{length}*{height}*{scale}.png')




def true_runs_time(nvel, width=4, length=4, height=6, scale=1, start_orbit=0.1, end_orbit=5, orbit_step=0.1):
    # get_position returns every point in a geometry based on cartesian coordinate. Test it out if you like!
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    # due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    initial_con = []
    # for every position we want to investigate, we generate nvel number of velocities.
    avr_final_speed = []
    max_final_speed = []
    norbits = []
    npts = 0
    norbits_steps = int((end_orbit - start_orbit) / orbit_step)
    for i in range(norbits_steps):
        for pos in initial_poss:
            initial_vel = generate_velocity(nvel)
            for vel in initial_vel:
                initial_con.append([pos, vel, start_orbit + orbit_step * i])
        pool = mp.Pool()
        pts = pool.map(generate_pts_time, initial_con)
        initial_con = []
        avr_final_speed.append(get_average_final_speed(pts))
        max_final_speed.append(get_max_final_speed(pts))
        norbits.append((start_orbit + orbit_step * i) * 1000)
        npts = npts + len(pts)

    fig = plt.figure(figsize=plt.figaspect(1 / 2))

    ax = fig.add_subplot(121)
    ax.plot(norbits, avr_final_speed)
    ax.set_xlabel('orbits')
    ax.set_ylabel('speed')
    # set axes to equal scale
    # ax.axis("equal")
    ax.set_title('avr final speed vs norbits', fontsize=10)

    ax = fig.add_subplot(122)
    ax.plot(norbits, max_final_speed)
    ax.set_xlabel('orbits')
    ax.set_ylabel('speed')
    # set axes to equal scale
    # ax.axis("equal")
    ax.set_title('max final speed vs norbits', fontsize=10)
    plt.savefig(
        f'Double Harris Sheet Periodic {npts} points E=1J from {start_orbit * 1000} to {end_orbit * 1000}, dt=0.01.png')
    plt.show()
    print(norbits)
    print(avr_final_speed)
    print(max_final_speed)

def avr_final_speed(pts):
    final_speed = []
    for pt in pts:
        new_v = pt.get_v()
        final_speed.append(np.sqrt(np.dot(new_v[-1], new_v[-1])))
    final_speed = np.asarray(final_speed)
    return np.average(final_speed)


def true_runs_E_field(nvel, width=4, length=4, height=4, scale=1, start_E=0.1, end_E=2, E_step=0.1):
    # get_position returns every point in a geometry based on cartesian coordinate. Test it out if you like!
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    # due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    initial_con = []
    # for every position we want to investigate, we generate nvel number of velocities.
    avr_final_speed = []
    max_final_speed = []
    E_strength = []
    npts = 0
    E_steps = int((end_E - start_E) / E_step)
    for i in range(E_steps):
        for pos in initial_poss:
            initial_vel = generate_velocity(nvel)
            for vel in initial_vel:
                initial_con.append([pos, vel, start_E + E_step * i])
        pool = mp.Pool()
        pts = pool.map(generate_pts_time, initial_con)
        initial_con = []
        avr_final_speed.append(get_average_final_speed(pts))
        max_final_speed.append(get_max_final_speed(pts))
        E_strength.append(start_E + E_step * i)
        npts = npts + len(pts)

    fig = plt.figure(figsize=plt.figaspect(1 / 2))

    ax = fig.add_subplot(121)
    ax.plot(E_strength, avr_final_speed)
    ax.set_xlabel('sigma: \u03BC')
    ax.set_ylabel('speed')
    # set axes to equal scale
    # ax.axis("equal")
    ax.set_title('avr final speed vs \u03BC', fontsize=10)

    ax = fig.add_subplot(122)
    ax.plot(E_strength, max_final_speed)
    ax.set_xlabel('\u03BC')
    ax.set_ylabel('speed')
    # set axes to equal scale
    # ax.axis("equal")
    ax.set_title('max final speed vs \u03BC', fontsize=10)
    plt.savefig(
        f'Double Harris Sheet Periodic {npts} points E=\u03BC * J. \u03BC from {start_E} to {end_E}, dt=0.01.png')
    plt.show()
    print(avr_final_speed)
    print(max_final_speed)


def true_runs_E_field_for(nvel, width=10, length=10, height=6, scale=2, start_E=0.05, end_E=5, E_step=0.05):
    # get_position returns every point in a geometry based on cartesian coordinate. Test it out if you like!
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    # due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    # for every position we want to investigate, we generate nvel number of velocities.
    avr_final_speed = []
    max_final_speed = []
    E_strength = []
    npts = 0
    pts = []
    E_steps = int((end_E - start_E) / E_step)
    for i in range(E_steps):
        for pos in initial_poss:
            initial_vel = generate_velocity(nvel)
            for vel in initial_vel:
                pt = generate_pts_E([pos, vel, start_E + E_step * i])
                pts.append(pt)
        avr_final_speed.append(get_average_final_speed(pts))
        max_final_speed.append(get_max_final_speed(pts))
        print_statistics(pts)
        E_strength.append(start_E + E_step * i)
        npts = npts + len(pts)
        pts = []

    fig = plt.figure(figsize=plt.figaspect(1 / 2))

    ax = fig.add_subplot(121)
    ax.plot(E_strength, avr_final_speed)
    ax.set_xlabel('\u03BC')
    ax.set_ylabel('speed')
    # set axes to equal scale
    # ax.axis("equal")
    ax.set_title('avr final speed vs \u03BC', fontsize=10)

    ax = fig.add_subplot(122)
    ax.plot(E_strength, max_final_speed)
    ax.set_xlabel('\u03BC')
    ax.set_ylabel('speed')
    # set axes to equal scale
    # ax.axis("equal")
    ax.set_title('max final speed vs \u03BC', fontsize=10)
    plt.savefig(
        f'Double Harris Sheet Periodic {npts} points E=\u03BC * J. \u03BC from {start_E} to {end_E}, dt=0.01, 2k orbits.png')
    plt.show()
    print(avr_final_speed)
    print(max_final_speed)

def generate_pts_vary(initial_con):
    initial_pos = initial_con[0]
    initial_vel = initial_con[1]
    sigma = initial_con[2]
    by0 = initial_con[3]
    norbits = 2
    dt = 0.1
    data_dump_path = "/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                    data_dump_path=data_dump_path, write_data=False,
                    silent=False, periodic=True, sigma=sigma,by0=by0)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=10, z_max=5, z_min=-5)  # this is for Spheromak and Harris Sheet
    #p.step(getDoubleHarrisField, int(norbits / dt), getDoubleHarrisELectricField)
    p.step_only_harris(getDoubleHarrisField, int(norbits / dt), getDoubleHarrisELectricField)
    return p



def avr_final_speed_helper_for(sigma,by0, nvel=15, width=2, length=1, height=2, scale=1):
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    # due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    initial_con = []
    pts = []
    # for every position we want to investigate, we generate nvel number of velocities.
    for pos in initial_poss:
        initial_vel = generate_velocity(nvel)
        for vel in initial_vel:
            initial_con.append([pos, vel, sigma, by0])
    # how multiprocessing module works
    for con in initial_con:
        pts.append(generate_pts_vary(con))
    print(f'this batch has {len(pts)} particles')
    return avr_final_speed(pts)

def true_runs_E_By(nvel, width=6, length=6, height=4, scale=1, start_sig=0.0, end_sig=1, sig_step=0.05, start_by0=0.00, end_by0=1,by0_step=0.05):
    #get_position returns every point in a geometry based on cartesian coordinate. Test it out if you lik

    avr_final_speed = []
    sig_steps = int((end_sig-start_sig)/sig_step)
    By0_steps = int((end_by0-start_by0)/by0_step)


    ssig = np.linspace(start_sig,end_sig,sig_steps,endpoint=False)
    BBy0 = np.linspace(start_by0,end_by0,By0_steps,endpoint=False)

    sigs,By0s = np.meshgrid(ssig,BBy0)
    print(sigs.shape)
    for i in range(sigs.flatten().size):
            #avr_final_speed.append(avr_final_speed_helper(sigs.flatten()[i],By0s.flatten()[i],nvel,width,length,height,scale))
            avr_final_speed.append(avr_final_speed_helper_for(sigs.flatten()[i], By0s.flatten()[i], nvel, width, length, height, scale))

    avr_final_speed = np.asarray(avr_final_speed)
    fig = plt.figure(figsize=plt.figaspect(1 / 2))

    ax = fig.add_subplot(121, projection='3d')
    print(sigs.flatten())
    print(By0s.flatten())
    print(avr_final_speed)
    print(sigs.flatten().shape)
    print(By0s.flatten().shape)
    print(avr_final_speed.shape)
    ax.plot_surface(sigs, By0s, avr_final_speed.reshape(sigs.shape))
    ax.set_xlabel('\u03BC')
    ax.set_ylabel('By0')
    ax.set_zlabel('speed')
    ax.set_title('<v> vs \u03BC vs By0', fontsize=10)

    ax = fig.add_subplot(122)
    print(avr_final_speed.reshape(sigs.shape).shape)
    ax.contour(sigs,By0s, avr_final_speed.reshape(sigs.shape))
    ax.set_xlabel('\u03BC')
    ax.set_ylabel('By0')
    ax.set_title('Contour Plot of <v> vs \u03BC and By0.', fontsize=10)

    plt.savefig(f'Double Harris Sheet Periodic points resistivity from {start_sig} to {end_sig} By0 from {start_by0} to {end_by0}, dt=0.01, 2500 orbits.png')
    plt.show()




if __name__ == '__main__':
    '''x = np.asarray([0,1,3])
    y = np.asarray([0,1,3])
    xv=x[1]
    yv=x[2]
    print(avr_final_speed_helper(xv,yv))'''
    true_runs_E_By(6)
# if __name__ == '__main__':
#   true_runs_E_field(6)

#

'''
lol1 = [1,3,2]
freak = [4,5,6]
fig = plt.figure(figsize=plt.figaspect(1 / 4))
ax = fig.add_subplot(141)
plt.scatter(lol1,freak,s=1,alpha=0.5)
plt.show()'''


def growth_traj(p):
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]
    vel = p.get_v()
    dist = []
    step = []
    speed = []
    for i in range(len(trajx)):
        new_dist = np.sqrt(np.dot(traj[i], traj[i]))
        dist.append(new_dist)
        step.append(i)
        speed.append(np.sqrt(np.dot(vel[i], vel[i])))
    dist = np.asarray(dist)
    step = np.asarray(step)
    speed = np.asarray(speed)
    fig = plt.figure(figsize=plt.figaspect(1 / 2))

    ax = fig.add_subplot(131, projection='3d')
    ax.plot3D(trajx, trajy, trajz, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # set axes to equal scale
    # ax.axis("equal")
    ax.set_title('particle trajectory', fontsize=10)

    ax = fig.add_subplot(132)
    ax.plot(step, dist, 'r')
    ax.set_xlabel('step')
    ax.set_ylabel('distance')
    ax.set_title('distance vs. step', fontsize=10)

    ax = fig.add_subplot(133)
    ax.plot(step, speed, 'r')
    ax.set_xlabel('step')
    ax.set_ylabel('speed')
    ax.set_title('speed vs. step', fontsize=10)
    plt.show()


def harris_statistics(init_pos=initial_pos, first_vel=initial_vel, v_incre=0.1, v_max=4):
    '''
    :param init_pos: initial position of the particle
    :param init_vel: the first intiial velocity of the particle. Other initial velocity will increase
    :param v_incre: increment of velocity
    :return:
    plot the number of particles lost as
    plot the final distance from center as speed increases.
    plot the ratio between terminal SPEED and initial SPEED
    '''
    n_iter = int((v_max - first_vel[0]) / v_incre)
    init_vel = []
    terminal_dist = []
    terminal_speed_ratio = []
    init_speed = []
    z_deviation = []
    r_deviation = []
    for i in range(n_iter):
        new_init_vel = [first_vel[0] + i * v_incre, first_vel[1], first_vel[2]]
        init_vel.append(new_init_vel)
        new_p = generate_pts(norbits, init_pos, init_vel[i], dt)
        new_init_speed = np.sqrt(np.dot(new_init_vel, new_init_vel))
        init_speed.append(new_init_speed)
        new_traj = new_p.get_r()
        new_vel = new_p.get_v()
        new_terminal_pos = new_traj[-1]
        z_deviation.append(np.abs(new_terminal_pos[2]))
        r_deviation.append(np.sqrt(new_terminal_pos[0] ** 2 + new_terminal_pos[1] ** 2))
        new_terminal_vel = new_vel[-1]
        new_terminal_dist = np.sqrt(np.dot(new_terminal_pos, new_terminal_pos))
        new_terminal_speed = np.sqrt(np.dot(new_terminal_vel, new_terminal_vel))
        terminal_dist.append(new_terminal_dist)
        # terminal_speed_ratio.append(float(new_terminal_speed/new_init_speed))
        terminal_speed_ratio.append(float(new_terminal_speed))

    terminal_dist = np.asarray(terminal_dist)
    terminal_speed_ratio = np.asarray(terminal_speed_ratio)
    init_speed = np.asarray(init_speed)
    z_deviation = np.asarray(z_deviation)
    r_deviation = np.asarray(r_deviation)

    fig = plt.figure(figsize=plt.figaspect(1 / 4))
    ax = fig.add_subplot(141)
    ax.plot(init_speed, terminal_speed_ratio)
    ax.set_yscale('linear')
    ax.set(xlabel='init_speed', ylabel='vf')
    ax.set_title('vf/vi', fontsize=9)

    ax = fig.add_subplot(142)
    ax.plot(init_speed, terminal_dist)
    ax.set(xlabel='init_speed', ylabel='distance')
    ax.set_title('terminal_dist vs. init_speed', fontsize=9)

    ax = fig.add_subplot(143)
    ax.plot(init_speed, z_deviation)
    ax.set(xlabel='init_speed', ylabel='z')
    ax.set_title('z vs vi', fontsize=8)

    ax = fig.add_subplot(144)
    ax.plot(init_speed, r_deviation)
    ax.set(xlabel='init_speed', ylabel='r')
    ax.set_title('r vs vi', fontsize=9)

    plt.show()







