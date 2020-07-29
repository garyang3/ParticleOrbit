import h5py
import sys
import copy
import os
import time
import pdb
import seaborn as sns

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
#import particle_orbits.taylor_field_tools
#import particle_orbits.orbit_statistics
#from particle_orbits import orbit_statistics


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from plotly.subplots import make_subplots
import concurrent.futures
import plotly.graph_objects as go
import numpy as np

from classes.Fields import getDipoleField, anotherHarris, getHarrisField, E, getUniformField, \
    getSpheromakField, nullField, getWirePotentialZ, getHarrisPotential, getWireField, getDipolePotential, \
    getHarrisCurrent, getDipoleFlux, getSpheromakFlux, getSpeiserEfield1, getSpeiserBField1, getSpeiserBField2, \
    getSpheromakPotential, getHarrisElectricField, get2DSpheromakFlux, getDoubleHarrisField, \
    getDoubleHarrisELectricField, getHarrisElectricFieldStrength, getHarrisFieldStrength

#from particle_orbits.orbit_statistics import plot_U0_histograms

sys.path.insert(0, "./classes")

import Particle as pt
import Fields

#from path_locations import field_data_path, data_dump_path


initial_pos = np.array([0.35,0.7,0.82])
initial_vel = np.array([0.01, 0, 0])
initial_con = np.array([[0.35,0.7,0.82],[1, 0, 0],0.1,1])
norbits = 1
dt = 0.01

def generate_pts(norbits,initial_pos,initial_vel,dt):
    '''intial position and intial velocity randomized
    use analyitical field instead of discrete data
    npt: number of particles
    '''

    data_dump_path="/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                     data_dump_path=data_dump_path, write_data=False,
                     silent=False, periodic=False)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max = 10, z_max= 6 , z_min =-6)

    p.step(getSpheromakField, int(norbits/dt))

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
    data_dump_path="/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                     data_dump_path=data_dump_path, write_data=False,
                     silent=False,periodic=True)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=10, z_max=5, z_min=-5) #this is for Spheromak and Harris Sheet
    p.step(getDoubleHarrisField, int(norbits/dt), getDoubleHarrisELectricField)
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
    data_dump_path="/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                     data_dump_path=data_dump_path, write_data=False,
                     silent=False,periodic=True)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=10, z_max=5, z_min=-5) #this is for Spheromak and Harris Sheet
    p.step(getDoubleHarrisField, int(norbits/dt), getDoubleHarrisELectricField)
    return p

def generate_pts_E(initial_con):
    '''intial position and intial velocity randomized
    use analyitical field instead of discrete data
    npt: number of particles
    '''
    initial_pos = initial_con[0]
    initial_vel = initial_con[1]
    sigma = initial_con[2]
    norbits = 1
    dt = 0.01
    data_dump_path="/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                     data_dump_path=data_dump_path, write_data=False,
                     silent=False, periodic=True,sigma=sigma)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=100, z_max=50, z_min=-50) #this is for Spheromak and Harris Sheet
    p.step_only_harris(getDoubleHarrisField, int(norbits/dt), getDoubleHarrisELectricField)
    return p

def generate_pts_L(initial_con):
    '''intial position and intial velocity randomized
    use analyitical field instead of discrete data
    npt: number of particles
    '''
    initial_pos = initial_con[0]
    initial_vel = initial_con[1]
    L = initial_con[2]
    norbits = 1
    dt = 0.01
    data_dump_path="/Users/a1/downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=100000000,
                     data_dump_path=data_dump_path, write_data=False,
                     silent=False, periodic=True,sigma=1,by0=1,L=L)

    # maybe need to calculate the field once again using the interpolator? Guess not.
    p.set_boundaries(radius_max=100, z_max=50, z_min=-50) #this is for Spheromak and Harris Sheet
    p.step_only_harris(getDoubleHarrisField, int(norbits/dt), getDoubleHarrisELectricField)
    return p

def generate_pts_vary(initial_con):
    initial_pos = initial_con[0]
    initial_vel = initial_con[1]
    sigma = initial_con[2]
    by0 = initial_con[3]
    norbits = 0.5
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

def write_single_position_data(p1,filename,groupname,write_mode='w-'):

    try:
        with h5py.File(filename,write_mode) as hf:
            #create a new group if it doesn't already exist
            try:
                gp = hf.create_group(groupname)
            except ValueError:
                print("Duplicate group name: "+groupname)

            gp.create_dataset('r',data=p1.r[:-1])
            gp.create_dataset('v',data=p1.v[:-1])
            gp.create_dataset('B',data=p1.Bfield[:-1])
            # subtract 1 from the shape for initial condition

            gp.create_dataset('iter', data=[p1.iter])
            gp.create_dataset('outOfBounds', data=[p1.outOfBounds])
            gp.create_dataset('dt', data=[p1.dt])

    except IOError as fileerr:
        # for debugging
        print(fileerr)

        #if the file already exists, ask if you want to overwrite
        print("\nThis data file already exists.")
        user_input = input("Do you wish to overwrite it (o), append to it (a), or cancel data dump (c)? (o/a/c)")

        if user_input == 'o':
            write_single_position_data(p1,filename,groupname,write_mode='w')
            print("Data file will be overwritten.")
        elif user_input == 'a':
            write_single_position_data(p1,filename,groupname,write_mode='a')
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
    #print(traj)


    fig = plt.figure(figsize=plt.figaspect(1/2))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(trajx, trajy, trajz,s=0.01)
    #ax.plot3D(trajx, trajy, trajz, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # set axes to equal scale
    #ax.axis("equal")
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

def plot_BField(p):
    '''trying to plot the B field in the area of particle's trajectory
    trying to get 10*10*10 arrows'''

    orbit = p.get_r()
    x = np.array(orbit)[:, 0]
    xmax,xmin = np.amax(x), np.amin(x)
    y = np.array(orbit)[:, 1]
    ymax, ymin = np.amax(y), np.amin(y)
    z = np.array(orbit)[:, 2]
    zmax, zmin = np.amax(z), np.amin(z)

    #the number of arrows u want
    numofarrows = 5
    #put +- 2 to avoid the sheet situation
    xx,yy,zz = np.meshgrid(np.linspace(xmin-2,xmax+2,numofarrows),np.linspace(ymin-2,ymax+2,numofarrows),np.linspace(zmin-2,zmax+2,numofarrows))

    u,v,w = getHarrisField([xx,yy,zz],0)

    traj = plt.axes(projection='3d')
    traj.quiver(xx, yy, zz, u,v,w,normalize=True)
    plt.show()

def plot_current_surface_distance(p):
    orbit = p.get_r()
    x = np.array(orbit)[:, 0]
    xmax, xmin = np.amax(x), np.amin(x)
    y = np.array(orbit)[:, 1]
    ymax, ymin = np.amax(y), np.amin(y)
    z = np.array(orbit)[:, 2]
    zmax, zmin = np.amax(z), np.amin(z)

    # the number of arrows u want
    numofarrows = 5
    offset = 0.3

    fig = plt.figure()

    traj = fig.add_subplot(121, projection='3d')

    # put +- 2 to avoid the sheet situation
    xx, yy, zz = np.meshgrid(np.linspace(xmin-offset, xmax+offset, numofarrows), np.linspace(ymin - offset, ymax+offset, numofarrows),
                             np.linspace(zmin-offset, zmax+offset, numofarrows))

    u, v, w = getHarrisCurrent([xx, yy, zz])
    print(u,v,w)
    scalef = 2

    point = initial_pos
    normal = np.array([0, 0, 1])

    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)

    # create x,y
    xx, yy = np.meshgrid(np.linspace(xmin-offset, xmax+offset, numofarrows), np.linspace(ymin - offset, ymax+offset, numofarrows))

    # calculate corresponding z
    z0 = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    traj.plot_surface(xx, yy, z0)


    traj.quiver(xx, yy, zz, u/scalef, v/scalef, w/scalef)
    traj.plot3D(x, y, z)
    traj.set_xlabel('X axis')
    traj.set_ylabel('y axis')
    traj.set_zlabel('Z axis')
    traj.set_title('trajectory with flux surface',fontsize=10)

    ax = fig.add_subplot(122)
    steps = []
    Dist = []
    vperp = np.sqrt(initial_vel[1] ** 2 + initial_vel[2] ** 2)
    z00 = initial_pos[2]
    for i in range(len(x)):
        currentB = np.linalg.norm(getHarrisField([x[i],y[i],z[i]], 0))
        rl = 1 * vperp / (1 * currentB)
        Dist.append((z[i] - z00) / rl)
        steps.append(i)
    print(np.amax(Dist))
    print(np.amin(Dist))
    print(rl)
    Dist = np.asarray(Dist)
    steps = np.asarray(steps)
    ax.plot(steps, Dist)
    ax.set_xlabel('steps')
    ax.set_ylabel('distance')
    ax.set_title('distance w/ respect to Larmor radius', fontsize=10)

    plt.show()

def plotFluxSurfaceAndTraj(p):
    orbit = p.get_r()
    x = np.array(orbit)[:, 0]
    xmax, xmin = np.amax(x), np.amin(x)
    y = np.array(orbit)[:, 1]
    ymax, ymin = np.amax(y), np.amin(y)

    # the number of arrows u want
    numofarrows = 5
    offset = 3
    # put +- 2 to avoid the sheet situation
    xx, yy = np.meshgrid(np.linspace(xmin - offset, xmax + offset, numofarrows),
                             np.linspace(ymin - offset, ymax + offset, numofarrows))

    # z does not matter, so set it to zero
    z = getHarrisPotential([xx, yy, 0], 0)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour(xx,yy, z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.show()


def plotly_harris_traj_two_surface(p):
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]
    xmax, xmin = np.amax(trajx), np.amin(trajx)
    ymax, ymin = np.amax(trajy), np.amin(trajy)
    zmax, zmin = np.amax(trajz), np.amin(trajz)
    X, Y, Z = np.mgrid[xmin:xmax:10j, ymin:ymax:10j, zmin:zmax:10j]

    # ellipsoid
    values = getHarrisPotential([X, Y, Z], 0)
    trace1 = go.Scatter3d(x=trajx, y=trajy, z=trajz, marker=dict(size=2))
    trace2 = go.Isosurface(x=X.flatten(),
                           y=Y.flatten(),
                           z=Z.flatten(),
                           value=values.flatten(),
                           surface_count=4,  # number of isosurfaces, 2 by default: only min and max
                           colorbar_nticks=3,  # colorbar ticks correspond to isosurface values
                           caps=dict(x_show=False, y_show=False), opacity=0.5)
    data = [trace1, trace2]
    fig = go.Figure(data=data)
    fig.show()

def plot_spheromak_2d(p):
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]
    r = []
    for i in range(len(trajx)):
        r_mag = np.sqrt(trajx[i]**2+trajy[i]**2)
        r.append(r_mag)
    r = np.asarray(r)
    #trace1 = go.Scatter(x=trajx,y=trajy)
    trace1 = go.Scatter(x=r,y=trajz)

    rr, zz = np.linspace(0, 1, 100, endpoint=True), np.linspace(0, 1, 100, endpoint=True)
    RR, ZZ = np.meshgrid(rr, zz)
    values = get2DSpheromakFlux(RR, ZZ)

    trace2 = go.Contour(z=values.flatten(), x=RR.flatten(), y=ZZ.flatten())
    data = [trace1,trace2]
    fig = go.Figure(data=data)
    fig.update_layout(
        title={'text':"Particle Trajectory and Flux Surface",'y':0.9,'x':0.5,'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title="r",
        yaxis_title="z",width=600,
    height=600
    )
    fig.show()
#plot_spheromak_2d(generate_pts(norbits,initial_pos,initial_vel,dt))

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
    plt.contourf(RR,ZZ,values)
    plt.xlabel('r', fontsize=9)
    plt.ylabel('z', fontsize=9)
    plt.title('Particle Trajectory and Flux Surface', fontsize=9)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.show()

def paper_plot_Harris():
    left = 0.2  # the left side of the subplots of the figure
    right = 0.85  # the right side of the subplots of the figure
    bottom = 0.15  # the bottom of the subplots of the figure
    top = 0.85  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 0.1  # the amount of height reserved for white space between subplots


    fig = plt.figure(num=1, figsize=(3, 3), dpi=200, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    xx,yy, zz = np.linspace(-2, 2, 100, endpoint=True), np.linspace(-2, 2, 100, endpoint=True), np.linspace(-2, 2, 100, endpoint=True)
    XX,ZZ = np.meshgrid(xx, zz)
    values = getHarrisFieldStrength(XX,ZZ)
    ax = plt.subplot(1, 1, 1)  # (num rows, num columns, subplot position)
    plt.contourf(XX,ZZ,values,)
    plt.xlabel('x', fontsize=9)
    plt.ylabel('z', fontsize=9)
    plt.title('Contour Plot of Harris Magnetic Field', fontsize=9)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.show()

#paper_plot_Harris()

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
        normal = [-trajy[i],trajx[i]]/(np.sqrt(trajy[i]**2+trajx[i]**2))
        v_planar = [velx[i],vely[i]]
        v_new = np.dot(normal,v_planar)
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
        r_mag : float = np.sqrt(trajy[i] ** 2 + trajx[i] ** 2)
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
    #plt.savefig("Spheromak=" + str(initial_pos) + " " + str(norbits) + "orbits v=" + str(initial_vel) + ".png")


def getPos(x,y,z):
    return np.asarray([x,y,z])

def generate_velocity(nvel,mu=0,std=1):
    '''nvel: the number of velocity you want to give to a particle
    this function creates an array of velocities. A little different from
    '''
    s = []
    for i in range(0,nvel):
        v = np.random.normal(mu,std,3) # 0=center 1=std 3=three component
        s.append([v[0], v[1], v[2]])
    s = np.asarray(s)
    return s

def plotly_traj(p):
    traj = p.get_r()
    '''for i in range(int(len(vel))):
        print(vel[i])'''
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]
    fig = go.Figure(data=go.Scatter3d(x = trajx, y = trajy, z = trajz, marker=dict(size=0.5)))
    #write_single_position_data(p,'trialFilename.hdf5','trailgroupname')
    fig.show()

def test_plotly_traj_flux_surface1(p):
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]
    X, Y, Z = np.mgrid[-1:1:40j, -1:1:40j, -1:1:40j]

    # ellipsoid
    fig = make_subplots(rows=1,cols=2,)
    values = getSpheromakFlux([X, Y, Z], 0)

    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        surface_count=20,  # number of isosurfaces, 2 by default: only min and max
        colorbar_nticks=5,  # colorbar ticks correspond to isosurface values
        caps=dict(x_show=False, y_show=False)
    ))
    fig = go.Figure(data=go.Scatter3d(x = trajx, y = trajy, z = trajz, marker=dict(size=1)))
    fig.show()

def test_plotly_traj_flux_surface2(p):
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]
    X, Y, Z = np.mgrid[-1:1:40j, -1:1:40j, -1:1:40j]

    # ellipsoid
    values = getSpheromakFlux([X, Y, Z], 0)
    trace1 = go.Scatter3d(x = trajx, y = trajy, z = trajz, marker=dict(size=2))
    trace2 = go.Isosurface(x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        surface_count=10,  # number of isosurfaces, 2 by default: only min and max
        colorbar_nticks=5,  # colorbar ticks correspond to isosurface values
        caps=dict(x_show=False, y_show=False),opacity=0.1)
    data = [trace1, trace2]
    layout = go.Layout(xaxis=dict(domain=[0,1]),yaxis=dict(domain=[0,1]))
    fig = go.Figure(data=data,layout=layout)
    fig.show()


def test_plotly_traj_flux_surface3(p):
    traj = p.get_r()
    trajx = np.array(traj)[:, 0]
    trajy = np.array(traj)[:, 1]
    trajz = np.array(traj)[:, 2]
    X, Y, Z = np.mgrid[-10:10:90j, -10:10:90j, -20:180:90j]

    # ellipsoid
    values = getWirePotentialZ([X, Y, Z], 0)
    trace1 = go.Scatter3d(x = trajx, y = trajy, z = trajz, marker=dict(size=2))
    trace2 = go.Isosurface(x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        surface_count=20,  # number of isosurfaces, 2 by default: only min and max
        colorbar_nticks=5,  # colorbar ticks correspond to isosurface values
        caps=dict(x_show=False, y_show=False),opacity=0.2)
    data = [trace1, trace2]
    layout = go.Layout(xaxis=dict(domain=[0,1]),yaxis=dict(domain=[0,1]))
    fig = go.Figure(data=data,layout=layout)
    fig.show()

def plotly_harris_field_lines():
    X, Y, Z = np.mgrid[-0.2:0.2:6j, -0.2:0.2:6j, -1.1:1.1:6j]
    xx,yy,zz = np.mgrid[-0.2:0.2:6j, -0.2:0.2:6j, -1.1:1.1:6j]
    u, v, w = getHarrisField([X, Y, Z],0)
    print(u.flatten())
    fig = go.Figure(data=go.Cone(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        u=u.flatten(),
        v=v.flatten(),
        w=w.flatten(),colorscale='Blues',
    sizeref=0.5))

    fig.show()

#plotly_harris_field_lines()
def get_position(width=2, length=2, height=2, scale=2):
    '''
    only works if you want to get data point from a box
    :param width:
    :param length:
    :param height:
    :param scale: the larger the scale, the finer the "mesh" is going to be
    :return: an array of x,y,z positions
    '''
    width, length, height = width*scale, length*scale, height*scale
    initial_pos = list(np.ndindex(width+1,length+1,height+1))
    offx = width/2
    offy = length / 2
    offz = height / 2
    initial_pos = np.array(initial_pos)
    initial_pos = initial_pos - [offx,offy,offz]
    initial_pos = initial_pos/scale
    initial_pos = np.asarray(initial_pos)
    return initial_pos

print(len(get_position(20,20,20,1)))

def plot_speed_distribution(vs1,vs2):
    speed1 = []
    speed2 = []
    for v in vs1:
        speed1.append(np.sqrt(np.dot(v,v)))
    for v in vs2:
        speed2.append(np.sqrt(np.dot(v,v)))
    speed1 = np.asarray(speed1)
    speed2 = np.asarray(speed2)
    sns.distplot(speed1, hist=False, kde=True,
                 kde_kws={'linewidth': 1}, label='initial speed distribution')
    sns.distplot(speed2, hist=True, kde=True,norm_hist=True)

    plt.title('Initial and Final Speed Distribution')
    plt.xlabel('Speed')
    plt.ylabel('Density')

    plt.show()

def plot_speed_distribution_5(vs1,vs2,vs3,vs4):
    speed1 = []
    speed2 = []
    speed3 = []
    speed4 = []
    speed5 = []
    for v in vs1:
        speed1.append(np.sqrt(np.dot(v,v)))
    for v in vs2:
        speed2.append(np.sqrt(np.dot(v,v)))
    for v in vs3:
        speed3.append(np.sqrt(np.dot(v, v)))
    for v in vs4:
        speed4.append(np.sqrt(np.dot(v, v)))
    speed1 = np.asarray(speed1)
    speed2 = np.asarray(speed2)
    speed3 = np.asarray(speed3)
    speed4 = np.asarray(speed4)
    sns.distplot(speed1, hist=False, kde=True,
                 kde_kws={'linewidth': 1}, label='Initial Speed Distribution')
    sns.distplot(speed3, hist=False, kde=True,
                 kde_kws={'linewidth': 1}, label='Speed Distribution after 300 Thermal Orbits')
    sns.distplot(speed4, hist=False, kde=True,
                 kde_kws={'linewidth': 1}, label='Speed Distribution after 600 Thermal Orbits')
    sns.distplot(speed2, hist=False, kde=True,
                 kde_kws={'linewidth': 1}, label='Speed Distribution after 1000 Thermal Orbits')

    plt.title('Speed Distributions')
    plt.xlabel('Speed')
    plt.ylabel('Density')

    plt.show()

def plot_speed_distribution_pts(pts):
    vs1 = []
    vs2 = []
    for pt in pts:
        new_v = pt.get_v()
        vs1.append(new_v[0])
        vs2.append(new_v[-1])
    plot_speed_distribution(vs1, vs2)

def plot_speed_distribution_pts_5(pts):
    vs1 = []
    vs2 = []
    vs3 = []
    vs4 = []
    for pt in pts:
        new_v = pt.get_v()
        vs1.append(new_v[0])
        vs2.append(new_v[-1])
        vs3.append(new_v[int(len(new_v)/3)])
        vs4.append(new_v[int(len(new_v) / 3*2)])
    plot_speed_distribution_5(vs1, vs2, vs3, vs4)


def get_average_final_speed(pts):
    speed_f = []
    for pt in pts:
        new_v = pt.get_v()
        speed_f.append(np.sqrt(np.dot(new_v[-1],new_v[-1])))
    speed_f = np.asarray(speed_f)
    return np.average(speed_f)

def get_max_final_speed(pts):
    speed_f = []
    for pt in pts:
        new_v = pt.get_v()
        speed_f.append(np.sqrt(np.dot(new_v[-1],new_v[-1])))
    speed_f = np.asarray(speed_f)
    return np.amax(speed_f)

def plot_U0_histograms(v0,v0_esc,ntot=30800,**kwargs):

    ax = plt.gcf().add_subplot(111)

    U0 = (0.5*v0**2).sum(axis=-1)
    U0_esc = (0.5*v0_esc**2).sum(axis=-1)

    num_esc,bins_esc = np.histogram(U0_esc,bins=10)
    num,bins = np.histogram(U0,bins=bins_esc)

    ax.bar(bins_esc[:-1],100*num_esc/ntot,align='edge',alpha=0.4,color='r',label='Escaped')
    ax.bar(bins[:-1],100*num/ntot,align='edge',alpha=0.4,color='b',label='Confined')

    ax.set_ylim(0,30)
    ax.set_xlim(0,10)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_xlabel('Particle Energy (in units of $m_p v_{th}^2$)')
    ax.set_ylabel('Fraction of Particles')
    ax.set_title('Distribution of Energy for Escaped and Confined Particles',y=1.01)

    ax.legend(bbox_to_anchor=(0.82,0.83))
    ax.grid()

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
    ax.scatter(speed_0,speed_f)
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
    ax.quiver(confined_init_x,confined_init_y,confined_init_z,confined_init_vx,confined_init_vy,confined_init_vz, normalize=True)
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
    num_slower = 0
    for pt in pts:
        new_r = pt.get_r()
        new_pos_f = new_r[-1]
        new_pos_0 = new_r[0]
        if pt.is_out_of_bounds() is True:
            num_lost+=1
        else:
            confined.append(pt)
        new_v = pt.get_v()
        if(np.dot(new_v[0], new_v[0])>np.dot(new_v[-1], new_v[-1])):
            print('1.found a particle that slowed down')
            num_slower+=1
            print(f'2.its initial velocity is {new_v[0]}')
            print(f'3.its final velocity is {new_v[-1]}')
            print(f'4.its initial position is {new_pos_0}')
            print(f'5.its final position is {new_pos_f}')
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
    print(f'the number of particle slower is {num_slower}')

def histogram_distribution_of_energy(pts):
    v0 = []
    v0_esc = []
    for pt in pts:
        if pt.is_out_of_bounds() is False:
            new_v = pt.get_v()
            v0.append(new_v[0])
        else:
            new_v = pt.get_v()
            v0_esc.append(new_v[0])
    v0,v0_esc = np.asarray(v0), np.asarray(v0_esc)
    plot_U0_histograms(v0,v0_esc,len(pts))
    plt.show()

def avr_final_speed(pts):
    final_speed = []
    for pt in pts:
        new_v = pt.get_v()
        final_speed.append(np.sqrt(np.dot(new_v[-1], new_v[-1])))
    final_speed = np.asarray(final_speed)
    return np.average(final_speed)


def true_runs(nvel, width=4, length=6, height=4, scale=1):
    #get_position returns every point in a geometry based on cartesian coordinate. Test it out if you like!
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    #due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    initial_con = []
    #for every position we want to investigate, we generate nvel number of velocities.
    for pos in initial_poss:
        initial_vel = generate_velocity(nvel)
        for vel in initial_vel:
            initial_con.append([pos, vel])
    # how multiprocessing module works
    pool = mp.Pool()
    #print(initial_con)
    pts = pool.map(easy_generate_pts,initial_con)
    print_statistics(pts)
    #lost_final_r(pts)
    #plt.savefig(f'final position of lost particles. nvel = {nvel} dimension = {width}*{length}*{height}*{scale}.png')
    #confined_init_r_and_init_v(pts)
    #plt.savefig(f'quiver with confined initial position and initial velocity. nvel = {nvel} dimension = {width}*{length}*{height}*{scale}.png')
    #histogram_distribution_of_energy(pts)
    plot_speed_distribution_pts(pts)
    #confined_init_r(pts)
    #plt.savefig(f'initial position of confined particles. nvel = {nvel} dimension = {width}*{length}*{height}*{scale}.png')
    #init_vs_final_speed(pts)
    #plt.savefig(f'init vs final speed. nvel = {nvel} dimension = {width}*{length}*{height}*{scale}.png')
    #init_speed_vs_final_distance(pts)
    #plt.savefig(f'init speed vs final distance. nvel = {nvel} dimension = {width}*{length}*{height}*{scale}.png')


def true_runs_time(nvel, width=6, length=6, height=6, scale=1,start_orbit=0.1,end_orbit=5,orbit_step=0.1):
    #get_position returns every point in a geometry based on cartesian coordinate. Test it out if you like!
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    #due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    initial_con = []
    #for every position we want to investigate, we generate nvel number of velocities.
    avr_final_speed = []
    max_final_speed = []
    norbits = []
    npts=0
    norbits_steps = int((end_orbit-start_orbit)/orbit_step)
    for i in range(norbits_steps):
        for pos in initial_poss:
            initial_vel = generate_velocity(nvel)
            for vel in initial_vel:
                initial_con.append([pos, vel, start_orbit+orbit_step*i])
        pool = mp.Pool()
        pts = pool.map(generate_pts_time,initial_con)
        initial_con=[]
        avr_final_speed.append(get_average_final_speed(pts))
        max_final_speed.append(get_max_final_speed(pts))
        norbits.append((start_orbit+orbit_step*i)*1000)
        npts = npts + len(pts)

    fig = plt.figure(figsize=plt.figaspect(1 / 2))

    ax = fig.add_subplot(121)
    ax.plot(norbits,avr_final_speed)
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
    plt.savefig(f'Double Harris Sheet Periodic {npts} points E=1J from {start_orbit*1000} to {end_orbit*1000}, dt=0.01.png')
    plt.show()
    print(norbits)
    print(avr_final_speed)
    print(max_final_speed)

def true_runs_E_field(nvel, width=4, length=4, height=4, scale=1,start_E=0.1,end_E=4.5,E_step=0.5):
    #get_position returns every point in a geometry based on cartesian coordinate. Test it out if you like!
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    #due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    initial_con = []
    #for every position we want to investigate, we generate nvel number of velocities.
    avr_final_speed = []
    max_final_speed = []
    E_strength = []
    npts=0
    E_steps = int((end_E-start_E)/E_step)
    for i in range(E_steps):
        for pos in initial_poss:
            initial_vel = generate_velocity(nvel)
            for vel in initial_vel:
                initial_con.append([pos, vel, start_E+E_step*i])
        pool = mp.Pool()
        pts = pool.map(generate_pts_E,initial_con)
        initial_con=[]
        avr_final_speed.append(get_average_final_speed(pts))
        max_final_speed.append(get_max_final_speed(pts))
        E_strength.append(start_E+E_step*i)
        npts = npts + len(pts)

    fig = plt.figure(figsize=plt.figaspect(1 / 2))

    ax = fig.add_subplot(121)
    ax.plot(E_strength,avr_final_speed)
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
    plt.savefig(f'Double Harris Sheet Periodic {npts} points E=\u03BC * J. \u03BC from {start_E} to {end_E}, dt=0.01.png')
    plt.show()
    print(avr_final_speed)
    print(max_final_speed)


def true_runs_E_field_for(nvel, width=8, length=8, height=6, scale=2,start_E=0.1,end_E=5,E_step=0.05):
    #get_position returns every point in a geometry based on cartesian coordinate. Test it out if you like!
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    #due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    #for every position we want to investigate, we generate nvel number of velocities.
    avr_final_speed = []
    max_final_speed = []
    E_strength = []
    npts=0
    pts = []
    E_steps = int((end_E-start_E)/E_step)
    for i in range(E_steps):
        for pos in initial_poss:
            initial_vel = generate_velocity(nvel)
            for vel in initial_vel:
                pt = generate_pts_E([pos, vel, start_E+E_step*i])
                pts.append(pt)
        avr_final_speed.append(get_average_final_speed(pts))
        max_final_speed.append(get_max_final_speed(pts))
        E_strength.append(start_E+E_step*i)
        npts = npts + len(pts)
        pts = []

    fig = plt.figure(figsize=plt.figaspect(1 / 2))

    ax = fig.add_subplot(121)
    ax.plot(E_strength,avr_final_speed)
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
    plt.savefig(f'Double Harris Sheet Periodic {npts} points E=\u03BC * J. \u03BC from {start_E} to {end_E}, dt=0.01.png')
    plt.show()
    print(avr_final_speed)
    print(max_final_speed)


def avr_final_speed_helper(sigma,by0, nvel=1, width=1, length=1, height=2, scale=1):
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    # due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    initial_con = []
    # for every position we want to investigate, we generate nvel number of velocities.
    for pos in initial_poss:
        initial_vel = generate_velocity(nvel)
        for vel in initial_vel:
            initial_con.append([pos, vel,sigma,by0])
    # how multiprocessing module works
    pool = mp.Pool()
    # print(initial_con)
    pts = pool.map(generate_pts_vary, initial_con)
    return avr_final_speed(pts)

def avr_final_speed_helper_for(sigma,by0, nvel=6, width=2, length=1, height=2, scale=1):
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

def true_runs_E_By(nvel, width=2, length=1, height=2, scale=1, start_sig=0.0, end_sig=1, sig_step=0.1, start_by0=0.00, end_by0=0.1,by0_step=0.01):
    #get_position returns every point in a geometry based on cartesian coordinate. Test it out if you like!


    npts=0

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
    ax.set_title('avr final speed vs \u03BC vs By0', fontsize=10)

    ax = fig.add_subplot(122)
    print(avr_final_speed.reshape(sigs.shape).shape)
    ax.contour(sigs,By0s, avr_final_speed.reshape(sigs.shape))
    ax.set_xlabel('\u03BC')
    ax.set_ylabel('By0')
    ax.set_title('Contour Plot of <v> vs E and By0.', fontsize=10)

    #plt.savefig(f'Double Harris Sheet Periodic {npts} points E=\u03BC * J. \u03BC from {start_E} to {end_E}, dt=0.01.png')
    plt.show()


def true_runs_L(nvel, width=4, length=4, height=4, scale=1,start_L=0.1,end_L=1.5,L_step=0.2):
    #get_position returns every point in a geometry based on cartesian coordinate. Test it out if you like!
    initial_poss = np.asarray(get_position(width, length, height, scale=scale))

    #due to the limitation of multiprocessing module, I have to put the initial conditions in one list
    initial_con = []
    #for every position we want to investigate, we generate nvel number of velocities.
    avr_final_speed = []
    max_final_speed = []
    Lhar = []
    npts=0
    E_steps = int((end_L-start_L)/L_step)
    for i in range(E_steps):
        for pos in initial_poss:
            initial_vel = generate_velocity(nvel)
            for vel in initial_vel:
                initial_con.append([pos, vel, start_L+L_step*i])
        pool = mp.Pool()
        pts = pool.map(generate_pts_L,initial_con)
        initial_con=[]
        avr_final_speed.append(get_average_final_speed(pts))
        max_final_speed.append(get_max_final_speed(pts))
        Lhar.append(start_L+L_step*i)
        npts = npts + len(pts)

    fig = plt.figure(figsize=plt.figaspect(1 / 2))

    ax = fig.add_subplot(111)
    ax.plot(Lhar,avr_final_speed)
    ax.set_xlabel('Lhar')
    ax.set_ylabel('speed')
    # set axes to equal scale
    # ax.axis("equal")
    ax.set_title('avr final speed vs Lhar', fontsize=10)
    plt.show()
    print(avr_final_speed)
    print(max_final_speed)


#if __name__ == '__main__':
 #  true_runs(5)

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
        new_dist = np.sqrt(np.dot(traj[i],traj[i]))
        dist.append(new_dist)
        step.append(i)
        speed.append(np.sqrt(np.dot(vel[i],vel[i])))
    dist = np.asarray(dist)
    step = np.asarray(step)
    speed = np.asarray(speed)
    fig = plt.figure(figsize=plt.figaspect(1/2))

    ax = fig.add_subplot(131, projection='3d')
    ax.plot3D(trajx, trajy, trajz, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # set axes to equal scale
    #ax.axis("equal")
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

def harris_statistics(init_pos=initial_pos, first_vel=initial_vel, v_incre=0.1,v_max=4):
    '''
    :param init_pos: initial position of the particle
    :param init_vel: the first intiial velocity of the particle. Other initial velocity will increase
    :param v_incre: increment of velocity
    :return:
    plot the number of particles lost as
    plot the final distance from center as speed increases.
    plot the ratio between terminal SPEED and initial SPEED
    '''
    n_iter = int((v_max-first_vel[0])/v_incre)
    init_vel = []
    terminal_dist = []
    terminal_speed_ratio = []
    init_speed = []
    z_deviation = []
    r_deviation = []
    for i in range(n_iter):
        new_init_vel = [first_vel[0]+i*v_incre, first_vel[1], first_vel[2]]
        init_vel.append(new_init_vel)
        new_p = generate_pts(norbits,init_pos,init_vel[i],dt)
        new_init_speed = np.sqrt(np.dot(new_init_vel, new_init_vel))
        init_speed.append(new_init_speed)
        new_traj = new_p.get_r()
        new_vel = new_p.get_v()
        new_terminal_pos = new_traj[-1]
        z_deviation.append(np.abs(new_terminal_pos[2]))
        r_deviation.append(np.sqrt(new_terminal_pos[0]**2+new_terminal_pos[1]**2))
        new_terminal_vel = new_vel[-1]
        new_terminal_dist = np.sqrt(np.dot(new_terminal_pos, new_terminal_pos))
        new_terminal_speed = np.sqrt(np.dot(new_terminal_vel, new_terminal_vel))
        terminal_dist.append(new_terminal_dist)
        #terminal_speed_ratio.append(float(new_terminal_speed/new_init_speed))
        terminal_speed_ratio.append(float(new_terminal_speed))


    terminal_dist = np.asarray(terminal_dist)
    terminal_speed_ratio = np.asarray(terminal_speed_ratio)
    init_speed = np.asarray(init_speed)
    z_deviation = np.asarray(z_deviation)
    r_deviation = np.asarray(r_deviation)


    fig = plt.figure(figsize=plt.figaspect(1 / 4))
    ax = fig.add_subplot(141)
    ax.plot(init_speed,terminal_speed_ratio)
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

def preliminary_statistics(init_pos=initial_pos, first_vel=initial_vel, v_incre=0.1,v_max=5,dist_max=50):
    '''
    :param init_pos: initial position of the particle
    :param init_vel: the first intiial velocity of the particle. Other initial velocity will increase
    :param v_incre: increment of velocity
    :return:
    plot the number of particles lost as
    plot the final distance from center as speed increases.
    plot the ratio between terminal SPEED and initial SPEED
    '''
    #n_iter = 3
    n_iter = int((v_max-first_vel[0])/v_incre)
    init_vel = []
    terminal_dist = []
    terminal_speed_ratio = []
    init_speed = []
    num_lost = []
    current_lost = 0
    for i in range(n_iter):
        new_init_vel = [first_vel[0]+i*v_incre, first_vel[1], first_vel[2]]
        init_vel.append(new_init_vel)
        new_p = generate_pts(norbits,init_pos,init_vel[i],dt)
        new_init_speed = np.sqrt(np.dot(new_init_vel, new_init_vel))
        init_speed.append(new_init_speed)
        new_traj = new_p.get_r()
        new_vel = new_p.get_v()
        new_terminal_pos = new_traj[-1]
        new_terminal_vel = new_vel[-1]
        new_terminal_dist = np.sqrt(np.dot(new_terminal_pos, new_terminal_pos))
        new_terminal_speed = np.sqrt(np.dot(new_terminal_vel, new_terminal_vel))
        terminal_dist.append(new_terminal_dist)
        terminal_speed_ratio.append(float(new_terminal_speed/new_init_speed))
        if new_terminal_dist>=dist_max:
            current_lost+=1
        num_lost.append(current_lost)
    terminal_dist = np.asarray(terminal_dist)
    terminal_speed_ratio = np.asarray(terminal_speed_ratio)
    num_lost = np.asarray(num_lost)
    init_speed = np.asarray(init_speed)
    print(terminal_dist)
    print(terminal_speed_ratio)
    print(init_speed)
    print(num_lost)


    fig = plt.figure(figsize=plt.figaspect(1 / 4))
    ax = fig.add_subplot(141)
    ax.plot(init_speed,terminal_speed_ratio)
    ax.set_yscale('linear')
    ax.set(xlabel='init_speed', ylabel='ratio')
    ax.set_title('vf/vi', fontsize=8)

    ax = fig.add_subplot(142)
    ax.plot(init_speed, terminal_dist)
    ax.set(xlabel='init_speed', ylabel='distance')
    ax.set_title('terminal_dist vs. init_speed', fontsize=8)

    ax = fig.add_subplot(143)
    ax.plot(init_speed, num_lost)
    ax.set(xlabel='init_speed', ylabel='number')
    ax.set_title('# of lost particle', fontsize=8)
    plt.show()
    #plt.savefig("Spheromak from " + str(initial_vel) + " to " + str(v_max) + "grid.png")
    ''''
    fig = make_subplots(rows=1, cols=3)

    fig.add_trace(
        go.Scatter(x=init_speed, y=terminal_speed_ratio, mode='lines', name='terminal speed ratio'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=init_speed, y=terminal_dist, mode='lines', name='terminal distance'),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=init_speed, y=num_lost, mode='lines', name='# of particle lost'),
        row=1, col=3
    )

    fig.update_layout(height=600, width=800, title_text="Terminal Distance, Speed, # of particle lost vs. init_speed")
    fig.show()
    '''



#harris_statistics()
#preliminary_statistics()
#plot_spheromak_2d(generate_pts(norbits,initial_pos,initial_vel,dt))
#plot_Spheromak_angular_momentum_traj(generate_pts(norbits, initial_pos, initial_vel, dt))
#plot_Spheromak_momentum_traj(generate_pts(norbits, initial_pos, initial_vel, dt))
#plotly_harris_traj_two_surface(generate_pts(norbits, initial_pos, initial_vel, dt))
#plot_traj(generate_pts(norbits, initial_pos, initial_vel, dt))
#plot_traj_wire(generate_pts(norbits,initial_pos,initial_vel,dt))
#test_plotly_traj_flux_surface3(generate_pts(norbits,initial_pos,initial_vel,dt))
#test_plotly_traj_flux_surface2(generate_pts(10,initial_pos,initial_vel,0.01))
#plotly_traj(generate_pts_vary(initial_con))
#test_plotly_traj_flux_surface1(generate_pts(norbits,initial_pos,initial_vel,dt))
#plotHarrisMomentumAndTraj(generate_pts(norbits,initial_pos,initial_vel,dt))
#plot_orbits(generate_pts(norbits,initial_vel,initial_vel,dt))
#plot_current_surface_distance(generate_pts(norbits,initial_pos,initial_vel,dt))
#print("Harris dt="+str(dt)+" "+str(norbits)+"orbits v="+str(initial_vel)+"r="+str(initial_pos)+"grid.png")
#plot_traj(generate_pts(norbits,initial_pos,initial_vel,dt))
#growth_traj(generate_pts(norbits, initial_pos, initial_vel, dt))

#plotly_traj(easy_generate_pts(initial_con))





def gen_pts(norbits, npt):
    initial_pos = np.array([-75, 45, 10])
    initial_vel = np.array([1, 0, 0])
    dt = 0.01
    data_dump_path = "Macintosh HD/Users/a1/Downloads"
    p = pt.particle(initial_pos, initial_vel, dt, dump_size=500,
                    data_dump_path=data_dump_path, write_data=True,
                    silent=False)

    ''' This was supposed to be responsbile for creating a grid and calculate B field off of it. Not Working
    x,y,z = np.linspace(-95, 95, 5), np.linspace(-95, 95, 5), np.linspace(-95, 95, 5)

    r = np.asarray(np.meshgrid(x, y, z, indexing='ij', sparse=True))
    #print(r)
    B = np.array(getHarrisField(r,t=0))
    print(B)
    bx = np.asarray(B)[0][:]
    #by = np.asarray(B)[1][:]
    #bz = np.asarray(B)[2][:]
    print(bx)
    '''
    r = list(np.ndindex(100,100,100))
    xoff, yoff, zoff = 50,50,50
    bx,by,bz = getDipoleField(r, 0)

    sizes = (50,50,50)

    bx = bx.reshape(sizes)  # putting the data in arrays based on location in file
    by = by.reshape(sizes)
    bz = bz.reshape(sizes)

    Harris = Fields.interpolator(bx, by, bz, x_shape=(-25,25), y_shape=(-25,25), z_shape=(-25,25), b0=1.0)
    p.set_boundaries(radius_max=1500, z_max=2000)
    p.step(Harris.field, int(norbits / p.dt) - 1)

    return p
#plot_orbits(gen_pts(800,1))

