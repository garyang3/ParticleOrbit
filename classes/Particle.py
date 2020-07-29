import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from numpy.core.multiarray import ndarray
from scipy.special import j0, j1, jn_zeros
import time
from scipy.interpolate import RegularGridInterpolator
import h5py
import pdb

import Fields

class particle(object):

    def __init__(self, init_position=None, init_velocity=None, dt=None, dump_size=10000,data_dump_path='./',write_data=True, silent=False,periodic=False,sigma=1,by0=1,L=1):
        '''
        :param init_position:
        :param init_velocity:
        Removed the if statements because python started throwing an error since all doesn't operate on a bool like
        init_position != None
        '''
        self.periodic = periodic
        self.r0 = init_position
        self.r = [init_position]
        self.v0 = init_velocity
        self.v = [init_velocity]
        self.m = 1
        self.q = 1
        self.dt = dt
        # better to count the number of iterations as opposed to the time
        self.iter = 4 # this causes some problem
        self.outOfBounds = False
        self.line_step = 0.01
        self.dump_size = dump_size
        self.dump_path = data_dump_path
        self.write_data = write_data
        self.Bfield = []
        self.silent = silent
        self.sigma = sigma
        self.By0 = by0
        self.L = L

    def get_r(self):
        return np.asarray(self.r)

    def get_v(self):
        return np.asarray(self.v)

    def get_B(self):
        return np.asarray(self.Bfield)
    def get_q(self):
        return self.q
    def get_m(self):
        return self.m
    def get_periodic(self):
        return self.periodic

    def set_boundaries(self, radius_max, z_max, z_min=0):
        '''
        :param radius_max: max radius of the cylinder
        :param z_max: max z value
        :param z_min: min z value, defaulted to 0
        Used to set the boundary of our container
        '''
        self.radius_max = radius_max
        self.z_max = z_max
        self.z_min = z_min

    def step(self, B, noOfSteps, E=Fields.nullField):
        '''
        Returns the particle object itself, after having Boris Pushed it through noOfSteps steps, in a magnetic field B.
        Inputs:
        B(r, t): function that returns the magnetic field vector
        noOfSteps: number of iterations you want to push the particle
        '''

        start = time.time()

        m = self.m
        q = self.q

        self.Bfield.append(B(self.r0, self.iter * self.dt))

        ####### debug
        # self.dump_size=100
        #######
        if self.write_data:
            dump_size = self.dump_size
        else:
            dump_size = noOfSteps + 1

        # only using the self notation for parameters which need to be updated
        # such as v, r, and t. In other words, all parameters on the left
        # hand side that we intend to preserve

        # if many, many steps are required, want to dump data in chunks and not
        # keep it all in memory
        if noOfSteps > dump_size:
            # reduce number of steps to be an integer number of data dumps
            nBlocks = int(np.floor(noOfSteps / dump_size))
            if not self.silent:
                print(
                    "Total number of steps will be {:d} blocks by {:d} iterations = {:d}".format(nBlocks + 1, dump_size,
                                                                                                 (
                                                                                                             nBlocks + 1) * dump_size))

            # set number of steps for each block to be dumpsize
            noOfSteps = dump_size

        else:
            ######nBlocks = 0
            nBlocks = noOfSteps

        for block in range(nBlocks + 1):
            if self.outOfBounds == True:
                break

            # include initial position in number of steps so that total length = num_iterations
            for ii in range(0, noOfSteps):
                # check if particle is still within cube
                last_position = self.r[-1]

                if ii == 100 and not self.silent:
                    print("Estimated total iteration time for this particle: {:4.1f} s".format(
                        noOfSteps * (time.time() - start) / self.iter))

                x = last_position[0]
                y = last_position[1]
                z = last_position[2]
                current_radius = np.sqrt(x ** 2 + y ** 2)

                if (self.outOfBounds is True or (current_radius > self.radius_max or z < self.z_min or z > self.z_max)) and self.periodic is False:
                    """
                    if self.outOfBounds is True:
                        print("HERE 1")
                    if current_radius >= self.radius_max:
                        print("HERE 2")
                    if z <= self.z_min:
                        print("HERE 3")
                    if z >= self.z_max:
                        print("HERE 4")
                    """
                    # set flag to true
                    self.outOfBounds = True
                    self.write_data = False

                    # pad the rest of the array with the previous position
                    '''
                    self.v.extend((iter_count - self.iter) * [0, 0, 0])
                    self.r.extend((iter_count - self.iter) * last_position)
                    '''
                    for j in range(0, (noOfSteps * (block + 1) - self.iter)):
                        self.v.append(np.asarray([0,0,0]))
                        self.r.append(last_position)
                        self.Bfield.append(B(last_position,(self.iter + j) * self.dt))
                    # quit the loop
                    break
                elif (np.abs(x) > self.radius_max/np.sqrt(2) or np.abs(y) > self.radius_max/np.sqrt(2) or z < self.z_min or z > self.z_max) and self.periodic is True:
                    if np.abs(x) > self.radius_max/np.sqrt(2):
                        x = -1 * np.sign(x) * self.radius_max/np.sqrt(2)
                    if np.abs(y) > self.radius_max/np.sqrt(2):
                        y = -1 * np.sign(y) * self.radius_max / np.sqrt(2)
                    if z < self.z_min or z > self.z_max:
                        z = -1 * np.sign(z) * self.z_max

                    self.r[-1] = np.array([x,y,z])
                    half_electrical_impulse = q * self.dt * E(self.r[ii], self.iter * self.dt) / (2 * m)
                    v_minus = self.v[-1] + half_electrical_impulse

                    t_help = q * self.dt * B(self.r[-1], self.iter * self.dt) / (2 * m)
                    v_prime = v_minus + np.cross(v_minus, t_help)

                    s_help = 2 * t_help / (1 + np.linalg.norm(t_help) ** 2)
                    v_plus = v_minus + np.cross(v_prime, s_help)
                    self.v.append(v_plus + half_electrical_impulse)
                    self.r.append(self.r[-1] + self.dt * self.v[-1])
                    self.Bfield.append(B(self.r[ii], self.iter * self.dt))
                    self.iter += 1
                else:
                    half_electrical_impulse = q * self.dt * E(self.r[ii], self.iter * self.dt) / (2 * m)
                    v_minus = self.v[-1] + half_electrical_impulse

                    t_help = q * self.dt * B(self.r[-1], self.iter * self.dt) / (2 * m)
                    v_prime = v_minus + np.cross(v_minus, t_help)

                    s_help = 2 * t_help / (1 + np.linalg.norm(t_help) ** 2)
                    v_plus = v_minus + np.cross(v_prime, s_help)
                    self.v.append(v_plus + half_electrical_impulse)
                    self.r.append(self.r[-1] + self.dt * self.v[-1])
                    self.Bfield.append(B(self.r[ii], self.iter * self.dt))
                    self.iter += 1

            # now save these time points to disk and set up new v and r arrays
            # with the inital value being the value at the current iteration to save memory

            self.iter_time = (time.time() - start) / self.iter

            if self.write_data:
                if not self.silent:
                    print("\nDumping data...")
                self.dump_data()

            if not self.silent:
                print("Time taken to execute " + str(self.iter) + " iterations: " + str(time.time() - start) + ' s')
                print("Average iteration time: {:f} s".format(self.iter_time))

        return self


    def step_only_harris(self, B, noOfSteps, E=Fields.nullField):
        '''
        Returns the particle object itself, after having Boris Pushed it through noOfSteps steps, in a magnetic field B.
        Inputs:
        B(r, t): function that returns the magnetic field vector
        noOfSteps: number of iterations you want to push the particle
        '''

        start = time.time()

        m = self.m
        q = self.q

        self.Bfield.append(B(self.r0, self.iter * self.dt))

        ####### debug
        # self.dump_size=100
        #######
        if self.write_data:
            dump_size = self.dump_size
        else:
            dump_size = noOfSteps + 1

        # only using the self notation for parameters which need to be updated
        # such as v, r, and t. In other words, all parameters on the left
        # hand side that we intend to preserve

        # if many, many steps are required, want to dump data in chunks and not
        # keep it all in memory
        if noOfSteps > dump_size:
            # reduce number of steps to be an integer number of data dumps
            nBlocks = int(np.floor(noOfSteps / dump_size))
            if not self.silent:
                print(
                    "Total number of steps will be {:d} blocks by {:d} iterations = {:d}".format(nBlocks + 1, dump_size,
                                                                                                 (
                                                                                                             nBlocks + 1) * dump_size))

            # set number of steps for each block to be dumpsize
            noOfSteps = dump_size

        else:
            ######nBlocks = 0
            nBlocks = noOfSteps

        for block in range(nBlocks + 1):
            if self.outOfBounds == True:
                break

            # include initial position in number of steps so that total length = num_iterations
            for ii in range(0, noOfSteps):
                # check if particle is still within cube
                last_position = self.r[-1]

                if ii == 100 and not self.silent:
                    print("Estimated total iteration time for this particle: {:4.1f} s".format(
                        noOfSteps * (time.time() - start) / self.iter))

                x = last_position[0]
                y = last_position[1]
                z = last_position[2]
                current_radius = np.sqrt(x ** 2 + y ** 2)

                if (self.outOfBounds is True or (current_radius > self.radius_max or z < self.z_min or z > self.z_max)) and self.periodic is False:
                    """
                    if self.outOfBounds is True:
                        print("HERE 1")
                    if current_radius >= self.radius_max:
                        print("HERE 2")
                    if z <= self.z_min:
                        print("HERE 3")
                    if z >= self.z_max:
                        print("HERE 4")
                    """
                    # set flag to true
                    self.outOfBounds = True
                    self.write_data = False

                    # pad the rest of the array with the previous position
                    '''
                    self.v.extend((iter_count - self.iter) * [0, 0, 0])
                    self.r.extend((iter_count - self.iter) * last_position)
                    '''
                    for j in range(0, (noOfSteps * (block + 1) - self.iter)):
                        self.v.append(np.asarray([0,0,0]))
                        self.r.append(last_position)
                        self.Bfield.append(B(last_position,(self.iter + j) * self.dt))
                    # quit the loop
                    break
                elif (np.abs(x) > self.radius_max/np.sqrt(2) or np.abs(y) > self.radius_max/np.sqrt(2) or z < self.z_min or z > self.z_max) and self.periodic is True:
                    if np.abs(x) > self.radius_max/np.sqrt(2):
                        x = -1 * np.sign(x) * self.radius_max/np.sqrt(2)
                    if np.abs(y) > self.radius_max/np.sqrt(2):
                        y = -1 * np.sign(y) * self.radius_max / np.sqrt(2)
                    if z < self.z_min or z > self.z_max:
                        z = -1 * np.sign(z) * self.z_max

                    self.r[-1] = np.array([x,y,z])
                    half_electrical_impulse = q * self.dt * E(self.r[ii], self.iter * self.dt, mu=self.sigma,By0=self.By0,L=self.L) / (2 * m)
                    v_minus = self.v[-1] + half_electrical_impulse

                    t_help = q * self.dt * B(self.r[-1], self.iter * self.dt, mu=self.sigma,By0=self.By0,L=self.L) / (2 * m)
                    v_prime = v_minus + np.cross(v_minus, t_help)

                    s_help = 2 * t_help / (1 + np.linalg.norm(t_help) ** 2)
                    v_plus = v_minus + np.cross(v_prime, s_help)
                    self.v.append(v_plus + half_electrical_impulse)
                    self.r.append(self.r[-1] + self.dt * self.v[-1])
                    self.Bfield.append(B(self.r[ii], self.iter * self.dt, mu=self.sigma,By0=self.By0,L=self.L))
                    self.iter += 1
                else:
                    half_electrical_impulse = q * self.dt * E(self.r[ii], self.iter * self.dt, mu=self.sigma,By0=self.By0, L=self.L) / (2 * m)
                    v_minus = self.v[-1] + half_electrical_impulse

                    t_help = q * self.dt * B(self.r[-1], self.iter * self.dt, mu=self.sigma,By0=self.By0 ,L=self.L) / (2 * m)
                    v_prime = v_minus + np.cross(v_minus, t_help)

                    s_help = 2 * t_help / (1 + np.linalg.norm(t_help) ** 2)
                    v_plus = v_minus + np.cross(v_prime, s_help)
                    self.v.append(v_plus + half_electrical_impulse)
                    self.r.append(self.r[-1] + self.dt * self.v[-1])
                    self.Bfield.append(B(self.r[ii], self.iter * self.dt, mu=self.sigma,By0=self.By0,L=self.L))
                    self.iter += 1

            # now save these time points to disk and set up new v and r arrays
            # with the inital value being the value at the current iteration to save memory

            self.iter_time = (time.time() - start) / self.iter

            if self.write_data:
                if not self.silent:
                    print("\nDumping data...")
                self.dump_data()

            if not self.silent:
                print("Time taken to execute " + str(self.iter) + " iterations: " + str(time.time() - start) + ' s')
                print("Average iteration time: {:f} s".format(self.iter_time))

        return self


    def is_out_of_bounds(self):
        return self.outOfBounds


    def easy_step(self,B,noOfSteps):
        '''
                :param B: the magnetic field vector as a function of position and time
                :param noOfSteps: the desired number of iterations
                Makes the position of the particle change according to how it should in the field
                assume no E field
                '''

        # setting the positions to 0 aside from the initial position
        # iterating the desired number of times
        for i in range(0,noOfSteps):
            current_field = B(self.r[i], i * self.dt)  # field at the desired point in space

            delta_v = (self.q/self.m) * np.cross(self.v[i], current_field)

            next_v = [self.v[i][0] + delta_v[0], self.v[i][1] + delta_v[1], self.v[i][2] + delta_v[2]]

            self.v.append(next_v)
            dx,dy,dz = self.dt*self.v[i][0],self.dt*self.v[i][1],self.dt*self.v[i][2]
            new_r = [self.r[i][0]+dx, self.r[i][1]+dy, self.r[i][2]+dz]
            self.r.append(self.r[i] + new_r)
            self.Bfield.append(current_field)

            new_position = self.r[i + 1]  # position adjusted accordingly
            x = new_position[0]
            y = new_position[1]
            z = new_position[2]
            new_radius_square = x ** 2 + y ** 2

            # fixing in case the step takes you out of bounds
            if new_radius_square >= self.radius_max ** 2 or z <= self.z_min or z >= self.z_max or self.outOfBounds:
                print("out of bounds!")
                break
        return self

    def step_field(self, B, noOfSteps):
        '''
        :param B: the magnetic field vector as a function of position and time
        :param noOfSteps: the desired number of iterations
        Makes the position of the particle change according to how it should in the field
        '''

        # setting the positions to 0 aside from the initial position
        ## question: is this necessary? I though this was done in the constructor.
        r_initial = self.r[0]
        self.r = np.zeros((noOfSteps + 1, 3))
        self.r[0] = r_initial

        # iterating the desired number of times
        for i in range(noOfSteps):
            current_field = B(self.r[i], 0)  # field at the desired point in space
            step = current_field * self.line_step / np.linalg.norm(current_field)  # the step in position depends on the field at that point
            self.r[i + 1] = self.r[i] + step  # step is added to become new position

            current_position = self.r[i + 1]  # position adjusted accordingly
            x = current_position[0]
            y = current_position[1]
            z = current_position[2]
            current_radius_square = x ** 2 + y ** 2

            # fixing in case the step takes you out of bounds
            if current_radius_square >= self.radius_max ** 2 or z <= self.z_min or z >= self.z_max or self.outOfBounds:
                self.outOfBounds = True
                last_position = self.r[i + 1]
                for j in range(i + 2, noOfSteps + 1):
                    self.r[j] = last_position
                break

            self.iter += 1


        self.r = self.r[0] #get rid of outer list container
        self.v = self.v[0]
        return self

    def reverse_step_field(self, B, noOfSteps):
        '''
        :param B: field vector as a function of position/time
        :param noOfSteps: number of iterations
        Moves back in time instead of forward, the step is subtracted instead of added
        '''
        r_initial = self.r[0]
        self.r = np.zeros((noOfSteps + 1, 3))
        self.r[0] = r_initial

        for i in range(noOfSteps):
            current_field = B(self.r[i], 0)
            step = current_field * self.line_step / np.linalg.norm(current_field)
            self.r[i + 1] = self.r[i] - step

            current_position = self.r[i + 1]
            x = current_position[0]
            y = current_position[1]
            z = current_position[2]
            current_radius_square = x ** 2 + y ** 2

            if current_radius_square >= self.radius_max ** 2 or z <= self.z_min or z >= self.z_max or self.outOfBounds:
                self.outOfBounds = True
                last_position = self.r[i + 1]
                for j in range(i + 2, noOfSteps + 1):
                    self.r[j] = last_position
                break

            self.iter += 1

        return self


    def dump_data(self,nsteps=0,append_to=None,
                  overwrite=False):
        # creates an h5 file with information about the various parameters of the particle
        initial_velocity = self.v0
        initial_position = self.r0
        if append_to is None:
            filename = self.dump_path + 'traj{:d}_'.format(nsteps) + str(initial_position) + "_" + str(initial_velocity) + ".h5"
            # getting rid of spaces to make the files easy to load
            filename = filename.replace(' ', '_')

            if self.iter < self.dump_size:
                only_one_dump = True
                dump_size = self.iter
            else:
                only_one_dump = False
                dump_size = self.dump_size

                #check if file exists for this trajectory
                #if not, initialize and write first dump

                if self.iter == dump_size:
                    #this is the first time the data is dumped
                    if overwrite or only_one_dump:
                        write_mode='w'
                    else:
                        write_mode='w-'

                else:
                    filename = append_to
                    write_mode = 'a'

                try:
                    with h5py.File(filename,write_mode) as hf:

                        # subtract 1 from the shape for initial condition
                        # leave off last value as it will be included as initial value
                        # in the next block
                        write_length = np.asarray(self.r).shape[0]
                        rdata = hf.create_dataset("r", (write_length-1,3), maxshape=(None, 3))
                        rdata[:,:] = self.r[:-1]

                        vdata = hf.create_dataset("v", (write_length-1,3), maxshape=(None, 3))
                        vdata[:,:] = self.v[:-1]

                        Bdata = hf.create_dataset("B", (write_length-1,3), maxshape=(None, 3))
                        Bdata[:,:] = self.Bfield[:-1]

                        hf.create_dataset('iter', data=[self.iter])
                        hf.create_dataset('outOfBounds', data=[self.outOfBounds])
                        hf.create_dataset('dt', data=[self.dt])

                except IOError as fileerr:
                    # for debugging
                    print(fileerr)

                    #if the file already exists, ask if you want to overwrite
                    print("\nThis data file already exists.")
                    user_input = input("Do you wish to overwrite it? (y/N)")

                    if user_input == 'y':
                        self.dump_data(nsteps=nsteps,location=location,overwrite=True)
                        print("Data file will be overwritten.  Continuing with iterations...")
                    else:
                        sys.exit("Canceling data dump.\n")
                        pass

        #otherwise append to existing datasets
        else:
            with h5py.File(filename,'a') as hf:
                rdata = hf["r"]
                vdata = hf["v"]
                Bdata = hf["B"]

                #debugging
                #print(rdata.shape)
                #print np.asarray(self.r).shape

                #append to existing data
                rdata.resize((rdata.shape[0] + dump_size, 3))
                rdata[-dump_size:] = self.r[:-1]

                vdata.resize((vdata.shape[0] + dump_size, 3))
                vdata[-dump_size:] = self.v[:-1]

                Bdata.resize((Bdata.shape[0] + dump_size, 3))
                Bdata[-dump_size:] = self.Bfield[:-1]

                hf["iter"][...] = [rdata.shape[0]]
                hf["outOfBounds"][...] = [self.outOfBounds]

        if not only_one_dump:
            self.v = [self.v[-1]]
            self.r = [self.r[-1]]
            self.Bfield = [self.Bfield[-1]]
        return self

    def save_h5(self, prefix, location=''):
        # creates an h5 file with information about the various parameters of the particle
        initial_velocity = self.v[0]
        initial_position = self.r[0]

        filename = location + prefix + str(initial_position) + "_" + str(initial_velocity) + ".h5"
        # getting rid of spaces to make the files easy to load
        filename = filename.replace(' ', '_')

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('v', data=self.v)
            hf.create_dataset('r', data=self.r)
            hf.create_dataset('iter', data=[self.iter])
            hf.create_dataset('outOfBounds', data=[self.outOfBounds])
        return self

    def get_pitch_angle(self):
        '''
        Returns the pitch angle, in degrees, between the velocity of the particle
        and the field vector at that location
        '''
        angle_radian = np.sum(self.get_v()*self.get_Bfield(),axis=1) / (np.linalg.norm(self.get_v()) * np.linalg.norm(self.get_Bfield()))
        angle_degree = 180 * angle_radian / np.pi
        return angle_degree

    def get_Bfield(self):
        '''
        gives us the B field at every position the particle inhabits during its motion
        '''
        return np.asarray(self.Bfield)

    def get_vperp(self):
        """
        Calculates magnitude of velocity component perpendicular to B
        at each point in the trajectory.

        """

        #define unit vector in direction of B
        Bmagnitude = np.sqrt((self.get_Bfield()**2).sum(axis=1))
        bhat = (self.get_Bfield().T/Bmagnitude).T #have to use transpose because shapes are set up backwards for broadcasting

        #get parallel component of velocity
        vparallel = self.get_v()*bhat

        #subtract vparallel from total v to get vperp
        vperp = self.get_v() - vparallel

        #return only magnitude of vperp
        return np.sqrt((vperp**2).sum(axis=1))

    def get_mu_along_traj(self, m=1):
        """
        Calculates particle magnetic moment at each point in trajectory.
        :param m: mass of particle (default is normalized to m_p = 1)
        """

        vperp = self.get_vperp()
        Bmagnitude = np.sqrt((self.get_Bfield**2).sum(axis=1))

        #return mu = m vperp**2 / 2B
        return m*vperp**2/(2*Bmagnitude)





    def plot_traj(self, title="Trajectory"):
        # plots the path of the particle
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #set_up_ax(title, ax)
        ax.plot(self.r[:, 0], self.r[:, 1], self.r[:, 2])
        plt.show()
        return self

    def plot_poincare(self, lag, title=""):
        '''
        :param lag: desired lag to be attributed to the plot
        :param title: title of the plot
        makes a Poincare plot using the position data, specifically in the x direction
        '''
        x = self.r[0: 30000 - lag]
        x = x[:, 0]
        x_prime = self.r[lag: 30000]
        x_prime = x_prime[:, 0]
        plt.plot(x, x_prime)
        plt.title(title)
        plt.show()

    # Thanks for this function, Nick!
    def plot_puncture(self, dimension, value, title="", precision=0.005, min1=-10, max1=10, min2=-10, max2=10):
        '''
        Thanks, Nick!
        Takes in an array and creates a trajectory puncture plot
        Arguments:
            dimension: int (0:x, 1:y, 2:z) along which to slice to make plot
            value: number at which to slice for plot
            title: title of your plot!
            precision: number indicating how far from value to find points
            min1, max1, min2, max2: number; extents of puncture plot on the two
            non-"dimension" axes
        Return: void (makes plot)
        '''
        trajectoryArray = np.asarray([self.r[:, 0], self.r[:, 1], self.r[:, 2]])

        dict = {0: 'x', 1: 'y', 2: 'z'}

        dim1Num = (dimension + 1) % 3
        dim2Num = (dimension + 2) % 3

        if (trajectoryArray.shape[0] == 3):
            dim = trajectoryArray[dimension]
            dim1 = trajectoryArray[dim1Num]
            dim2 = trajectoryArray[dim2Num]
        else:
            st = str(trajectoryArray.shape)
            raise ValueError('array must have shape 3xN. array has shape %s' % st)

        dim1Array = np.array([])
        dim2Array = np.array([])
        for i in range(len(dim)):
            if np.abs(dim[i] - value) < precision:
                new = np.array([dim1[i]])
                dim1Array = np.append(dim1Array, new)
                new = np.array([dim2[i]])
                dim2Array = np.append(dim2Array, new)

        plt.plot(dim1Array, dim2Array, 'bo')
        plt.ylabel(dict[dim2Num])
        plt.xlabel(dict[dim1Num])
        plt.ylim((min2, max2))
        plt.xlim((min1, max1))
        plt.title(title + ": %s vs. %s" % (dict[dim1Num], dict[dim2Num]))
        plt.axis('equal')
        plt.show()
    #####changed

    def setIter(self,iter):
        self.iter = iter