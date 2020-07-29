import matplotlib as mpl
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.core.multiarray import ndarray
from scipy.special import j0, j1, jn_zeros
import time
from scipy.interpolate import RegularGridInterpolator
import h5py


class interpolator(object):

    def __init__(self, bx, by, bz, x_shape=(-100, 100), y_shape=(-100, 100), z_shape=(0, 1000), b0=1.0,
                 mask_max_radius=False):
        self.x_min = x_shape[0]
        self.x_max = x_shape[1]
        self.y_min = y_shape[0]
        self.y_max = y_shape[1]
        self.z_min = z_shape[0]
        self.z_max = z_shape[1]
        self.grid_shape = bx.shape  # we expect this to be the same as by.shape and bz.shape
        # might no longer need this, as fields are normalized at the generation stage
        self.b0 = b0
        self.mask_max_radius = mask_max_radius

        x = np.linspace(self.x_min, self.x_max, self.grid_shape[0])
        y = np.linspace(self.y_min, self.y_max, self.grid_shape[1])
        z = np.linspace(self.z_min, self.z_max, self.grid_shape[2])

        self.bx_function = RegularGridInterpolator((x, y, z), bx, method='linear',bounds_error=False)
        self.by_function = RegularGridInterpolator((x, y, z), by, method='linear',bounds_error=False)
        self.bz_function = RegularGridInterpolator((x, y, z), bz, method='linear',bounds_error=False)

    def field(self, r, t):
        if self.mask_max_radius is True:
            # print('Trigger 1')
            max_radius = (self.x_max - self.x_min) / 2
            # print(r[0] ** 2 + r[1] ** 2 , max_radius)
            if (r[0] ** 2 + r[1] ** 2 > max_radius ** 2):
                # print('Trigger 2')
                return np.asarray([0, 0, 0])

        bx = self.bx_function(r)[0]  # because RGI returns results in an array
        by = self.by_function(r)[0]
        bz = self.bz_function(r)[0]
        result = self.b0 * np.asarray([bx, by, bz])
        return result


def getUniformField(r, t):
    return np.asarray([0, 1, 0])



def nullField(r, t):
    return np.asarray([0, 0, 0])


def getSpheromakField(r, t):
    '''
    r =  array corresponding to the position at which we want the field
    t =  not sure - not used explicitly in the definition - perhaps meant to be for time evolution?
    returns the array of field components at various points
    '''
        #befor, r=100, l=100
    def getSpheromakFieldAtPosition(x, y, z, center=(0, 0, 0), B0=1, R=1, L=1):
        '''
        The spheromak center in z must be L. Our spheromak has ratio L/R=1.
        '''

        # parameters
        j1_zero1 = jn_zeros(1, 1)[0]
        kr = j1_zero1 / R
        kz = np.pi / L

        lam = np.sqrt(kr ** 2 + kz ** 2)

        # construct cylindrical coordinates centered on center
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        theta = np.arctan2(y, x)
        centZ = z - center[2]

        # calculate cylindrical fields
        Br = -B0 * kz / kr * j1(kr * r) * np.cos(kz * centZ)
        Bt = B0 * lam / kr * j1(kr * r) * np.sin(kz * centZ)

        # convert back to cartesian, place on grid.
        Bx = Br * np.cos(theta) - Bt * np.sin(theta)
        By = Br * np.sin(theta) + Bt * np.cos(theta)
        Bz = B0 * j0(kr * r) * np.sin(kz * centZ)

        return Bx, By, Bz

    # using the previous function to determine the field components at every position observed
    Bx, By, Bz = getSpheromakFieldAtPosition(r[0], r[1], r[2])
    B_vector = np.asarray([Bx, By, Bz])

    return B_vector


def getDipoleField(r, t):
    # magnetic moment, s.t M0 = field strength one proton radius from origin on the xy plane
    #M0 = -1000

    M0 = 1

    # for better readability
    x = r[0]
    y = r[1]
    z = r[2]

    # explicit calculation of the dipole field using position values
    B_vector = np.asarray([3 * M0 * x * z, 3 * M0 * y * z, M0 * (2 * z ** 2 - x ** 2 - y ** 2)])

    # normalizing the field?
    B_vector = B_vector / (np.dot(r, r) ** (5 / 2))

    return B_vector

def getDipolePotential(r, t):
    # magnetic moment, s.t M0 = field strength one proton radius from origin on the xy plane
    M0 = 1

    # for better readability
    x = r[0]
    y = r[1]
    z = r[2]

    # explicit calculation of the dipole field using position values
    A_vector = np.asarray([-M0*y/((x**2 + y**2 + z**2) ** (3 / 2)), -M0*x/((x**2 + y**2 + z**2) ** (3 / 2)), 0])

    # normalizing the field?
    #A_vector = A_vector / ((x**2 + y**2 + z**2) ** (3 / 2))

    return A_vector

def getDipoleFlux(r,t):
    M0 = 1

    # for better readability
    x = r[0]
    y = r[1]
    z = r[2]

    # explicit calculation of the dipole field using position values
    flux = np.asarray([M0*np.sin(np.arccos(z/np.sqrt(x**2+y**2+z**2)))/(x**2+y**2+z**2)])
    return flux

def getSpheromakFlux(r,t,R=1,L=1,b0=1,center=(0, 0, 0)):
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    j1_zero1 = jn_zeros(1, 1)[0]
    kr = j1_zero1 / R
    kz = np.pi / L
    centZ = z - center[2]
    return b0*r_mag*j1(kr*r_mag)*np.sin(kz*centZ)

def getSpheromakPotential(r,t,R=1,L=1,b0=1,center=(0, 0, 0)):
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    j1_zero1 = jn_zeros(1, 1)[0]
    kr = j1_zero1 / R
    kz = np.pi / L
    centZ = z - center[2]
    return b0 * j1(kr * r_mag) * np.sin(kz * centZ)

def getHarrisCurrent(r):
    z = r[2]
    L = 0.5
    B0 = 1
    By0 = 0
    # current density
    jx = By0 * np.sinh(z / L) / (B0 * np.cosh(z / L) ** 2 * L)
    jy = 1 / (np.cosh(z / L) ** 2 * L)
    jz = 0

    # the E field arising from the current
    J_vector = np.asarray([jx, jy, jz])
    return J_vector

def getHarrisElectricFieldStrength(x,z):
    mu=1
    L=1
    E_mag = mu * 1 / (np.cosh(z / L) ** 2 * L)
    return E_mag
def getHarrisElectricField(r,t): #don't need this for now
    '''operating in cartesian coordinates
           r: position
           L: half-thickness (don't know what this is)
           B0:
           By0: zero in a harris sheet, but just in case
           t: time evolution?
           c: conductivity
           '''
    # use cartesian coordinates

    z = r[2]
    L = 1
    B0 = 1
    By0 = 0
    mu = 1
    # current density
    Ex = mu*By0 * np.sinh(z / L) / (B0 * np.cosh(z / L) ** 2 * L)
    Ey = mu * 1 / (np.cosh(z / L) ** 2 * L)
    Ez = 0

    # the E field arising from the current
    E_vector = np.asarray([Ex, Ey, Ez])
    return E_vector

def getHarrisFieldStrength(x,z):
    L =1
    B0=1
    return B0*np.tanh(z / L)
def getHarrisField(r, t):
    '''operating in cartesian coordinates
       r: position
       L: half-thickness (don't know what this is)
       B0: 1 for scaling
       By0: zero in a harris sheet, but just in case
       t: time evolution?'''
    L = 1
    B0 = 1
    By0 = 0.1
    Bz0 = 0
    # use cartesian coordinates
    x = r[0]
    y = r[1]
    z = r[2]

    # three components
    Bx = np.tanh(z / L)
    By = By0 / (B0 * np.cosh(z / L))
    Bz = Bz0 / (B0 * np.cosh(z / L))

    # the B field
    B_vector = B0 * np.asarray([Bx, By, Bz])

    return B_vector

def anotherHarris(r,t,xoff,yoff,zoff):
    '''operating in cartesian coordinates
           r: position
           L: half-thickness (don't know what this is)
           B0: 1 for scaling
           By0: zero in a harris sheet, but just in case
           t: time evolution?
           ****this function requires r to be a list (or not) and be x,y,z. You can mesh up the
           coordinate using
           r = list(np.ndindex(2,2,2))'''
    L = 1
    B0 = 1
    By0 = 0
    # use cartesian coordinates

    Bx = np.array([])
    By = np.array([])
    Bz = np.array([])
    # three components
    for i in range(0,len(r)):
        x = r[i][0]-xoff
        y = r[i][1]-yoff
        z = r[i][2]-zoff
        Bxx = np.tanh(z / L)
        Byy = By0 / (B0 * np.cosh(z / L))
        Bzz = 0
        Bx = np.append(Bx, Bxx)
        By = np.append(By, Byy)
        Bz = np.append(Bz, Bzz)

    # the B field
    return Bx,By,Bz

def E(r, t):
    return np.asarray([0, 0, 0])


def getWireField(r, t, I=1):
    '''def getWireField(r, t, I=1):
    infinite wire at z-axis
    r =  array corresponding to the position at which we want the field
    t =  not sure - not used explicitly in the definition - perhaps meant to be for time evolution?
    I = current in the wire
    returns the array of field components at various points
    '''

    # for better readability
    x = r[0]
    y = r[1]
    z = r[2]

    # explicit calculation of the wire field using position values
    B_vector = np.asarray([-y, x, 0])/(x**2+y**2)

    # multiplying the field by magnutide scalar
    mu_0 = 4*np.pi * 10**(-7)
    #B_vector = B_vector * mu_0 * I / (2 * np.pi * (x**2+y**2))

    return B_vector


def getHarrisPotential(r,t):
    '''
        Harris Field as illustrated in Monday June 8
        Calculate potential to find out momentum
        r =  array corresponding to the position at which we want the field
        returns the float of vector potential in the z-direction at various points
        '''
    #parameters
    B0 = 1
    L = 1
    z = r[2]
    return -B0*L*np.log(np.cosh(z/L))

def getWirePotentialZ(r, t, I=1):
    '''
    infinite wire at z-axis
    r =  array corresponding to the position at which we want the field
    t =  not sure - not used explicitly in the definition - perhaps meant to be for time evolution?
    I = current in the wire
    returns the float of vector potential in the z-direction at various points
    '''

    # for better readability
    x = r[0]
    y = r[1]
    z = r[2]
    r_mag = (x**2 + y**2)**(1/2)

    # explicit calculation of the wire vector potential using position values
    Az = - np.log(r_mag)

    # multiplying the field by magnutide scalar
    mu_0 = 4*np.pi * 10**(-7)
    #Az = Az * mu_0 * I / (2 * np.pi)

    return Az

def create_spheromak_interpolator(grid_filename, x_shape=(-100, 100), y_shape=(-100, 100), z_shape=(0, 100)):
    hf = h5py.File(grid_filename, 'r')
    bx_hf, by_hf, bz_hf = [hf.get(x) for x in ['bx', 'by', 'bz']]

    bx, by, bz = [np.zeros((x.shape)) for x in [bx_hf, by_hf, bz_hf]]

    for i in range(bx_hf.shape[0]):
        bx[i], by[i], bz[i] = [x[i] for x in [bx_hf, by_hf, bz_hf]]  # defining bx/by/bz based off of file values

    # The data masking gives us slightly inaccurate results near the edges
    '''
    for i in range(bx.shape[0]):
        for j in range(bx.shape[1]):
            for k in range(bx.shape[2]):
                # converting indices to coordinates

                x = x_shape[0] + i * (x_shape[1] - x_shape[0])/(bx.shape[0] - 1)
                y = y_shape[0] + j * (y_shape[1] - y_shape[0])/(by.shape[0] - 1)
                # z = z_shape[0] + k * (z_shape[1] - z_shape[0])/(bz.shape[0] - 1)

                # assuming that x_shape = y_shape
                max_radius = (x_shape[1] - x_shape[0]) / 2

                # masking
                if x ** 2 + y ** 2 > max_radius ** 2:
                    bx[i][j][k] = 0
                    by[i][j][k] = 0
                    bz[i][j][k] = 0
    '''

    hf.close()
    spheromak_interpolator = interpolator(bx, by, bz, x_shape=(-100, 100), y_shape=(-100, 100), z_shape=(0, 100),
                                          mask_max_radius=True)
    return spheromak_interpolator


def create_taylor_interpolator(filename):
    sizes = (72, 72, 200)  # setting parameters for sizes (x, y, and z)
    data = np.loadtxt(filename, skiprows=3)  # loading the data from the file specified in calling the function

    '''
    x = data[:,0].reshape(sizes)
    y = data[:,1].reshape(sizes)
    z = data[:,2].reshape(sizes)
    '''
    # inputting the data in the respective positions (x data in first, y data in second, etc.)
    cached_bx = data[:, 3].reshape(sizes)  # same for magnetic field components
    cached_by = data[:, 4].reshape(sizes)
    cached_bz = data[:, 5].reshape(sizes)

    # setting values for the field components equal to 0 if they don't exceed a certain value
    for i in range(cached_bx.shape[0]):
        for j in range(cached_bx.shape[1]):
            for k in range(cached_bx.shape[2]):
                if (cached_bx[i][j][k] <= -1 * 10 ** 98):
                    cached_bx[i][j][k] = 0
                if (cached_by[i][j][k] <= -1 * 10 ** 98):
                    cached_by[i][j][k] = 0
                if (cached_bz[i][j][k] <= -1 * 10 ** 98):
                    cached_bz[i][j][k] = 0

    # final interpolation using the values from the file after adjustment, which is then returned
    taylor = interpolator(cached_bx, cached_by, cached_bz, x_shape=(-100, 100), y_shape=(-100, 100), z_shape=(0, 1000),
                          b0=1 / 0.38239010712927873)

    return taylor

def getSpeiserBField1(r,t=0):
    '''field from Speiser paper'''

    b=1.0
    d=1.0

    # for better readability
    x = r[0]

    # explicit calculation of the dipole field using position values
    B_vector = np.asarray([0, -b*(x/d), 0])


    return B_vector

def getSpeiserEfield1(r,t=0):
    return np.asarray([0, 0, -1])


def getSpeiserBField2(r, t=0):
    '''field from Speiser paper'''

    b = 1
    d = 1
    ita = 1

    # for better readability
    x = r[0]

    # explicit calculation of the dipole field using position values
    B_vector = np.asarray([1, -b * (x / d), 0])
    return B_vector

def getDoubleHarrisField(r,t,mu=0.01,By0=1,L=0.1):
    z = r[2]
     #change this to make skinnier sheet
    B0 = 1
    # current density
    Bx = B0 * (np.tanh((z-3*L)/L)-np.tanh((z+3*L)/L))
    By = By0 * (1/np.cosh((z-3*L)/L)-1/np.cosh((z+3*L)/L))
    Bz = 0

    B_vector = B0 * np.asarray([Bx, By, Bz])
    return B_vector

def getDoubleHarrisELectricField(r,t,mu=0.01,By0=1,L=0.1):
    z = r[2]
    B0 = 1
    # current density
    Ex = mu * By0 * np.sinh((z-3) / L) / (B0 * np.cosh((z-3) / L) ** 2 * L) - mu * By0 * np.sinh((z+3) / L) / (B0 * np.cosh((z+3) / L) ** 2 * L)
    Ey = mu * B0/(L*np.cosh((z-3)/L)**2) - B0/(L*np.cosh((z+3)/L)**2)
    Ez = 0

    E_vector = np.asarray([Ex,Ey,Ez])
    return E_vector


def get2DSpheromakFlux(r,z,R=1,L=1,b0=1):
    j1_zero1 = jn_zeros(1, 1)[0]
    kr = j1_zero1 / R
    kz = np.pi / L
    return b0*r*j1(kr*r)*np.sin(kz*z)