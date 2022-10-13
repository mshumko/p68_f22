import numpy as np
import matplotlib.pyplot as plt

R = 6.371E6 # m

def dipole(X, x0=(0,0,0)):
    """
    Calculate a dipole magnetic field vector at point X from an Earth dipole 
    centered at x0.

    See: https://ccmc.gsfc.nasa.gov/static/files/Dipole.pdf for the dipole 
    field conversion to cartesian coordinates.

    Parameters
    ----------
    X: np.array
        Position in meters.
    
    Returns
    -------
    np.array
        The magnetic field vector in cartesian coordinates.
    """
    M = -8E15 # T/m^3
    r = np.linalg.norm(np.array(X)-np.array(x0))
    dx, dy, dz = (X[0]-x0[0]), (X[1]-x0[1]), (X[2]-x0[2])

    Bx = 3*M*dx*dz/r**5
    By = 3*M*dy*dz/r**5
    Bz = M*(3*dz**2-r**2)/r**5
    return [Bx, By, Bz]


if __name__ == '__main__':
    y = R*np.arange(-10.0, 10.0, .1) #create a grid of points from y = -10 to 10
    z = R*np.arange(-10.0, 10.0, .1) #create a grid of points from z = -10 to 10
    Y, Z = np.meshgrid(y,z) #turn this into a mesh
    ilen, jlen = np.shape(Y) #define the length of the dimensions, for use in iteration
    Bf = np.zeros((ilen,jlen,3)) #set the points to 0

    for i in range(0, ilen): #iterate through the grid, setting each point equal to the magnetic field value there
        for j in range(0, jlen):
            Bf[i,j] = dipole([0.0, Y[i,j], Z[i,j]]) 
            
    plt.streamplot(Y,Z, Bf[:,:,1], Bf[:,:,2])
    plt.show()