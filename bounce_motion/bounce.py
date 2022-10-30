import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

m_e = 9.11E-31 # kg
q_e = -1.6E-19 # C
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
    return np.array([Bx, By, Bz])

def dqdt(X, t, q, m):
    """
    Calculate the derivatives of the X vector containing the position and 
    velocity of the charged particle that we want to solve its equation of 
    motion for.

    Parameters
    ----------
    X: np.array
        A vector of positions and velocities. In the order 
        [x, y, z, v_x, v_y, v_z].T.
    t: float
        The time in units of seconds. A dummy variable unless E or B change 
        in time.
    q: float
        Particle's charge in units of Coulumbs.
    m: float
        Particle's mass in units of kilograms
    B: np.array
        The magnetic field vector.

    """
    qm = q/m
    B = dipole(X[0:3])
    ode_matrix = qm*np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, B[2], -B[1]],
            [0, 0, 0, -B[2], 0, B[0]],
            [0, 0, 0, B[1], -B[0], 0]
        ])
    return np.matmul(ode_matrix, X)


if __name__ == '__main__':
    q0 = (0, 4*R, 0, 1, 0, 1)  # pos, vel

    # # L = 5
    # # B = [0, 0, 3.12E-5*(1/L)**3]  # T

    args = (q_e, m_e)  # q, m , B
    t = np.linspace(0, 0.1, num=5000)
    solution = odeint(dqdt, q0, t, args=args)

    # # Remove coordinates if the particle escaped.
    # r = np.linalg.norm(solution, axis=1)
    # trapped_idx = np.where(r < 10*R)[0]
    # solution = solution[trapped_idx, :]
    # print(np.min(r)/R, np.max(r)/R)

    # # ax = plt.subplot(111, projection='3d')
    # # ax.plot3D(solution[:, 0], solution[:, 1], solution[:, 2])
    # plt.plot(solution[:, 0], solution[:, 2])
    fig, ax = plt.subplots(3, 1, sharex=True)
    for ax_i, solution_i in zip(ax, solution[:, :3].T):
        ax_i.plot(t, solution_i, 'k')
    plt.show()


    # y = R*np.linspace(-10.0, 10.0, 100) #create a grid of points from y = -10 to 10
    # z = R*np.linspace(-10.0, 10.0, 100) #create a grid of points from z = -10 to 10
    # Y, Z = np.meshgrid(y,z) #turn this into a mesh
    # ilen, jlen = np.shape(Y) #define the length of the dimensions, for use in iteration
    # Bf = np.zeros((ilen,jlen,3)) #set the points to 0

    # for i in range(0, ilen): #iterate through the grid, setting each point equal to the magnetic field value there
    #     for j in range(0, jlen):
    #         Bf[i,j] = dipole([0.0, Y[i,j], Z[i,j]], (0, 0, -10*R)) + dipole([0.0, Y[i,j], Z[i,j]], (0, 0, 110*R)) 
            
    # plt.streamplot(Y,Z, Bf[:,:,1], Bf[:,:,2], color='k')
    plt.show()