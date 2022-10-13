"""
Generalized Lorentz equation integrator.
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

m_e = 9.11E-31 # kg
q_e = -1.6E-19 # C
R_e = 6371E3 # m

def dqdt(X, t, q, m, B):
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
    qm = np.sqrt(np.abs(q/m))
    ode_matrix = qm*np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, B[2], -B[1]],
            [0, 0, 0, -B[2], 0, B[0]],
            [0, 0, 0, B[1], -B[0], 0]
        ])
    return np.matmul(ode_matrix, X)

q0 = (0, 0, 0, 1E7, 0, 1E7)  # pos, vel

L = 5
B = [0, 0, 3.12E-5*(1/L)**3]  # T

args = (q_e, m_e, B)  # q, m , B
w_c = np.sqrt(np.abs(q_e*np.linalg.norm(B)/m_e))
print(w_c)
n_gyrations = 10
t = np.linspace(0, n_gyrations*2*np.pi/w_c, num=5000)
solution = odeint(dqdt, q0, t, args=args)

ax = plt.subplot(111, projection='3d')
ax.plot3D(solution[:, 0], solution[:, 1], solution[:, 2])
# plt.plot(solution[:, 0], solution[:, 1])
plt.show()