"""
Generalized Lorentz equation integrator.
"""
import numpy as np
from scipy.integrate import ode, odeint
import matplotlib.pyplot as plt

m_e = 9.11E-31 # kg
q_e = -1.6E-19 # C
R_e = 6371E3 # m

def ode(X, t, q, m, B):
    """
    Calculate the derivatives of the position and velocity.

    q = [x, y, z, v_x, v_y, v_z].T vector
    """
    em = np.abs(q/m)
    ode_matrix = em*np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, B[2], -B[1]],
            [0, 0, 0, -B[2], 0, B[0]],
            [0, 0, 0, B[1], -B[0], 0]
        ])
    return np.matmul(ode_matrix, X)

X0 = (0, 0, 0, 1E7, 0, 1E7)  # pos, vel

L = 5
B = [0, 0, 3.12E-5*(1/L)**3]  # T

args = (q_e, m_e, B)  # q, m , B
w_c = np.sqrt(np.abs(q_e*np.linalg.norm(B)/m_e))
print(w_c)
t = np.linspace(0, 2*np.pi/w_c, num=5000)
solution = odeint(ode, X0, t, args=args)

# ax = plt.subplot(111, projection='3d')
# ax.plot3D(solution[:, 0], solution[:, 1], solution[:, 2])
plt.plot(solution[:, 0], solution[:, 1])
plt.show()