import numpy as np
from qutip import *
import matplotlib.pyplot as plt

"""
In this file we want to implement the projective evolution of jump

M0 = 1 - i H_{eff} dt where H_{eff} = H + dt gamma * (1-f)*sp*sm + dt gamma * f*sm*sp
M1 = sqrt(gamma * (1-f) dt ) * sm 
M2 = sqrt(gamma * f dt) * sp
"""

def qubit_measurement_superoperators(H, gamma, f, dt):
    """
    Return the Kraus operators for the qubit jump operator.
    """
    sp = sigmap()
    sm = sigmam()
    M0 = to_super(1 - 1j * H * dt)
    M1 = to_super(np.sqrt(gamma * (1-f) * dt) * sm)
    M2 = to_super(np.sqrt(gamma * f * dt) * sp)
    return [M0.full(), M1.full(), M2.full()]

def evolve_matrix(N):
    """
    Evolve the matrix through the measurement superoperators
    """
    M = qubit_measurement_superoperators(H, gamma, f, dt)

    M_update = np.kron(np.diag(np.ones(N)), M[0]) + np.kron(np.diag(np.ones(N-1), k=1), M[1]) + np.kron(np.diag(np.ones(N-1), k=-1), M[2])
    
    return M_update


if __name__ == "__main__":
    # Parameters
    gamma = 0.1
    f = 0.1
    dt = 0.01
    t = np.arange(0, 10, dt)
    H = sigmaz()
    c_ops = [np.sqrt(gamma * (1-f)) * sigmam(), np.sqrt(gamma * f) * sigmap()]    

    # number of chargers to truncate at 
    N = 10

    M = qubit_measurement_superoperators(H, gamma, f, dt)
    M_update = evolve_matrix(10)

    # define initial state
    rho0 = operator_to_vector(steadystate(H, c_ops)).full()

    # initialise rho array
    rho_n = np.zeros((len(t), 4*N, 1), dtype=complex)
    rho_n[0, :4, 0] = rho0.flatten()

    # evolve rho
    for i in range(1, len(t)):
        rho_n[i, :, :] = M_update @ rho_n[i-1, :, :]