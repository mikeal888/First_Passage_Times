import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
In this file we want to implement the projective evolution of jump

M0 = 1 - i H_{eff} dt where H_{eff} = H + dt gamma * (1-f)*sp*sm + dt gamma * f*sm*sp
M1 = sqrt(gamma * (1-f) dt ) * sm 
M2 = sqrt(gamma * f dt) * sp
"""

class ProjectiveEvolutionPnt:
    """
    This class is used to compute the projective evolution of the N resolved density operator
    We use vectorised density operators and the projective evolution of the jump operator
    """

    def __init__(self, H, c_ops, t, N):
        """
        Parameters
        ----------
        H : qutip.Qobj
            The system Hamiltonian
        c_ops : list of qutip.Qobj
            The list of collapse operators
        t : list of float
            The list of times
        N : list of float
            The list of N values
        """
        self.H = H
        self.c_ops = c_ops
        self.t = t
        self.N = N
        self.N_len = len(N)
        self.dt = t[1] - t[0]   # assuming uniform grid
        self.dN = N[1] - N[0]   # assuming unifrom grid
        self.super_operators = self.measurement_superoperator()

    def measurement_superoperator(self):
        """
        Return the Kraus operators for the qubit jump operator.
        """
        M0 = to_super(1 - 1j * self.H*self.dt).full()
        Mi = [to_super(np.sqrt(self.dt)*c_op).full() for c_op in self.c_ops]
        return [M0] + Mi
    
    def evolution_matrix(self):
        """
        Compute the evolution matrix 
        """
        M = self.super_operators()
        M_update = np.kron(np.diag(np.ones(self.N_len)), M[0]) + np.kron(np.diag(np.ones(self.N_len-1), k=1), M[1]) + np.kron(np.diag(np.ones(self.N_len-1), k=-1), M[2])
        return M_update
    
    


def qubit_measurement_superoperators(H, c_ops, dt):
    """
    Return the Kraus operators for the qubit jump operator.
    """
    M0 = to_super(1 - 1j * H*dt).full()
    Mi = [to_super(np.sqrt(dt)*c_op).full() for c_op in c_ops]
    return [M0] + Mi

def qubit_measurement_operators(H, c_ops, dt):
    """
    Return the Kraus operators for the qubit jump operator.
    """
    M0 = 1 - 1j * H*dt
    Mi = [(np.sqrt(dt)*c_op).full() for c_op in c_ops]
    return [M0.full()] + Mi

def evolve_matrix(N, H, c_ops, dt):
    """
    Evolve the matrix through the measurement superoperators
    """
    M = qubit_measurement_superoperators(H, c_ops, dt)

    M_update = np.kron(np.diag(np.ones(N)), M[0]) + np.kron(np.diag(np.ones(N-1), k=1), M[1]) + np.kron(np.diag(np.ones(N-1), k=-1), M[2])

    return M_update


if __name__ == "__main__":
    # Parameters
    gamma = 1
    nbar = 0.2
    Omega = 1
    dt = 0.001
    t = np.arange(0, 50, dt)

    # Define system operators
    sp = sigmap()
    sm = sigmam()
    sx = sigmax()
    sz = sigmaz()
    H = Omega*sx

    # define dissipator and Hamiltonian
    c_ops = [np.sqrt(gamma * (1+nbar)) * sm, np.sqrt(gamma * nbar) * sp]    
    H_eff = H - 0.5j * sum([c_op.dag() * c_op for c_op in c_ops])

    # number of chargers to truncate at 
    Nm = 50
    dN = 1
    N = np.arange(-Nm, Nm, dN)

    #----------------- Superoperator -----------------#
    # Repeat analysis for superoperator method

    # M = qubit_measurement_superoperators(H_eff, c_ops, dt)
    M_update = evolve_matrix(len(N), H_eff, c_ops, dt)

    # # define initial state
    rho0 = operator_to_vector(steadystate(H, c_ops)).full()

    # # initialise rho array
    i0 = int(Nm/dN)
    rho_n_vec = np.zeros((4*len(N), len(t)), dtype=complex)
    rho_n_vec[4*i0:4*i0+4, 0] = rho0.flatten()
    
    # # create Ivec
    Ivec = np.array([1, 0, 0, 1])

    # Initialise Pn
    Pn_vec = np.zeros((len(N), len(t)))
    Pn_vec[:, 0] = [np.real(np.dot(Ivec, rho_n_vec[4*n:4*(n+1), 0])) for n in range(len(N))]

    # evolve rho
    for i in tqdm(range(1, len(t)), desc="Evolution Superoperator"):
        rho_n_vec[:, i] = M_update @ rho_n_vec[:, i-1] 

        # calculate Pn
        for j in range(len(N)):
            Pn_vec[j, i] = np.real(np.dot(Ivec, rho_n_vec[4*j:4*(j+1), i]))


    #----------------- Plotting -----------------#
    fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))

    # create a subplot of Pnt vs time
    ax1.plot(t, Pn_vec.T, color='r')
    ax1.plot(t, np.sum(Pn_vec, axis=0), color='black', linestyle='--')

    # create a subplot of Pnt vs n
    ax2.bar(N, Pn_vec[:, np.argmin(np.abs(t-5))], color='k', alpha=0.5, width=1, edgecolor='k')
    ax2.bar(N, Pn_vec[:, np.argmin(np.abs(t-15))], color='r', alpha=0.5, width=1, edgecolor='k')
    ax2.bar(N, Pn_vec[:, np.argmin(np.abs(t-30))], color='g', alpha=0.5, width=1, edgecolor='k')
    ax2.bar(N, Pn_vec[:, np.argmin(np.abs(t-50))], color='b', alpha=0.5, width=1, edgecolor='k')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Pn(t)')

    ax2.grid(linestyle='--', linewidth=0.5, alpha=0.5)

    ax2.set_xlim([-35, 5])
    ax2.set_ylim(0, 0.3)
    fig2.show()


    #----------------- Old Code  -----------------#

    #  # create measurement operators 
    # M = qubit_measurement_operators(H_eff, c_ops, dt)

    # # Initialise Pn
    # Pn = np.zeros((len(N), len(t)))
    # Pn[:, 0] = [np.real(np.trace(rho_n[n, :, :, 0])) for n in range(len(N))]

    # #evolve rho
    # for i in tqdm(range(1, len(t)), desc="Evolution Direct"):
    #     for n in range(len(N)):
    #         if n == 0:
    #             rho_n[n, :, :, i] = M[0] @ rho_n[n, :, :, i-1] @ M[0].conj().T + M[1] @ rho_n[n+1, :, :, i-1] @ M[1].conj().T
    #         elif n == len(N)-1:
    #             rho_n[n, :, :, i] = M[0] @ rho_n[n, :, :, i-1] @ M[0].conj().T + M[2] @ rho_n[n-1, :, :, i-1] @ M[2].conj().T
    #         else:
    #             rho_n[n, :, :, i] = M[0] @ rho_n[n, :, :, i-1] @ M[0].conj().T + M[1] @ rho_n[n+1, :, :, i-1] @ M[1].conj().T + M[2] @ rho_n[n-1, :, :, i-1] @ M[2].conj().T

    #         # calculate Pn
    #         Pn[n, i] = np.real(np.trace(rho_n[n, :, :, i]))

    # #----------------- Plotting -----------------#

    # # # create figure
    # # fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))

    # # # create a subplot of Pnt vs time
    # # ax1.plot(t, Pn.T, color='r')
    # # ax1.plot(t, np.sum(Pn, axis=0), color='black', linestyle='--')

    # # # # create a subplot of Pnt vs n
    # # ax2.plot(N, Pn[:, ::1000], color='r')
    # # fig1.show()
