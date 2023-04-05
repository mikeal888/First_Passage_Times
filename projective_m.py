import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from _funcs import *

"""
In this file we want to implement the projective evolution of jump

M0 = 1 - i H_{eff} dt where H_{eff} = H + dt gamma * (1-f)*sp*sm + dt gamma * f*sm*sp
M1 = sqrt(gamma * (1-f) dt ) * sm 
M2 = sqrt(gamma * f dt) * sp
"""    


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

def evolve_matrix_new(N, H, c_ops, dt, nu_k):
    """
    Evolve the matrix through the measurement superoperators
    """
    M = qubit_measurement_superoperators(H, c_ops, dt)

    # M_update = np.kron(np.diag(np.ones(N)), M[0]) + np.kron(np.diag(np.ones(N-1), k=1), M[1]) + np.kron(np.diag(np.ones(N-1), k=-1), M[2])

    M_update_ops = [np.kron(np.diag(np.ones(N - np.abs(nu_k[i])),  k=nu_k[i]), M[i]) for i in range(len(nu_k))]
    M_update = sum(M_update_ops)

    return M_update

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

    # Initial index
    i0 = int(Nm/dN)

    # Compute initial state
    rho0 = operator_to_vector(steadystate(H, c_ops))
    nu_k = [0, 1, -1]

    #----------------- Solve -----------------#

    proj = ProjectiveEvolutionPnt(H_eff, c_ops, t, N)
    Pn_vec = proj.solve(rho0, nu_k, i0)

    # Repeat analysis for superoperator method
    # M_update = evolve_matrix(len(N), H_eff, c_ops, dt)

    # # # define initial state
    # rho0 = operator_to_vector(steadystate(H, c_ops)).full()
    # dim = rho0.shape[0]

    # # # initialise rho array
    # i0 = int(Nm/dN)
    # rho_n_vec = np.zeros((4*len(N), len(t)), dtype=complex)
    # rho_n_vec[dim*i0:dim*(i0+1), 0] = rho0.flatten()
    
    # # # create Ivec
    # Ivec = np.array([1, 0, 0, 1])

    # # Initialise Pn
    # Pn_vec = np.zeros((len(N), len(t)))
    # Pn_vec[:, 0] = [np.real(np.dot(Ivec, rho_n_vec[dim*n:dim*(n+1), 0])) for n in range(len(N))]

    # # evolve rho
    # for i in tqdm(range(1, len(t)), desc="Evolution Superoperator"):
    #     rho_n_vec[:, i] = M_update @ rho_n_vec[:, i-1] 

    #     # calculate Pn
    #     for j in range(len(N)):
    #         Pn_vec[j, i] = np.real(np.dot(Ivec, rho_n_vec[dim*j:dim*(j+1), i]))


    #----------------- Plotting -----------------#
    fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))

    plot_times = [5, 15, 30, 50]

    # create a subplot of Pnt vs time
    ax1.plot(t, Pn_vec.T, color='r')
    ax1.plot(t, np.sum(Pn_vec, axis=0), color='black', linestyle='--')

    # create a subplot of Pnt vs n
    for t_plot in plot_times:
        ax2.bar(N, Pn_vec[:, np.argmin(np.abs(t-t_plot))], alpha=0.5, width=1, edgecolor='k')

    ax2.set_xlabel('n')
    ax2.set_ylabel('Pn(t)')

    ax2.grid(linestyle='--', linewidth=0.5, alpha=0.5)

    ax2.set_xlim([-35, 5])
    ax2.set_ylim(0, 0.3)
    fig2.show()


    # #----------------- Old Code  -----------------#


    # # proj = ProjectiveEvolutionPnt(H, c_ops, t, N)
    # # M = proj.evolution_matrix(nu_k)
    # # Pn_vec = proj.solve(rho0, nu_k, i0)

    # #  # create measurement operators 
    # # M = qubit_measurement_operators(H_eff, c_ops, dt)

    # # # Initialise Pn
    # # Pn = np.zeros((len(N), len(t)))
    # # Pn[:, 0] = [np.real(np.trace(rho_n[n, :, :, 0])) for n in range(len(N))]

    # # #evolve rho
    # # for i in tqdm(range(1, len(t)), desc="Evolution Direct"):
    # #     for n in range(len(N)):
    # #         if n == 0:
    # #             rho_n[n, :, :, i] = M[0] @ rho_n[n, :, :, i-1] @ M[0].conj().T + M[1] @ rho_n[n+1, :, :, i-1] @ M[1].conj().T
    # #         elif n == len(N)-1:
    # #             rho_n[n, :, :, i] = M[0] @ rho_n[n, :, :, i-1] @ M[0].conj().T + M[2] @ rho_n[n-1, :, :, i-1] @ M[2].conj().T
    # #         else:
    # #             rho_n[n, :, :, i] = M[0] @ rho_n[n, :, :, i-1] @ M[0].conj().T + M[1] @ rho_n[n+1, :, :, i-1] @ M[1].conj().T + M[2] @ rho_n[n-1, :, :, i-1] @ M[2].conj().T

    # #         # calculate Pn
    # #         Pn[n, i] = np.real(np.trace(rho_n[n, :, :, i]))

    # # #----------------- Plotting -----------------#

    # # # # create figure
    # # # fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))

    # # # # create a subplot of Pnt vs time
    # # # ax1.plot(t, Pn.T, color='r')
    # # # ax1.plot(t, np.sum(Pn, axis=0), color='black', linestyle='--')

    # # # # # create a subplot of Pnt vs n
    # # # ax2.plot(N, Pn[:, ::1000], color='r')
    # # # fig1.show()
