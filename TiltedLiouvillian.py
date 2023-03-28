import numpy as np
from qutip import *
from scipy.integrate import trapz
from scipy.linalg import expm
from time import time
from tqdm import tqdm

def tilted_liouvillian(H: Qobj, L: Qobj, chi: float, v: float) -> np.ndarray:
    '''
    Compute the tilted liouvillian for a single jump operator

    Parameters
    ----------
    H : qobj
        The Hamiltonian of the system
    L : qobj
        The jump operator
    chi : float 
        The value of the counting field
    v : float
        The value of the counting field

    Returns
    -------
    L_tilt : qobj
        The tilted liouvillian
    '''
    
    # Only works for one jump operator
    L_vec = liouvillian(H, [L])
    L_tilt = L_vec + 1j*chi * ( spre(L) + spost(L.dag()) ) - (chi**2 / (2))
    
    return L_tilt.full()

def solve_tilted_liouvillian(H, L, chi_list, rho0, t, dt):
    """
    Solve the Tilted Liouville Equation for a single operator 

    Parameters
    ----------
    H : qobj
        The Hamiltonian of the system
    L : qobj
        The jump operator
    chi_list : list
        The list of chi values for the counting field
    v : float
        The value of the counting field
    rho0 : qobj
        The initial state of the system
    t : list
        The time list
    dt : float
        The time step
    
    Returns
    -------
    pchis : np.ndarray
        The list of pchi values for each chi value as a function of time 
    """
    shp = rho0.shape
    # Vectorize the initial state
    rho_ss_vec = operator_to_vector(rho0).full()

    # Initialize the list of pchi values
    pchis = np.zeros((len(chi_list), len(t)), dtype=complex)

    # Loop over the chi values
    liouvs = [tilted_liouvillian(H, L, i, 1) for i in tqdm(chi_list, desc="Computing tilted liouvillians")]

    # Loop over the chi values and times 
    for i in tqdm(range(len(chi_list)), desc="Computing pchi values"):

        L_i = expm(liouvs[i]*dt)
        rho_temp = rho_ss_vec
        for j in range(len(t)):

            rho_new = L_i @ rho_temp
            pchis[i, j] = np.trace(rho_new.reshape(shp, order='F'))
            rho_temp = rho_new

    return pchis


if __name__ == "__main__":


    # Define the parameters
    # Define system parameters
    k = 0.5
    Ω = 1

    # useful operators
    sz = sigmaz()
    sx = sigmax()
    sy = sigmay()
    sm = sigmam()
    sp = sigmap()

    # ground state and excited states
    ground = fock_dm(2, 0)
    excited = fock_dm(2, 1)

    ground_vec = operator_to_vector(ground)
    excited_vec = operator_to_vector(excited)

    # Create time list for average and stochastic
    t0 = 0
    t1 = 25
    dt = 0.005
    t1s = 25
    dts = 0.005
    t = np.arange(t0, t1, dt)
    ts = np.arange(t0, t1s, dts)

    # Define dissipator
    L = np.sqrt(k)*sm
    c_ops = [L]

    # Define system Hamiltonian
    H = (Ω/2)*sx

    # Define steadystate and convert to vector
    rho_ss = steadystate(H, c_ops)
    rho_ss_vec = operator_to_vector(rho_ss)

    # Create chi space and compute tilted Liouvillians
    dchi = 0.005
    chi = np.arange(-30, 30, dchi)

    liouvs = [tilted_liouvillian(H, L, i, 1) for i in tqdm(chi, desc="Computing tilted liouvillians")]

    # # Creatte time space
    # dtimes = 0.01
    # times = np.arange(0.0, 30, dtimes)

    # # Compute tilted Liouvillians
    # pchis = solve_tilted_liouvillian(H, 1j*L, chi, rho_ss, times, dtimes)

    # # Now compute integral over n = 1
    # dn = 0.1
    # nvals = np.arange(-20, 20, dn)

    # tic = time()
    # Pnt = np.real(np.array([trapz(np.exp(-1j*ni*chi)*pchis.T, chi, dx=dchi, axis=1)/(2*np.pi) for ni in nvals]))*dn
    # toc = time()

    # print('Time to compute Pnt: {}'.format(toc - tic))
