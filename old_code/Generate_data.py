import numpy as np
from qutip import *
from time import time
from scipy.integrate import trapz
import pickle as pkl
import time as time

def time_wrapper(func):
    def wrapper(*args, **kwargs):
        tic = time.time()
        func(*args, **kwargs)
        toc = time.time()
        print('Time to run {}: {}'.format(func.__name__, toc - tic))

# We need a quicker way to compute the tilted liouvillian 


def tilted_liouvillian(H, L, chi, v):
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
    
    return L_tilt

# We can pass this tilted liouvillian to the mesolve function
@time_wrapper
def solve_tilted_liouvillian(H, L, chi_list, v, psi0, t):
    '''

    Solve the Titlted Liouvillian for a single operator using mesolve

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
    psi0 : qobj
        The initial state of the system
    t : list
        The time list   
    
    Returns
    -------
    pchis : list
        The list of pchi values for each chi value as a function of time 
    '''
    
    # Initialize the list of pchi values
    pchis = np.zeros((len(chi_list), len(t)))

    # Loop over the chi values
    for i, chi in enumerate(chi_list):
        # Define the tilted liouvillian
        L_tilt = tilted_liouvillian(H, L, chi, v)
        output = mesolve(L_tilt, psi0, t, [], [])
        pchis[i, :] = [rho.tr() for rho in output.states]
    
    return pchis


if __name__ == '__main__':
        
    # Define the parameters
    k = 0.5
    Omega = 1;
    
    # Define the Hamiltonian
    H = Omega/2 * sigmay()
    
    # Define the jump operator
    L = np.sqrt(k) * sigmam()
    
    # Define the initial state
    psi0 = basis(2, 1)
    
    # Define the time list
    dt = 0.2
    t0, tf = 0.0, 1
    t = np.arange(t0, tf, dt)
    # t = [0.5]

    # Define chilist
    dchi = 0.05
    chilist = np.arange(-20, 20, dchi)

    # Solve the tilted liouvillian
    tic = time()
    pchis = solve_tilted_liouvillian(H, L, chilist, 1, psi0, t)
    toc = time()
    print('Time to solve the tilted liouvillian: ', toc - tic)

    # Now compute integral over n = 1
    dn = 0.1
    nvals = np.arange(-10, 10, dn)
    Pnt = np.real(np.array([trapz(np.exp(-1j*ni*chilist)*pchis.T, chilist, dx=dchi, axis=1)/(2*np.pi) for ni in nvals]))


    # Save data to pickle
    filedir = './Data/data_OM={}_k={}_N.pkl'.format(Omega, k)
    data = {'t': t, 
            'nvals': nvals, 
            'Pnt': Pnt, 
            'pchis': pchis, 
            'chilist': chilist}
    with open(filedir, 'wb') as f:
        print("Saving data to {}".format(filedir))
        pkl.dump(data, f)

