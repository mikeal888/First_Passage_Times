import numpy as np
from qutip import *
from tqdm import tqdm

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
        self.dim = H.shape[0]**2    # dimension of the Liouvillian space

    def measurement_superoperator(self):
        """
        Return the Kraus operators for the qubit jump operator.
        """
        M0 = to_super(1 - 1j * self.H*self.dt).full()
        Mi = [to_super(np.sqrt(self.dt)*c_op).full() for c_op in self.c_ops]
        return [M0] + Mi
    
    def evolution_matrix(self, nu_k: list):

        """
        Implements an absorbing boundary condition on the n-resolved density matrix 

        Parameters
        ----------
        nu_k : list of ints
            List of what n state is coupled to

        M[0] reserved for no jump evolution
        M[i] reserved for jump evolution

        Returns
        -------
        M_update : np.array
            The evolution matrix
        """

        M = self.measurement_superoperator()

        # check that the length of nu_k is the same as the number of collapse operators
        assert len(nu_k)==len(M), f"length of nu_k =! len(c_ops)"
        assert nu_k[0] == 0, f"nu_k[0] must be 0"

        # Compute M_update superoperats
        M_update_ops = [np.kron(np.diag(np.ones(self.N_len - np.abs(nu_k[i])),  k=nu_k[i]), M[i]) for i in range(len(nu_k))]
        M_update = sum(M_update_ops)
        

        return M_update
    
    def solve(self, rho0, nu_k, ix=0):
        """
        Solve the projective evolution of the density matrix

        Parameters
        ----------
        rho0 : Qobj
            The initial density matrix
        ix : int        
            The index of the initial state in the n-resolved basis

        Returns
        -------
        Pnt : np.array
            Solution to the n-resolved density matrix
        """

        # Compute the evolution matrix
        M_update = self.evolution_matrix(nu_k)

        # Convert the initial state to a vector 
        if rho0.type == 'oper':
            print("Converting initial state to vector form")
            rho0 = operator_to_vector(rho0).full()
        elif rho0.type == 'ket':
            print("Converting initial state to vector form")
            rho0 = operator_to_vector(ket2dm(rho0)).full()
        else:
            print("Initial state is already in vector form")
            rho0 = rho0.full()

        # Initialise the density matrix vector
        rho_n_vec = np.zeros((self.dim*self.N_len, len(self.t)), dtype=complex)
        rho_n_vec[self.dim*ix:self.dim*(ix+1), 0] = rho0.flatten()

        # Get Ivec
        Ivec = np.eye(int(np.sqrt(self.dim))).reshape(self.dim,)

        # Initialise Pn
        Pn_vec = np.zeros((self.N_len, len(self.t)))
        Pn_vec[:, 0] = [np.real(np.dot(Ivec, rho_n_vec[self.dim*n:self.dim*(n+1), 0])) for n in range(self.N_len)]

        # evolve rho
        for i in tqdm(range(1, len(self.t)), desc="Evolution Superoperator"):
            rho_n_vec[:, i] = M_update @ rho_n_vec[:, i-1] 

            # calculate Pn
            for j in range(self.N_len):
                Pn_vec[j, i] = np.real(np.dot(Ivec, rho_n_vec[self.dim*j:self.dim*(j+1), i]))

        return Pn_vec
    
class ProjectiveEvolutionPntAbsorb(ProjectiveEvolutionPnt):
    """
    This class is used to compute the projective evolution of the N resolved density operator with absorbing boundary conditions
    We use vectorised density operators and the projective evolution of the jump operator
    """

    def __init__(self, H, c_ops, t, N, N_cutoff, kind='single'):
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
        N_cutoff : list of float
            The list of N values that are coupled to the absorbing boundary
        kind : str
            'single' or 'double' absorbing boundary condition with Nc for single and Nc and -Nc for double
        """
        super().__init__(H, c_ops, t, N)
        self.N_cutoff = N_cutoff
        self.kind = kind

    def evolution_matrix(self, nu_k: list):
        """
        Implements an absorbing boundary condition on the n-resolved density matrix 

        Parameters
        ----------
        nu_k : list of ints
            List of what n state is coupled to
        N_cutoff : list of ints
            List of the N values that are coupled to the absorbing boundary
        kind : str
            'single' or 'double' absorbing boundary condition with Nc for single and Nc and -Nc for double

        M[0] reserved for no jump evolution
        M[i] reserved for jump evolution

        Returns
        -------
        M_update : np.array
            The evolution matrix
        """

        M = self.measurement_superoperator()

        # check that the length of nu_k is the same as the number of collapse operators
        assert len(nu_k)==len(M), f"length of nu_k =! len(c_ops)"
        assert nu_k[0] == 0, f"nu_k[0] must be 0"
        
        # Replace N with N_cutoff
        if self.kind == 'single':
            N = self.N[self.N <= self.N_cutoff]
        elif self.kind == 'double':
            N = self.N[np.abs(self.N) <= self.N_cutoff]
       
        # update ProjectiveEvolutionPnt.N and ProjectiveEvolutionPnt.N_len
        self.N = N
        self.N_len = len(N)

        # Compute M_update superoperats
        M_update_ops = [np.kron(np.diag(np.ones(self.N_len - np.abs(nu_k[i])),  k=nu_k[i]), M[i]) for i in range(len(nu_k))]    
        M_update = sum(M_update_ops)

        # Add the absorbing boundary condition
        if self.kind == 'single':
            # Set first dim row to zero
            M_update[:self.dim, :] = 0
        elif self.kind ==' double':
            # Set first dim row to zero
            M_update[:self.dim, :] = 0
            # Set last dim row to zero
            M_update[-self.dim:, :] = 0
        
        return M_update




