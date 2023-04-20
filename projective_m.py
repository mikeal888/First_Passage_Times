import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from _funcs import *



# Parameters
gamma = 1
nbar = 0.2
Omega = 1
dt = 0.0005
t = np.arange(0, 100, dt)

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
N_cutoff = 10
N = np.arange(-10, Nm, dN)

# Initial index
i0 = np.argmin(np.abs(N))

# Compute initial state
rho0 = operator_to_vector(steadystate(H, c_ops))
nu_k = [0, 1, -1]

#----------------- Solve -----------------#

proj = ProjectiveEvolutionPntAbsorb(H_eff, c_ops, t, N, N_cutoff)
Pn_vec = proj.solve(rho0, nu_k, i0)

# Compute survival probability
P0 = np.sum(Pn_vec, axis=0)
dGt = -np.gradient(P0, dt)

#----------------- Plotting -----------------#
fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))

plot_times = [5, 15, 30, 50]

# create a subplot of Pnt vs time
# ax1.plot(t, P0, color='black', linestyle='-', label=r'$f(t)$')
ax1.plot(t, dGt, color='black', linestyle='-')
ax1.set_xlabel('t')
ax1.set_ylabel(r'$\frac{dG(t)}{dt}$')
ax1.grid(linestyle='--', linewidth=0.5, alpha=0.5)

for t_plot in plot_times:
    ax2.bar(proj.N, Pn_vec[:, np.argmin(np.abs(t-t_plot))], alpha=0.5, width=1, edgecolor='k')

ax2.axvline(x=-N_cutoff, color='k', linestyle='--', alpha=0.5)

ax2.set_xlabel('n')
ax2.set_ylabel('Pn(t)')

ax2.grid(linestyle='--', linewidth=0.5, alpha=0.5)

ax2.set_xlim([np.min(-15), np.max(10)])
ax2.set_ylim(0, 0.3)
fig2.show()

