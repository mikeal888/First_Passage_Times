from qutip import *
import matplotlib.pyplot as plt

# System dimension
N = 30

# Define useful system operators
a = destroy(N)
x = a.dag() + a
n = num(N)

# Define system parameters
Ham = w * a.dag()