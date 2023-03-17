import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

# Load data from pickle
with open('./Data/data_OM=1_k=5_N.pkl', 'rb') as f:
    data = pkl.load(f)
    t = data['t']
    nvals = data['nvals']
    Pnt = data['Pnt']
    pchis = data['pchis']
    chilist = data['chilist']

with open('./Data/data_OM=1_k=0.5_N.pkl', 'rb') as f:
    data2 = pkl.load(f)
    t = data['t']
    nvals = data['nvals']
    Pnt2 = data2['Pnt']
    pchis = data['pchis']
    chilist = data['chilist']



plt.plot(nvals, Pnt[:, 1:], linewidth=0.3, color='k', alpha=0.5)
plt.plot(nvals, Pnt2[:, 1:], linewidth=0.3, color='r', alpha=0.5)
plt.show()
