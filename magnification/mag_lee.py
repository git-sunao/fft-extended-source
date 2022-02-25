"""
python module for calculating microlensing magnification with finite source size effect
by Sunao Sugiyama
Jan 19, 2022
"""

import numpy as np
import os
from ctypes import c_double, c_int
from scipy.special import ellipk as spellipk
from scipy.special import ellipe as spellipe
from . import mag_wm

# Method developped by Lee et al. 2018
# Implimented based on arxiv: 0901.1316 
# This technique uses simpson integration. 
# So naive implementation by python-for-loop is slow. 
# For this, we implemented this method in C and wrapped by python.

try:
    lib = np.ctypeslib.load_library(os.path.join(os.path.dirname(__file__), 'mag_C.so'),'.')
    lib.A_disk.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64), 
                           c_double, 
                           np.ctypeslib.ndpointer(dtype=np.float64), 
                           c_int,
                           c_int]
    lib.A_limb.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64), 
                           c_double, 
                           np.ctypeslib.ndpointer(dtype=np.float64), 
                           c_int,
                           c_int, 
                           c_int,
                           c_double]
except:
    print('library not found.')
    c = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Makefile')
    print(f'Run {c}')

def A_disk(u, rho, n=100):
    if isinstance(u, float) or isinstance(u, int):
        u = np.array([u])
    
    Ngrid = u.size
    ans = np.zeros(u.shape)
    _ = lib.A_disk(u, rho, ans, n, Ngrid)
    return ans

def A_limb(u, rho, n1=100, n2=100, order=1):
    if isinstance(u, float) or isinstance(u, int):
        u = np.array([u])
    
    Ngrid = u.size
    ans = np.zeros(u.shape)
    _ = lib.A_limb(u, rho, ans, n1, n2, Ngrid, order)
    return ans
    
