"""
python module for calculating microlensing magnification with finite source size effect
by Sunao Sugiyama
Jan 19, 2022
"""

import numpy as np
import os
from scipy.special import ellipk as spellipk
from scipy.special import ellipe as spellipe
from mpmath import ellippi as mpellippi

# Overwrite notation to widely used in microlensing literature
def ellipk(k):
    return spellipk(k**2)

def ellipe(k):
    return spellipe(k**2)

def ellippi(n, k):
    # see for notation https://mpmath.org/doc/current/functions/elliptic.html
    ans = []
    for _n ,_k in zip(n,k):
        a = mpellippi(_n, np.pi/2.0, _k**2)
        ans.append(float(a))
    return np.array(ans)

def A_disk(u, rho):
    """
    Originally derived in Witt & Mao (1994).
    Implimented based on Mon. Mao & Witt, Not. R. Astron. Soc. 300, 1041–1046 (1998)
    """
    b1 = -(8-rho**2+u**2)*(u-rho)
    b2 = (4+(rho-u)**2)*(u+rho)
    b3 = 4*(1+rho**2)*(rho-u)**2/(rho+u)
    n = 4*u*rho/(u+rho)**2
    k = (4*n/(4+(u-rho)**2))**0.5
    prefactor = 1/2/np.pi/rho**2/(4+(u-rho)**2)**0.5
    return prefactor*(b1*ellipk(k)+b2*ellipe(k)+b3*ellippi(n,k))