"""
python module for calculating microlensing magnification with finite source size effect
by Sunao Sugiyama
Jan 19, 2022
"""

import numpy as np
import os
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.special import ellipk as spellipk
from scipy.special import ellipe as spellipe
from scipy.integrate import quad

# Method introduced initially by Gould.
# Generalized to linear limb-darkening by Yoo et al. 2004.
# Implimented based on arxiv: astro-ph/0309302 
def make_b0():
    z = np.linspace(2e-4, 4, 500)
    
    def integrand(r, t, _z):
        _u = (r**2+_z**2-2*r*_z*np.cos(t))**0.5
        return r/_u
    
    a = []
    for _z in z:
        _a = quad(lambda _t: quad(integrand, 0, 1, args=(_t, _z))[0], 0, np.pi)[0]
        a.append(_a)
    b0 = np.array(a)*z/np.pi*2
    np.savetxt(os.path.join(os.path.dirname(__file__), 'b0.txt'), np.array([z,b0]).T)

def make_b1():
    z = np.linspace(2e-4, 4, 500)
    
    def integrand(r, t, _z):
        _u = (r**2+_z**2-2*r*_z*np.cos(t))**0.5
        return r*(1-r**2)**0.5/_u
    
    a = []
    for _z in z:
        _a = quad(lambda _t: quad(integrand, 0, 1, args=(_t, _z))[0], 0, np.pi)[0]
        a.append(_a)
    b1 = np.array(a)*z/np.pi*2
    np.savetxt(os.path.join(os.path.dirname(__file__), 'b1.txt'), np.array([z,b1]).T)
    
def make_b2():
    z = np.linspace(2e-4, 4, 500)
    
    def integrand(r, t, _z):
        _u = (r**2+_z**2-2*r*_z*np.cos(t))**0.5
        return r*(1-r**2)/_u
    
    a = []
    for _z in z:
        _a = quad(lambda _t: quad(integrand, 0, 1, args=(_t, _z))[0], 0, np.pi)[0]
        a.append(_a)
    b2 = np.array(a)*z/np.pi*2
    np.savetxt(os.path.join(os.path.dirname(__file__), 'b2.txt'), np.array([z,b2]).T)
    
def make_b12():
    z = np.linspace(2e-4, 4, 500)
    
    def integrand(r, t, _z):
        _u = (r**2+_z**2-2*r*_z*np.cos(t))**0.5
        return r*(1-r**2)**0.25/_u
    
    a = []
    for _z in z:
        _a = quad(lambda _t: quad(integrand, 0, 1, args=(_t, _z))[0], 0, np.pi)[0]
        a.append(_a)
    b12 = np.array(a)*z/np.pi*2
    np.savetxt(os.path.join(os.path.dirname(__file__), 'b12.txt'), np.array([z,b12]).T)
    

def B0(z):
    _z, _b0 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'b0.txt'), unpack=True)
    return ius(_z,_b0, ext=3)(z)

def B1(z):
    _z, _b1 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'b1.txt'), unpack=True)
    return ius(_z,_b1, ext=3)(z)*3/2

def B2(z):
    _z, _b2 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'b2.txt'), unpack=True)
    return ius(_z,_b2, ext=3)(z)*2

def B12(z):
    _z, _b12 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'b12.txt'), unpack=True)
    return ius(_z,_b12, ext=3)(z)*5/4

def A_disk(u, rho):
    z = u/rho
    
    ans = np.ones(u.shape)
    sel = u>0
    ans[sel] = (u[sel]**2+2)/u[sel]/(u[sel]**2+4)**0.5 * B0(z[sel])
    ans[u==0] = (rho**2+4)**0.5/rho
    
    return ans

def A_limb(u, rho):
    z = u/rho
    
    ans = np.ones(u.shape)
    sel = u>0
    ans[sel] = (u[sel]**2+2)/u[sel]/(u[sel]**2+4)**0.5 * B1(z[sel])
    ans[u==0] = (2+1) * (2*(rho**2+2)*spellipe(-rho**2/4)-(rho**2+4)*spellipk(-rho**2/4)) / 3.0/rho**3
    
    return ans

def A_para(u, rho):
    z = u/rho
    
    ans = np.ones(u.shape)
    sel = u>0
    ans[sel] = (u[sel]**2+2)/u[sel]/(u[sel]**2+4)**0.5 * B2(z[sel])
    ans[u==0] = (2+2) * (rho*(2+rho**2)*(4+rho**2)**0.5 - 8*np.arcsinh(rho/2)) / 4/rho**4
    
    return ans
    