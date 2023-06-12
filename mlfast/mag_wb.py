"""
python module for calculating microlensing magnification with finite source size effect
by Sunao Sugiyama
Jan 19, 2022
"""

import numpy as np
from scipy.special import ellipk as spellipk
from scipy.special import ellipe as spellipe
from . import mag_wm

# Overwrite notation to widely used in microlensing literature
def ellipk(k):
    return spellipk(k**2)

def ellipe(k):
    return spellipe(k**2)

# Method developped by Witt & Barandela 2019
# Implimented based on arxiv: 1906.08378
def A_WittBarandela2019(x, rho, c, f, e, match=None):
    k_coeff = 0
    e_coeff = 0
    for n in range(3):
        k_coeff += c[n]*rho**(2*n-1)/np.pi * f[n]
        e_coeff += c[n]*rho**(2*n-1)/np.pi * e[n]
        
    k = 2*x**0.5/(1+x)
    mu_ext = (1-x)*k_coeff*ellipk(k) + (1+x)*e_coeff*ellipe(k)
    
    if match is not None:
        mu_ext[x>match] = mag_wm.A_disk(x[x>match]*rho, rho)
    
    return mu_ext

def A_disk(u, rho):
    x = u/rho
    
    c = [1, 3/8, -5/128]
    f = [2, -2/9*(1-x**2), -2/75*(1-x**2)*(13+3*x**2)]
    e = [2,  2/9*(7+x**2),  2/75*(43+82*x**2+3*x**4)]
    
    return A_WittBarandela2019(x, rho, c,f,e)

def A_limb(u, rho):
    x = u/rho
    
    c  = [1, 3/8, -5/128]
    f = []
    e = []
    
    a = [692583091200, 1339448033263, 312948159897, 72606018032, 502361257, 4109895, 79511, 835, 308]
    f0 = (a[1]-2*x**2*(a[2]+a[3]*x**2+64*x**4*(a[4]+64*x**2*(a[5]+28*x**2*(a[6]+48*x**2*(a[7]+a[8]*x**2))))))/a[0]
    f.append(f0)
    
    a = [1385166182400, 3894814720117, 243184124641, 41938764464, 252468377, 1997995, 41279, 527, 308]
    e0 = (a[1]-4*x**2*(a[2]+a[3]*x**2+64*x**4*(a[4]+64*x**2*(a[5]+28*x**2*(a[6]+48*x**2*(a[7]+a[8]*x**2))))))/a[0]
    e.append(e0)
    
    a = [400313026713600, -104737154144542, 118605243978153, -12149180921700, 19577824287, 5187655332, 7438983, 111191, 963, 308]
    f1 = (a[1]+a[2]*x**2+a[3]*x**4-64*x**6*(a[4]+a[5]*x**2+256*x**4*(a[6]+28*x**2*(a[7]+48*x**2*(a[8]+a[9]*x**2)))))/a[0]
    f.append(f1)
    
    a = [400313026713600, 586396203521047, 130620128454965, 2732363448097, 233199164080, 840367513, 4461547, 66815, 655, 308]
    e1 = (a[1]+a[2]*x**2-4*x**4*(a[3]+a[4]*x**2+64*x**4*(a[5]+64*x**2*(a[6]+28*x**2*(a[7]+48*x**2*(a[8]+a[9]*x**2))))))/a[0]
    e.append(e1)
    
    a = [144513002643609600, 41601097496530466, 25747524261728378, 16403635906248033, -760401005566596, 683702230503, 29226547953, 120111823, 1392911, 1107, 308]
    f2 = (a[1]+a[2]*x**2+a[3]*x**4-a[4]*x**6-64*x**8*(a[5]+4*x**2*(a[6]+64*x**2*(a[7]+28*x**2*(a[8]+432*x**2*(a[9]+a[10]*x**2))))))/a[0]
    f.append(f2)
    
    a = [144513002643609600, 129881415373196381, 302061597305296826, 17159057255113677, -717449096541348, 569981448915, 87184849284, 82551251, 931319, 799, 308]
    e2 = (a[1]+a[2]*x**2+a[3]*x**4+a[4]*x**6-64*x**8*(a[5]+a[6]*x**2+256*x**4*(a[7]+28*x**2*(a[8]+432*x**2*(a[9]+a[10]*x**2)))))/a[0]
    e.append(e2)
    
    return A_WittBarandela2019(x, rho,c,f,e, 2.5)

def A_para(u, rho):
    x = u/rho
    
    c = [1, 3/8, -5/128]
    f = [16/9*(1-x**2), -16/225*(1-x**2)*(4-x**2), -16/3675*(1-x**2)*(53+30*x**2-3*x**4)]
    e = [16/9*(2-x**2),  16/225*(19+6*x**2-x**4) ,  16/3675*(158+449*x**2+36*x**4-3*x**6)]
    
    if rho>=0.3:
        match = 3
    else:
        match = None
    return A_WittBarandela2019(x, rho,c,f,e, match)


