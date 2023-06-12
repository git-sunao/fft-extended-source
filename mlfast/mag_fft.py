"""
python module for calculating microlensing magnification with finite source size effect
by Sunao Sugiyama
Jan 19, 2022
"""

import numpy as np
from. import fftlog
from scipy.special import j0, j1, jn, gamma
from scipy.special import ellipk as spellipk
from scipy.special import ellipe as spellipe
from scipy.interpolate import InterpolatedUnivariateSpline as ius


# analytic expressions of Fourier counterparts of source star profiles, disk, limb and higher order limb.
def sk_disk(k, rho):
    x = k*rho
    ans = np.ones(x.shape) # x -> 0 limit
    sel = x>0
    ans[sel] = 2*j1(x[sel])/x[sel]
    return ans

def sk_limb(k, rho, n):
    x = k*rho
    ans = 1.0/(n+2)*np.ones(x.shape) # x-> 0 limit
    sel = x > 0
    nu = 1+n/2
    ans[sel] = 2**nu*gamma(nu)*jn(nu, x[sel])/x[sel]**nu * nu
    return ans

# FFT based magnification
class magnification:
    def __init__(self, profile_names=['disk'], profile_args=None, normalize_sk=True):
        self.fft_umin = -6 # this choice is validated for rho > 1e-4 to ensure 0.3% precision
        self.fft_umax =  3
        self.N_fft = 1024
        self.zero_alloc = 1e-200
        self.normalize_sk = normalize_sk
        self.set_profile(profile_names, profile_args)
        self.init_Apk()
        self.init_small_rho()
        
    def set_profile(self, profile_names, profile_args):
        self.profile_names = profile_names
        if profile_args is None:
            self.profile_args = dict()
        else:
            self.profile_args = profile_args
        for profile_name in self.profile_names:
            if profile_name in self.profile_args:
                continue
            self.profile_args[profile_name] = dict()
        
    def init_Apk(self):
        u = np.logspace(self.fft_umin, self.fft_umax, self.N_fft)
        u2Au = ((u**2+2.0)/(u**2+4.0)**0.5/u - 1) * u**2
        h = fftlog.hankel(u, u2Au, nu=1.5, N_extrap_high=512, N_extrap_low=512)
        self.k, apk = h.hankel(0)
        self.apk = apk*2*np.pi
        
    def init_small_rho(self, alloc=True):
        x = np.logspace(-5, 5, 1024)
        dump = np.exp(-(x/100)**2)
        for profile_name in self.profile_names:
            if profile_name == 'disk':
                fx = x * sk_disk(x, 1) * dump
                h = fftlog.hankel(x, fx, nu=1.5, N_pad=1024)
                u, af0 = h.hankel(0)
                self.profile_args[profile_name]['af0'] = ius(u, af0, ext=3)
            elif profile_name in ['limb1']:
                fx = x * sk_limb(x, 1, 1) * dump
                h = fftlog.hankel(x, fx, nu=1.5, N_pad=1024)
                u, af0 = h.hankel(0)
                self.profile_args[profile_name]['af0'] = ius(u, af0, ext=3)
            elif profile_name in ['limb2']:
                fx = x * sk_limb(x, 1, 2) * dump
                h = fftlog.hankel(x, fx, nu=1.5, N_pad=1024)
                u, af0 = h.hankel(0)
                self.profile_args[profile_name]['af0'] = ius(u, af0, ext=3)
            else:
                # compute sk
                u = np.logspace(self.fft_umin, self.fft_umax, self.N_fft)
                su = self.profile_args[profile_name]['su'](u, 1)
                # beyond rho should be set to nonzero small value for FFT convergence.
                u2su = u**2*su
                if alloc:
                    u2su[u>=1] = self.zero_alloc
                h = fftlog.hankel(u, u2su, nu=1.5, N_extrap_high=512, N_extrap_low=512)
                k, sk = h.hankel(0)
                
                fx = x * ius(k,sk,ext=1)(x) * dump
                h = fftlog.hankel(x, fx, nu=1.5, N_pad=1024)
                u, af0 = h.hankel(0)
                self.profile_args[profile_name]['af0'] = ius(u, af0, ext=3)
        # this approximation is validated for `rho<self.rho_thre`
        self.rho_thre = 1e-4
        # this approximation is validated for `u > self.u_thre*rho`.
        self.u_thre = 10
        
    def A(self, u, rho, profile_name):
        if profile_name == 'disk':
            return self.A_disk(u, rho)
        elif profile_name == 'limb1':
            return self.A_limb1(u, rho)
        elif profile_name == 'limb2':
            return self.A_limb2(u, rho)
        else:
            return self.A_user(u, rho, profile_name)
    
    def A_point(self, u):
        return (u**2+2)/u/(u**2+4)**0.5
        
    def A_disk(self, u, rho):
        u = np.atleast_1d(u)
        if rho == 0.0:
            return self.A_point(u)
        elif rho < self.rho_thre:
            a = np.ones(u.shape)
            sel = u<self.u_thre*rho
            a[sel] = self.profile_args['disk']['af0'](u[sel]/rho) / rho + 1
            sel = u>=self.u_thre*rho
            a[sel] = self.A_point(u[sel])
            return a
        else:
            k_rho = 2*np.pi/rho
            dump = np.exp(-(self.k/k_rho/50)**2)
            cj = self.apk*self.k**2 * sk_disk(self.k, rho) * dump
            h = fftlog.hankel(self.k, cj, nu=1.5, N_pad=512)
            u_fft, a_fft = h.hankel(0)
            a_fft = a_fft/2/np.pi
            a_fft = a_fft + 1
            
            a = np.ones(u.shape)
            sel = np.abs(u) > 0
            u_pad = [100]
            a_pad = [1]*len(u_pad)
            sel_fft = u_fft<100
            a[sel] = log_interp(np.concatenate([u_fft[sel_fft], u_pad]), 
                                np.concatenate([a_fft[sel_fft], a_pad]), np.abs(u[sel]) )
            a[u==0] = (rho**2+4)**0.5/rho
            
            return a
        
    def A_limb1(self, u, rho):
        u = np.atleast_1d(u)
        if rho == 0.0:
            return self.A_point(u)
        elif rho < self.rho_thre:
            a = np.ones(u.shape)
            sel = u<self.u_thre*rho
            a[sel] = self.profile_args['limb1']['af0'](u[sel]/rho) / rho + 1
            sel = u>=self.u_thre*rho
            a[sel] = self.A_point(u[sel])
            return a
        else:
            k_rho = 2*np.pi/rho
            dump = np.exp(-(self.k/k_rho/50)**2)
            cj = self.apk*self.k**2 * sk_limb(self.k, rho, 1) * dump
            h = fftlog.hankel(self.k, cj, nu=1.5, N_pad=512)
            u_fft, a_fft = h.hankel(0)
            a_fft = a_fft/2/np.pi
            a_fft = a_fft + 1
            
            a = np.ones(u.shape)
            sel = np.abs(u) > 0
            u_pad = [100]
            a_pad = [1]*len(u_pad)
            sel_fft = u_fft<100
            a[sel] = log_interp(np.concatenate([u_fft[sel_fft], u_pad]), 
                                np.concatenate([a_fft[sel_fft], a_pad]), np.abs(u[sel]) )
            a[u==0] = (2+1) * (2*(rho**2+2)*spellipe(-rho**2/4)-(rho**2+4)*spellipk(-rho**2/4)) / 3.0/rho**3
            
            return a
        
    def A_limb2(self, u, rho):
        u = np.atleast_1d(u)
        if rho == 0.0:
            return self.A_point(u)
        elif rho < self.rho_thre:
            a = np.ones(u.shape)
            sel = u<self.u_thre*rho
            a[sel] = self.profile_args['limb2']['af0'](u[sel]/rho) / rho + 1
            sel = u>=self.u_thre*rho
            a[sel] = self.A_point(u[sel])
            return a
        else:
            k_rho = 2.0*np.pi/rho
            dump = np.exp(-(self.k/k_rho/50)**2)
            cj = self.apk*self.k**2 * sk_limb(self.k, rho, 2) * dump
            h = fftlog.hankel(self.k, cj, nu=1.5, N_pad=512)
            u_fft, a_fft = h.hankel(0)
            a_fft = a_fft/2/np.pi
            a_fft = a_fft + 1
            
            a = np.ones(u.shape)
            sel = np.abs(u) > 0
            u_pad = [100]
            a_pad = [1]*len(u_pad)
            sel_fft = u_fft<100
            a[sel] = log_interp(np.concatenate([u_fft[sel_fft], u_pad]), 
                                np.concatenate([a_fft[sel_fft], a_pad]), np.abs(u[sel]) )
            a[u==0] = (2+2) * (rho*(2+rho**2)*(4+rho**2)**0.5 - 8*np.arcsinh(rho/2)) / 4/rho**4
            
            return a
        
    # functions below are used for finite source magnification 
    # with user-defined source profile
    def compute_sk(self, rho, profile_name, alloc=True):
        r = np.logspace(self.fft_umin, self.fft_umax, self.N_fft)
        su = self.profile_args[profile_name]['su'](r, rho)
        # beyond rho should be set to nonzero small value for FFT convergence.
        u2su = r**2*su
        if alloc:
            u2su[r>=rho] = self.zero_alloc
        h = fftlog.hankel(r, u2su, nu=1.5, N_extrap_high=512, N_extrap_low=512)
        k, sk = h.hankel(0)
        sk = sk*2*np.pi
        if self.normalize_sk:
            sk = sk/sk[0]
        return k, sk
        
    def A_user(self, u, rho, profile_name, alloc=True):
        u = np.atleast_1d(u)
        if rho == 0.0:
            return self.A_point(u)
        elif rho < self.rho_thre:
            a = np.ones(u.shape)
            sel = u<self.u_thre*rho
            a[sel] = self.profile_args[profile_name]['af0'](u[sel]/rho) / rho + 1
            sel = u>=self.u_thre*rho
            a[sel] = self.A_point(u[sel])
            return a
        else:
            # compute sk
            _, sk = self.compute_sk(rho, profile_name, alloc=alloc)
            
            # compute finite source
            k_rho = 2.0*np.pi/rho
            dump = np.exp(-(self.k/k_rho/50)**2)
            cj = self.apk*self.k**2 * sk * dump
            h = fftlog.hankel(self.k, cj, nu=1.5, N_pad=512)
            u_fft, a_fft = h.hankel(0)
            a_fft = a_fft/2/np.pi
            a_fft = a_fft + 1
            
            a = np.ones(u.shape)
            sel = np.abs(u) > 0
            u_pad = [100]
            a_pad = [1]*len(u_pad)
            sel_fft = u_fft<100
            a[sel] = log_interp(np.concatenate([u_fft[sel_fft], u_pad]), 
                                np.concatenate([a_fft[sel_fft], a_pad]), np.abs(u[sel]) )
            a[u==0] = self.profile_args[profile_name].get('a0', 0.0)
            
            return a
        

### Utility functions ####################

def log_interp(x,y,xnew):
    """
    Apply interpolation in logarithmic space for both x and y.
    Beyound input x range, returns 10^0=1
    """
    ynew = 10**ius(np.log10(x), np.log10(y), ext=3)(np.log10(xnew))
    return ynew