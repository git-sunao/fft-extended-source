import numpy as np
import os
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from numpy.fft import rfft, irfft

def sinovx(x):
    ans = np.ones(x.shape)
    sel = x != 0
    ans[sel] = np.sin(x[sel])/x[sel]
    return ans

def lin_extrap(x, N_extrap_low, N_extrap_high):
    low_x = high_x = []
    if(N_extrap_low):
        dx_low = x[1]-x[0]
        low_x = x[0] + dx_low * np.arange(-N_extrap_low, 0)
    if(N_extrap_high):
        dx_high= x[-1]-x[-2]
        high_x = x[-1] + dx_high * np.arange(1, N_extrap_high+1)
    x_extrap = np.hstack((low_x, x, high_x))
    return x_extrap

def timeave(t, a, dt, N_pad=1024):
    """
    Parameters
    ----------
    t : array of time. must be linearly spaced. Peak time must be included within this time range.
    a : array of magnification A(t).
    dt: exposure time (averaging time)
    N_pad: padding number for time, both lower and higher sides.
    """

    # padding
    if N_pad>0:
        t_pad = lin_extrap(t, N_pad, N_pad)
        a_pad = np.hstack((np.ones(N_pad), a, np.ones(N_pad)))
        N_total = t_pad.size
    else:
        t_pad = t.copy()
        a_pad = a.copy()
        N_total = t_pad.size

    # compute Fourier counterpart 
    c_m=rfft(a_pad)
    m=np.arange(0,N_total//2+1)
    eta_m = 2*np.pi/(float(N_total)*np.diff(t_pad)[0]) * m

    # exposure time average
    a_bar_pad = irfft(np.conj(c_m) * sinovx(eta_m*dt/2.0) )[::-1]
    a_bar = ius(t_pad, a_bar_pad)(t)
    
    return a_bar