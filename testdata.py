import numpy as np
import os
from scipy.integrate import simps, quad
from magnification import mag_wm

def s_disk(u, rho):
    if isinstance(u, float) or isinstance(u, int):
        u = np.array([u])
    ans = np.zeros(u.shape)
    ans[u<rho] = 1.0/np.pi/rho**2
    return ans

def s_limb(u, rho):
    ans = np.zeros(u.shape)
    ans[u<rho] = 3/2*(1-u**2/rho**2)**0.5/np.pi/rho**2
    return ans
    
def s_para(u, rho):
    ans = np.zeros(u.shape)
    ans[u<rho] = 2*(1-u**2/rho**2)/np.pi/rho**2
    return ans

def Ap(u):
    return (u**2+2)/u/(u**2+4)**0.5

def arccos(x,y):
    if y == 0:
        return 0.0
    else:
        if x/y<=-1:
            return 2*np.pi
        elif x/y>=1:
            return 0.0
        else:
            return np.arccos(x/y)
        
def A_2d_numeric_simps(u, s, rho, Ntheta=100, Nr=100):
    z = u/rho

    t = np.linspace(0, 2*np.pi, Ntheta)
    r = np.linspace(0, 1, Nr)
    w = []
    for _z in z:
        i = []
        for _t in t:
            u = rho*(r**2+_z**2-2*r*_z*np.cos(_t))**0.5
            integrand = r * s(r*rho, rho) * Ap(u)
            i.append( simps(integrand, r))
        w.append( simps(np.array(i), t) )
    a = np.array(w)*rho**2
    return a

def A_2d_numeric_quads(u, s, rho):
    z = u/rho
    
    def integrand(r, theta, _z):
        _u = rho*(r**2+_z**2-2*r*_z*np.cos(theta))**0.5
        return r*s(r*rho, rho) * Ap(_u)
    
    a = []
    for _z in z:
        _a = quad(lambda _t: quad(integrand, 0, 1, args=(_t, _z))[0], 0, np.pi)[0]
        a.append(2*_a* rho**2)
        
    return np.array(a)

def A_disk(u, rho):
    return A_2d_numeric_quads(u, s_disk, rho)

def A_limb(u, rho):
    return A_2d_numeric_quads(u, s_limb, rho)

def A_para(u, rho):
    return A_2d_numeric_quads(u, s_para, rho)


def gen(s, rho, fname, int_method='quads', int_args={'Ntheta':7000, 'Nr':700}, u=None):
    if u is None:
        u = rho*np.logspace(-5, 3, 200)
    if int_method == 'simps':
        a_2d_nu = A_2d_numeric_simps(u, s, rho, Ntheta=int_args['Ntheta'], Nr=int_args['Nr'])
        header = '%e\nsimps\nNtheta=%d\nNr=%d'%(rho,int_args['Ntheta'],int_args['Nr'])
    if int_method == 'quads':
        a_2d_nu = A_2d_numeric_quads(u, s, rho)
        header = '%e\nquads'%rho
    np.savetxt(os.path.join(os.path.dirname(__file__), fname), 
               np.array([u, a_2d_nu]).T, header=header)
    
def gen_disk_wm(rho, fname, u):
    a = mag_wm.A_disk(u, rho)
    header = '%e\nWittMao'%rho
    np.savetxt(os.path.join(os.path.dirname(__file__), fname), 
               np.array([u, a]).T, header=header)
    
    
def get_test_data(fname):
    with open(fname, 'r') as f:
        l = f.readline()
        rho = float(l.replace('#','').replace('\n',''))
    u, a = np.loadtxt(fname, unpack=True)
    return u, rho, a


if __name__ == '__main__':
    dirname = 'testdata'
    
    # generate disk test data
    if False:
        s = s_disk
        rhos = [1e-5, 1e-3, 1e-1, 1]
        fnames = ['disk_1e-5.txt', 'disk_1e-3.txt', 'disk_1e-1.txt', 'disk_1.txt']
        for rho, fname in zip(rhos, fnames):
            print(rho)
            u = rho*np.logspace(-5, 3, 200)
            gen(s, rho, os.path.join(dirname,fname), u=u)
        
    # generate limb test data
    if False:
        s = s_limb
        rhos = [1e-5, 1e-3, 1e-1, 1]
        fnames = ['limb_1e-5.txt', 'limb_1e-3.txt', 'limb_1e-1.txt', 'limb_1.txt']
        for rho, fname in zip(rhos, fnames):
            print(rho)
            u = rho*np.logspace(-5, 3, 200)
            gen(s, rho, os.path.join(dirname,fname), u=u)
        
    # generate para test data
    if False:
        s = s_para
        rhos = [1e-5, 1e-3, 1e-1, 1]
        fnames = ['para_1e-5.txt', 'para_1e-3.txt', 'para_1e-1.txt', 'para_1.txt']
        for rho, fname in zip(rhos, fnames):
            print(rho)
            u = rho*np.logspace(-5, 3, 200)
            gen(s, rho, os.path.join(dirname,fname), u=u)

    # generate test data for 2d residual plot
    dirname = 'testdata2dquads'
    if False:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        rhos = np.logspace(-5, 2, 100)
        s = s_disk
        #s = s_limb
        #s = s_para
        name = 'disk'
        #name = 'limb'
        #name = 'para'
        
        perrank =  rhos.size//size
        for i in range(rank*perrank, (rank+1)*perrank):
            rho = rhos[i]
            print(rho)
            u = rho*np.logspace(-5, 3, 200)
            gen(s, rho, os.path.join(dirname, '%s_%d'%(name,i)), u=u)
            