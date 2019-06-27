from .aux import eulsum
from scipy.special import k0, iti0k0
from .ffuncs import *

def IF3(u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid1, fid2):
    if3 = IF31(u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid1) +\
            IF32(u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid2)
    return if3

def IF31(u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid):
    return ihF2E(zd, zwd, x1, x2, u, xd, xwd, xed, xed, hd, yd, ywd, yed, buf, fid)

def IF32(u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid):
    if32 = IF321(u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid+"_1") +\
        IF322(u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid+"_2")
    return if32

def IF321(u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid):
    MAXITER = 1000
    EPS = 1e-6
    coef = 0.5*xed/np.pi
    sum1, sum2, m = np.zeros_like(x1), np.zeros_like(x1), np.zeros_like(x1)
    for n in range(1, MAXITER):
        v1, v2, vm = IF321_n(n, u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid)
        sum1 += v1
        sum2 += v2
        m += vm
        if np.linalg.norm(v1) < EPS*np.linalg.norm(sum1) and np.linalg.norm(v2) < EPS*np.linalg.norm(sum2):
            break
    ans = sum1 + sum2
    ans += IF321_aux_(u, zd, zwd, hd)*np.abs(0.5*m)
    return ans

def IF321_aux(u,zd, zwd, hd):
    MAXITER = 100000
    EPS = 1e-9
    sum_ = k0(np.sqrt(u*np.square((zd + zwd)*hd)))
    sum_ += k0(np.sqrt(u*np.square((zd - zwd)*hd)))
    for n in range(1, MAXITER):
        d = k0(np.sqrt(u*np.square((zd - zwd - 2*n)*hd))) + k0(np.sqrt(u*np.square((zd + zwd - 2*n)*hd)))
        d += k0(np.sqrt(u*np.square((zd - zwd + 2*n)*hd))) + k0(np.sqrt(u*np.square((zd + zwd + 2*n)*hd)))
        sum_ += d
        if np.linalg.norm(d) < EPS*np.linalg.norm(sum_):
            return 0.5*hd*sum_ - 0.5*np.pi/np.sqrt(u), sum_
    raise RuntimeWarning("IF321_aux did not converge")
    
def IF321_aux_(u,zd_, zwd_, hd):
    mshape = zd_.shape
    assert zwd_.shape == mshape
    zd = zd_[0,0]
    zwd = zwd_[0,0]
    sum_ = k0(np.sqrt(u*np.square((zd + zwd)*hd)))
    sum_ += k0(np.sqrt(u*np.square((zd - zwd)*hd)))
    for sz in [-1,1]:
        for sn in [-1,1]:
            vfunc = lambda n: k0(np.sqrt(u*np.square((zd + sz*zwd + sn*2*n)*hd)))
            sum_ += eulsum(vfunc)
    sum_ *= np.ones_like(zd_)
    return 0.5*hd*sum_ - 0.5*np.pi/np.sqrt(u)
        
def IF321_n(n, u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid):
    MAXITER = 1000
    EPS = 1e-6
    PI = np.pi
    mult = np.cos(n*PI*zd)*np.cos(n*PI*zwd)
    sum1, sum2, m = np.zeros_like(x1), np.zeros_like(x1), np.zeros_like(x1)
    for b in [-1, 1]:
        v1, v2, vm = IF321_nk(0, n, b, u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid)
        sum1 += v1
        sum2 += v2
        m += vm
    for k_ in range(1, MAXITER):
        for k in [-k_, k_]:
            for b in [-1, 1]:
                v1, v2, vm = IF321_nk(k, n, b, u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid)
                sum1 += v1
                sum2 += v2
                m += vm
                if np.linalg.norm(v1) < EPS*np.linalg.norm(sum1) and np.linalg.norm(v2) < EPS*np.linalg.norm(sum2):
                    return mult*sum1, mult*sum2, m
    raise RuntimeError
    
def IF321_nk(k, n, b, u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf, fid):
    # prep
    NDIGITS = 6
    mshape = x1.shape
    v_1 = np.zeros(mshape, dtype=np.double)
    v_2 = np.zeros(mshape, dtype=np.double)
    m1 = np.zeros(mshape, dtype=np.double)
    m2 = np.zeros(mshape, dtype=np.double)
    # calc
    dyd = yd - ywd
    inds_0 = np.isclose(dyd,0)
    inds_nnz = np.logical_not(inds_0)
    su = np.sqrt(u + np.square(n*np.pi/hd))
    if np.any(inds_0):
        t1 = xd[inds_0] + b*xwd[inds_0] - 2*k*xed - x2[inds_0]
        t2 = xd[inds_0] + b*xwd[inds_0] - 2*k*xed - x1[inds_0]
        mask1 = -1*(np.ones_like(t1) - 2*(t1>0))
        mask2 = -1*(np.ones_like(t1) - 2*(t2<0))
        t1 = np.round(np.abs(t1), decimals=NDIGITS).flatten()
        t2 = np.round(np.abs(t2), decimals=NDIGITS).flatten()
        nt = len(t1)
        t = np.append(t1,t2)
        ut, inv_t = np.unique(t, return_inverse=True)
        uv = ki1(ut*su)/su
        v = uv[inv_t]
        v1 = v[:nt]
        v2 = v[nt:]
        v_1[inds_0] = v1
        m1[inds_0] = mask1
        v_2[inds_0] = v2
        m2[inds_0] = mask2
    if np.any(inds_nnz):
        raise NotImplementedError
    return m1*v_1, m2*v_2, m1+m2  

def IF322(u, x1, x2, zd_, zwd_, hd, xd, xwd, xed, yd_, ywd_, yed, buf, fid):
    zd, zwd, yd, ywd = zd_[0,0], zwd_[0,0], yd_[0,0], ywd_[0,0]
    dyd = np.abs(yd-ywd)
    psum = k0(np.sqrt(u*(np.square((zd+zwd)*hd) + np.square(dyd)))) + k0(np.sqrt(u*(np.square((zd-zwd)*hd) + np.square(dyd))))
    for sz in [-1,1]:
        for sn in [-1,1]:
            func = lambda n: k0(np.sqrt(u*(np.square((zd+sz*zwd-2*sn*n)*hd) + np.square(dyd))))
            psum += eulsum(func)
    return 0.5*(x2-x1)*(0.5*hd*np.pi*psum - 0.5*np.exp(-np.sqrt(u)*dyd)/np.sqrt(u))
                        