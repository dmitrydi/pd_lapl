import numpy as np
from scipy.special import iti0k0, k0
from scipy.integrate import quad
from scipy.integrate import fixed_quad

def sexp(e_, yed):
    TINY = 1e-20
    EPS = 1e-9
    MAXITER = 10000
    BLK_SIZE = 100
    if isinstance(e_, np.ndarray):
        e = e_.flatten().reshape(-1,1)
        sum_ = np.zeros_like(e_.flatten())
    else:
        e = e_
        sum_ = 0
    for i in range(MAXITER):
        m = np.arange(1+i*BLK_SIZE, 1+(i+1)*BLK_SIZE)
        d = np.sum(np.exp(-2*m*yed*e), axis=-1)
        sum_ += d
        if np.all(d < sum_*EPS) or np.all(sum_<TINY):
            return sum_
    return sum_

def F1(u, yd, ywd, yed):
    sum_ = 0
    su = np.sqrt(u)
    u_sexp = sexp(su, yed)
    ans = np.exp(-su*(2*yed-(yd+ywd))) + np.exp(-su*(yd+ywd)) + np.exp(-su*(2*yed-np.abs(yd-ywd))) +\
            np.exp(-su*np.abs(yd-ywd))
    ans *= (1+u_sexp)/su
    return ans

def iF1(x1, x2, u, yd, ywd, yed):
    return (x2-x1)*F1(u, yd, ywd, yed)

def iF2E(x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed):
    EPS = 1e-12
    TINY = 1e-20
    blk_size = 10
    MAXITER = 1000
    mshape = ksid.shape
    sum_ = np.zeros(mshape, dtype=np.double)
    buf = []
    for i in range(MAXITER):
        d = np.zeros(mshape, dtype=np.double)
        k = np.arange(1+i*blk_size, 1+(i+1)*blk_size)
        d = np.sum(iF2Ek(k, buf, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed), axis=0)
        sum_ += d
        ns = np.linalg.norm(sum_) 
        if np.linalg.norm(d) < EPS*ns or ns < TINY :
            return sum_
    return sum_
        
def iF2Ek(k_, buf, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed):
    pi = np.pi
    if len(buf)==0:
        argx1 = pi*ksid/ksied
        argx2 = 0.5*pi/ksied*(x2-x1)
        argx3 = 0.5*pi/ksied*(2*ksiwd+x1+x2)
        argy1 = 2*yed-(yd+ywd)
        argy2 = yd+ywd
        argy3 = 2*yed - np.abs(yd-ywd)
        argy4 = np.abs(yd-ywd)
        buf = argx1, argx2, argx3, argy1, argy2, argy3, argy4
    elif len(buf) == 7:
        argx1, argx2, argx3, argy1, argy2, argy3, argy4 = buf
    else:
        raise RuntimeError("There is a problem with buffer in iF2Ek")
    ek_ = np.sqrt(u+np.square(k_*np.pi/ksiede)+np.square(a))
    _sexp_ = sexp(ek_, yed)
    if isinstance(k_, np.ndarray):
        _sexp_ = _sexp_.reshape(-1,1,1)
        k = k_.reshape(-1,1,1)
        ek = ek_.reshape(-1,1,1)
    else:
        k = k_
        ek = ek_
    ans = 2.*ksied/(pi*k*ek)*np.cos(k*argx1)*np.sin(k*argx2)*np.cos(k*argx3)
    ans *= (1+_sexp_)*(np.exp(-ek*argy1)+np.exp(-ek*argy2)+np.exp(-ek*argy3))+_sexp_*np.exp(-ek*argy4)
    return ans

def i1F2H(buf_yd0, buf_yd_nnz, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd):
    TINY = 1e-20
    mshape = x1.shape
    assert x2.shape == mshape
    sum_ = i1F2Hk(buf_yd0, buf_yd_nnz, 0, 1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
    sum_ += i1F2Hk(buf_yd0, buf_yd_nnz, 0, -1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
    EPS = 1e-12
    blk_size = 3
    MAXITER = 100
    for i in range(MAXITER):
        d = np.zeros(mshape, dtype = np.double)
        for k in range(1+i*blk_size, 1+(i+1)*blk_size):
            d+= i1F2Hk(buf_yd0, buf_yd_nnz, k, 1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
            d+= i1F2Hk(buf_yd0, buf_yd_nnz, -k, 1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
            d+= i1F2Hk(buf_yd0, buf_yd_nnz, k, -1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
            d+= i1F2Hk(buf_yd0, buf_yd_nnz, -k, -1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
        sum_ += d
        nrm = np.linalg.norm(sum_)
        if np.linalg.norm(d) < EPS*nrm or nrm < TINY:
            return sum_

def i1F2Hk(buf_yd0, buf_yd_nnz, k, b, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd):
    NDIGITS = 6
    mshape = x1.shape
    assert x2.shape == mshape
    m = np.zeros(mshape, dtype=np.double)
    inds_dyd0 = np.isclose(yd,ywd)
    inds_dyd_nnz = np.logical_not(inds_dyd0)
    su = np.sqrt(u+np.square(a))
    if np.any(inds_dyd0):
        if str(k)+str(b) not in buf_yd0.keys():
            t1 = ksiede/ksied*(ksid[inds_dyd0] + b*ksiwd[inds_dyd0] - 2*k*ksied - x2[inds_dyd0])
            t2 = ksiede/ksied*(ksid[inds_dyd0] + b*ksiwd[inds_dyd0] - 2*k*ksied - x1[inds_dyd0])
            mask1 = np.ones_like(t1) - 2*(t1>0)
            mask2 = np.ones_like(t1) - 2*(t2<0)
            t1 = np.round(np.abs(t1), decimals=NDIGITS).flatten()
            t2 = np.round(np.abs(t2), decimals=NDIGITS).flatten()
            nt = len(t1)
            t = np.append(t1,t2)
            ut, inv_t = np.unique(t, return_inverse=True)
            buf_yd0[str(k)+str(b)] = (ut, inv_t, nt, mask1, mask2)
        else:
            ut, inv_t, nt, mask1, mask2 = buf_yd0[str(k)+str(b)]
        uv = ksied/ksiede/su*iti0k0(ut*su)[1]
        v = uv[inv_t]
        v1 = v[:nt]
        v2 = v[nt:]
        m[inds_dyd0] = v1*mask1+v2*mask2
    if np.any(inds_dyd_nnz):
        if str(k)+str(b) not in buf_yd_nnz.keys():
            t1 = ksiede/ksied*(ksid[inds_dyd_nnz] + b*ksiwd[inds_dyd_nnz] - 2*k*ksied - x2[inds_dyd_nnz])
            t2 = ksiede/ksied*(ksid[inds_dyd_nnz] + b*ksiwd[inds_dyd_nnz] - 2*k*ksied - x1[inds_dyd_nnz])
            dyd = np.round(np.abs(yd-ywd)[inds_dyd_nnz], decimals=NDIGITS).flatten()
            t1 = np.round(t1, decimals=NDIGITS).flatten()
            t2 = np.round(t2, decimals=NDIGITS).flatten()
            t = np.vstack([t1,t2]).T
            t = np.sort(t, axis=1)
            t = np.hstack([t,dyd[np.newaxis].T])
            ut, inv_t = np.unique(t, axis=0, return_inverse=True)
            buf_yd_nnz[str(k)+str(b)] = (ut, inv_t)
        else:
            (ut, inv_t) = buf_yd_nnz[str(k)+str(b)]
        g = lambda x: ksied/ksiede*(fixed_quad(m_bessk0, x[0], x[1], args=(x[2], su))[0])
        uv = np.apply_along_axis(g, 1, ut).flatten()
        m[inds_dyd_nnz] = uv[inv_t]
    return m

def m_bessk0(x, dyd, su):
    return k0(su*np.sqrt(np.square(x)+np.square(dyd)))

def i2F2H(x1,x2,u,a,yd,ywd):
    su = np.sqrt(u+a*a)
    return -0.5*np.exp(-su*np.abs(yd-ywd))/su*(x2-x1)