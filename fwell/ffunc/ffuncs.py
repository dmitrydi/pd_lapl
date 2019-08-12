import numpy as np
from scipy.special import iti0k0, k0
from scipy.integrate import quad
from scipy.integrate import fixed_quad
from .cyfunc.cyfuncs import cy_m_bessk0, mcy_m_bessk0
from .integrate import qgaus
from time import time
from .aux import eulsum_v, sexp

#########################################################
# ____________________ frac-integrals____________________
#########################################################

def iF1(x1, x2, u, yd, ywd, yed):
    return (x2-x1)*F1(u, yd, ywd, yed)

def iF2(x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed, buf, fid1, fid2):
    ans = iF2E(x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed, buf, fid1)+\
    i1F2H(x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, buf, fid2)+i2F2H(x1,x2,u,a,yd,ywd)

def iF2E(x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed, buf, fid):
    EPS = 1e-12
    TINY = 1e-20
    blk_size = 10
    MAXITER = 1000
    mshape = ksid.shape
    sum_ = np.zeros(mshape, dtype=np.double)
    for i in range(MAXITER):
        d = np.zeros(mshape, dtype=np.double)
        k = np.arange(1+i*blk_size, 1+(i+1)*blk_size)
        d = np.sum(iF2Ek(k, buf, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed, fid), axis=0)
        sum_ += d
        ns = np.linalg.norm(sum_) 
        if np.linalg.norm(d) < EPS*ns or ns < TINY :
            return sum_
    return sum_
        
def iF2Ek(k_, buf, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed, fid):
    pi = np.pi
    if "F2E_" + fid not in buf.__dict__.keys():
        setattr(buf, "F2E_" + fid, {})
        argx1 = pi*ksid/ksied
        argx2 = 0.5*pi/ksied*(x2-x1)
        argx3 = 0.5*pi/ksied*(2*ksiwd+x1+x2)
        argy1 = 2*yed-(yd+ywd)
        argy2 = yd+ywd
        argy3 = 2*yed - np.abs(yd-ywd)
        argy4 = np.abs(yd-ywd)
        setattr(buf,"F2E_" + fid, (argx1, argx2, argx3, argy1, argy2, argy3, argy4))
    else:
        argx1, argx2, argx3, argy1, argy2, argy3, argy4 = getattr(buf,"F2E_" + fid)
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

def i1F2H(x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, buf, fid):
    TINY = 1e-20
    mshape = x1.shape
    assert x2.shape == mshape
    sum_ = i1F2Hk(0, 1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, buf, fid)
    sum_ += i1F2Hk(0, -1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, buf, fid)
    EPS = 1e-12
    blk_size = 3
    MAXITER = 100
    for i in range(MAXITER):
        d = np.zeros(mshape, dtype = np.double)
        for k in range(1+i*blk_size, 1+(i+1)*blk_size):
            d+= i1F2Hk(k, 1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, buf, fid)
            d+= i1F2Hk(-k, 1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, buf, fid)
            d+= i1F2Hk(k, -1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, buf, fid)
            d+= i1F2Hk(-k, -1, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, buf, fid)
        sum_ += d
        nrm = np.linalg.norm(sum_)
        if np.linalg.norm(d) < EPS*nrm or nrm < TINY:
            return sum_

def i1F2Hk(k, b, x1, x2, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, buf, fid):
    NDIGITS = 6
    mshape = x1.shape
    assert x2.shape == mshape
    m = np.zeros(mshape, dtype=np.double)
    inds_dyd0 = np.isclose(yd,ywd)
    inds_dyd_nnz = np.logical_not(inds_dyd0)
    su = np.sqrt(u+np.square(a))
    if np.any(inds_dyd0):
        if "yd0s_" + fid not in buf.__dict__.keys():
            setattr(buf, "yd0s_" + fid, {})
        if str(k)+"_"+str(b) not in getattr(buf, "yd0s_" + fid).keys():
            t1 = ksiede/ksied*(ksid[inds_dyd0] + b*ksiwd[inds_dyd0] - 2*k*ksied - x2[inds_dyd0])
            t2 = ksiede/ksied*(ksid[inds_dyd0] + b*ksiwd[inds_dyd0] - 2*k*ksied - x1[inds_dyd0])
            mask1 = np.ones_like(t1) - 2*(t1>0)
            mask2 = np.ones_like(t1) - 2*(t2<0)
            t1 = np.round(np.abs(t1), decimals=NDIGITS).flatten()
            t2 = np.round(np.abs(t2), decimals=NDIGITS).flatten()
            nt = len(t1)
            t = np.append(t1,t2)
            ut, inv_t = np.unique(t, return_inverse=True)
            getattr(buf, "yd0s_" + fid)[str(k)+"_"+str(b)] = (ut, inv_t, nt, mask1, mask2)
        else:
            ut, inv_t, nt, mask1, mask2 = getattr(buf, "yd0s_" + fid)[str(k)+"_"+str(b)]
        uv = ksied/ksiede/su*iti0k0(ut*su)[1]
        v = uv[inv_t]
        v1 = v[:nt]
        v2 = v[nt:]
        m[inds_dyd0] = v1*mask1+v2*mask2
    if np.any(inds_dyd_nnz):
        if "ydnnzs_" + fid not in buf.__dict__.keys():
            setattr(buf, "ydnnzs_" + fid, {})
        if str(k)+"_"+str(b) not in getattr(buf, "ydnnzs_" + fid).keys():
            t1 = ksiede/ksied*(ksid[inds_dyd_nnz] + b*ksiwd[inds_dyd_nnz] - 2*k*ksied - x2[inds_dyd_nnz])
            t2 = ksiede/ksied*(ksid[inds_dyd_nnz] + b*ksiwd[inds_dyd_nnz] - 2*k*ksied - x1[inds_dyd_nnz])
            dyd = np.round(np.abs(yd-ywd)[inds_dyd_nnz], decimals=NDIGITS).flatten()
            t1 = np.round(t1, decimals=NDIGITS).flatten()
            t2 = np.round(t2, decimals=NDIGITS).flatten()
            t = np.vstack([t1,t2]).T
            t = np.sort(t, axis=1)
            t = np.hstack([t,dyd[np.newaxis].T])
            ut, inv_t = np.unique(t, axis=0, return_inverse=True)
            getattr(buf, "ydnnzs_" + fid)[str(k)+"_"+str(b)] = (ut, inv_t)
        else:
            (ut, inv_t) = getattr(buf, "ydnnzs_" + fid)[str(k)+"_"+str(b)]
        uv = ksied/ksiede*qgaus(m_bessk0, ut[:,0], ut[:,1], (ut[:,2],su))
        m[inds_dyd_nnz] = uv[inv_t]
    return m

def ki1(x):
    return 0.5*np.pi - iti0k0(x)[1]

def m_bessk0(x, dyd, su):
    return k0(su*np.sqrt(np.square(x)+np.square(dyd)))

def i2F2H(x1,x2,u,a,yd,ywd):
    su = np.sqrt(u+a*a)
    return -0.5*np.exp(-su*np.abs(yd-ywd))/su*(x2-x1)

########################################################
# ____________________ non-integrals____________________
########################################################

def F1(u, yd, ywd, yed):
    sum_ = 0
    su = np.sqrt(u)
    u_sexp = sexp(su, yed)
    ans = np.exp(-su*(2*yed-(yd+ywd))) + np.exp(-su*(yd+ywd)) + np.exp(-su*(2*yed-np.abs(yd-ywd))) +\
            np.exp(-su*np.abs(yd-ywd))
    ans *= (1+u_sexp)/su
    return ans

def F2(buf, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed, fid):
    return F2E(buf, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed, fid) + F2H(u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)

def F2E(buf, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed, fid):
    # check!
    EPS = 1e-12
    blk_size = 10
    MAXITER = 100
    TINY = 1e-12
    mshape = ksid.shape
    assert ksiwd.shape == yd.shape == ywd.shape == mshape
    sum_ = np.zeros(mshape, dtype=np.double)
    for i in range(1, MAXITER):
        d = np.zeros(mshape, dtype=np.double)
        for k in range(i*blk_size, (i+1)*blk_size):
            d += F2E_k(k, buf, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed, fid)
        sum_ += d
        if np.linalg.norm(d) < EPS*(np.linalg.norm(sum_)+TINY):
            return sum_
    raise RuntimeError("F2E did not converge for u = {}".format(u))
            
def F2E_k(k, buf, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd, yed, fid):
    if "F2E_k"+ fid not in buf.__dict__.keys():
        yd_ywd = np.abs(yd-ywd)
        _yed_yd_ywd = 2*yed - yd_ywd
        yd_p_ywd = yd+ywd
        _yed_yd_p_ywd = 2*yed-yd_p_ywd
        a_ksid = np.pi*ksid/ksied
        a_ksiwd = np.pi*ksiwd/ksied
        setattr(buf,"F2E_k"+ fid,(a_ksid, a_ksiwd, yd_ywd, _yed_yd_ywd, yd_p_ywd, _yed_yd_p_ywd))
    else:
        a_ksid, a_ksiwd, yd_ywd, _yed_yd_ywd, yd_p_ywd, _yed_yd_p_ywd = getattr(buf, "F2E_k"+ fid)
    ek = np.sqrt(u+np.square(k*np.pi/ksiede)+np.square(a))
    ek_sexp = sexp(ek, yed)
    ans = np.cos(k*a_ksid)*np.cos(k*a_ksiwd)/ek
    ans *= np.exp(-ek*_yed_yd_ywd)*(1+ek_sexp)+np.exp(-ek*yd_ywd)*ek_sexp +\
            (np.exp(-ek*_yed_yd_p_ywd)+np.exp(-ek*yd_p_ywd))*(1+ek_sexp)
    return ans

def F2H(u, ksid, ksiwd, ksied, ksiede, a, yd, ywd):
    return ksiede/2/np.pi*F2H1(u, ksid, ksiwd, ksied, ksiede, a, yd, ywd) + F2H2(u, a, yd, ywd)

def F2H1_old(u, ksid, ksiwd, ksied, ksiede, a, yd, ywd):
    TINY = 1e-20
    sum_ = F2H1k(0, 1, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
    sum_ += F2H1k(0, -1, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
    EPS = 1e-12
    blk_size = 3
    MAXITER = 1000
    mshape = ksid.shape
    for i in range(MAXITER):
        d = np.zeros(mshape, dtype = np.double)
        for k in range(1+i*blk_size, 1+(i+1)*blk_size):
            d+= F2H1k(k, 1, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
            d+= F2H1k(-k, 1, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
            d+= F2H1k(k, -1, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
            d+= F2H1k(-k, -1, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd)
        sum_ += d
        nrm = np.linalg.norm(sum_)
        if np.linalg.norm(d) < EPS*nrm or nrm < TINY:
            return sum_
    raise RuntimeError("F2H1 did not converge for u = {}".format(u))
    
def F2H1(u, ksid, ksiwd, ksied, ksiede, a, yd, ywd):
    val = np.zeros_like(ksid)
    for sx in [-1,1]:
        val += k0(np.sqrt((u+a*a)*(np.square(ksiede/ksied*(ksid+sx*ksiwd))+np.square(yd-ywd))))
    for sx in [-1,1]:
        for sk in [-1,1]:
            vfunc = lambda k: k0(np.sqrt((u+a*a)*(np.square(ksiede/ksied*(ksid+sx*ksiwd+sk*2*k*ksied))+np.square(yd-ywd))))
            val += eulsum_v(vfunc, ksid.shape)
    return val
        
def F2H1k(k, b, u, ksid, ksiwd, ksied, ksiede, a, yd, ywd):
    arg = np.sqrt(u+a*a)
    arg *= np.sqrt(np.square(ksiede/ksied*(ksid+b*ksiwd-2*k*ksied))+np.square(yd-ywd))
    return k0(arg)

def F2H2(u, a, yd, ywd):
    return -0.5*np.exp(-np.sqrt(u+a*a)*np.abs(yd-ywd))/np.sqrt(u+a*a)
