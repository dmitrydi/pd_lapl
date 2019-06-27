import numpy as np

def eulsum(vfunc, j_0=False):
    # euler's algo for series summation 
    # vfunc - scalar-valued lambda-like function function
    # param: j_0 - whether to account for 0-th term in sum
    MAXTERM = 1000
    EPS = 1e-9
    TINY = 1e-50
    wksp = np.zeros(MAXTERM, dtype=np.float)
    nterm = 0
    sign = 1.
    ppsum = 0. #previous sum
    # account for 0-th term
    if j_0:
        psum = vfunc(0)
    else:
        psum = 0
    for j in range(1, MAXTERM):
        term = sign*wtrans(vfunc, j, verbose=0) # here we continue summation with n+1-th term
        sign *= -1.
        psum, wksp, nterm = eulsum_(psum, term, j, wksp, nterm) # but here we call eulsum_ with j-n jterm
        apsum = np.abs(psum)
        if np.abs(psum-ppsum) < EPS*apsum or apsum < TINY:
            return psum
        ppsum = psum.copy()
    raise RuntimeWarning("In eulsum: series did not converge in {} iterations".format(MAXTERM))
    
def eulsum_(psum, term, jterm, wksp, nterm):
    if jterm == 1:
        nterm = 1
        wksp[1] = term
        psum = 0.5*wksp[1]
    else:
        tmp = wksp[1]
        wksp[1] = term
        for j in range(1, nterm):
            dum = wksp[j+1]
            wksp[j+1] = 0.5*(wksp[j]+tmp)
            tmp=dum
            wksp[nterm+1] = 0.5*(wksp[nterm]+tmp)
            tmp=dum
        wksp[nterm+1] = 0.5*(wksp[nterm]+tmp)
        if (np.abs(wksp[nterm+1]) <= np.abs(wksp[nterm])):
            nterm += 1
            psum += (0.5*wksp[nterm])
        else:
            psum += wksp[nterm+1]
    return psum, wksp, nterm

def wtrans(vfunc, r, verbose=0):
    # returns VanWijngaarden - transformation terms w_r:
    # sum[v_r, r=1..oo] = sum[(-1)**(r-1)*w_r, r=1..oo]
    # vfunc - lambda-like function to eval with integer r
    # returns w_r
    MAXIT = 10000
    TINY = 1e-30
    EPS = 1e-12
    psum = vfunc(r)
    cpow = 2.
    for j in range(1,MAXIT):
        d = cpow*vfunc(cpow*r)
        psum += d
        cpow *= 2.
        ad = np.abs(d)
        if ad < np.abs(psum)*EPS or ad < TINY:
            if verbose==0:
                return psum
            elif verbose == 1:
                return psum, j
            elif verbose >= 2:
                return psum, j, d
    raise RuntimeError

#############vector functions###################
def wtrans_v(vfunc, r, verbose=0):
    # returns VanWijngaarden - transformation terms w_r:
    # sum[v_r, r=1..oo] = sum[(-1)**(r-1)*w_r, r=1..oo]
    # vfunc - lambda-like function to eval with integer r
    # returns w_r
    MAXIT = 10000
    TINY = 1e-30
    EPS = 1e-12
    psum = vfunc(r)
    cpow = 2.
    for j in range(1,MAXIT):
        d = cpow*vfunc(cpow*r)
        psum += d
        cpow *= 2.
        ad = np.max(np.abs(d))
        if ad < (np.min(np.abs(psum))+TINY)*EPS or ad < TINY:
            if verbose==0:
                return psum
            elif verbose == 1:
                return psum, j
            elif verbose >= 2:
                return psum, j, d
    raise RuntimeError("{}".format(r))
    
def eulsum_v_(psum, term, jterm, wksp, nterm):
    if jterm == 1:
        nterm = 1
        wksp[1] = term.copy()
        psum = 0.5*wksp[1]
    else:
        tmp = wksp[1].copy()
        wksp[1] = term.copy()
        for j in range(1, nterm):
            dum = wksp[j+1].copy()
            wksp[j+1] = 0.5*(wksp[j]+tmp)
            tmp=dum.copy()
            wksp[nterm+1] = 0.5*(wksp[nterm]+tmp)
            tmp=dum.copy()
        wksp[nterm+1] = 0.5*(wksp[nterm]+tmp)
        if np.max(np.abs(wksp[nterm+1])) <= np.max(np.abs(wksp[nterm])):
            nterm += 1
            psum += (0.5*wksp[nterm])
        else:
            psum += wksp[nterm+1]
    return psum, wksp, nterm

def eulsum_v(vfunc, vfunc_shape, j_0=False):
    # euler's algo for series summation 
    # vfunc - scalar-valued lambda-like function function
    # param: j_0 - whether to account for 0-th term in sum
    MAXTERM = 1000
    EPS = 1e-9
    TINY = 1e-20
    wksp = np.zeros((MAXTERM, *vfunc_shape), dtype=np.float)
    nterm = 0
    sign = 1.
    ppsum = np.zeros(vfunc_shape)
    # account for 0-th term
    if j_0:
        psum = vfunc(0)
    else:
        psum = np.zeros(vfunc_shape)
    for j in range(1, MAXTERM):
        term = sign*wtrans_v(vfunc, j, verbose=0) 
        sign *= -1.
        psum, wksp, nterm = eulsum_v_(psum, term, j, wksp, nterm) 
        apsum = np.min(np.abs(psum))+0.5*TINY
        if np.max(np.abs(psum-ppsum)) < EPS*apsum or apsum < TINY:
            return psum
        ppsum = psum.copy()
    raise RuntimeWarning("In eulsum: series did not converge in {} iterations".format(MAXTERM))