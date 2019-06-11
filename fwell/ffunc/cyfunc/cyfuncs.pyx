import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, fabs, log
from scipy.special import iti0k0, k0
from scipy.integrate import quad
cimport cython


ctypedef np.float_t DTYPE_t

cdef float bessi0(float x):
    cdef float ax, ans, y
    ax = fabs(x)
    if ax < 3.75:
        y=x/3.75
        y*=y
        ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492\
                +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))))
    else:
        y = 3.75/ax
        ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1\
            +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2\
            +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1\
            +y*0.392377e-2))))))))
    return ans
        
cdef float bessk0(float x):
    cdef float y, ans
    if x <= 2.:
        y=x*x/4.0
        ans=(-log(x/2.0)*bessi0(x))+(-0.57721566+y*(0.42278420\
            +y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2\
            +y*(0.10750e-3+y*0.74e-5))))))  
    else:
        y=2./x
        ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1\
                +y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2\
                +y*(-0.251540e-2+y*0.53208e-3))))))
    return ans

@cython.boundscheck(False)
def mcy_m_bessk0(np.ndarray[DTYPE_t, ndim=1] x, float dyd, float su):
    cdef int imax = x.shape[0]
    cdef int i
    cdef np.ndarray ans = np.zeros([imax], dtype=np.float)
    for i in range(imax):
        ans[i] = cy_m_bessk0(x[i], dyd, su)
    return ans

def cy_m_bessk0(float x, float dyd, float su):
    return bessk0(su*sqrt(x*x+dyd*dyd))