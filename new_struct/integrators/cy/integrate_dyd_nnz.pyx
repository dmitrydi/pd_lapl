cimport numpy as np
import numpy as np
from scipy.integrate import quad
from libc.math cimport sqrt, log, fabs, exp

ctypedef np.double_t DTYPE_t
DTYPE = np.double

cdef double bessi0(double x):
    cdef double ax, ans, y
    ax = fabs(x)
    if ax < 3.75:
        y = x/3.75
        y*=y
        ans=1.0+y*(3.5156229+y*(3.0899424+y*1.2067492+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2))))
    else:
        y = 3.75/ax
        ans = y*(-0.1647633e-1+y*0.392377e-2)
        ans = 0.916281e-2+y*(-0.2057706e-1+y*(0.2635537e-1+ans))
        ans = 0.225319e-2+y*(-0.157565e-2+y*ans)
        ans = 0.39894228+y*(0.1328592e-1+y*ans)
        ans *= (exp(ax)/sqrt(ax))
    return ans
        
cdef double bessk0(double x):
    cdef double y, ans
    if x <= 2.:
        y = x*x/4.0
        ans=(-log(x/2.0)*bessi0(x))+(-0.57721566+y*(0.42278420+\
            y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2+\
            y*(0.10750e-3+y*0.74e-5))))))
    else:
        y=2./x
        ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1+\
            y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2+\
            y*(-0.251540e-2+y*0.53208e-3))))))
    return ans
        
cdef double c_frac_fun(double x, double dyd, double su):
    return 0.5/su*bessk0(sqrt(x*x + su*su*dyd*dyd))

cdef double ci_frac_nnz(double upper_lim, double dx, double dyd, double su):
    return quad(c_frac_fun, su*(upper_lim - dx), su*upper_lim, args=(dyd, su))[0]

cdef np.ndarray[DTYPE_t, ndim=1] integrate_over_uniques_nnnn(double su, np.ndarray[DTYPE_t, ndim=2] uniques):
    cdef Py_ssize_t I, J, i, j
    I = uniques.shape[0]
    J = uniques.shape[1]
    cdef double upper_lim, dx, dyd
    cdef np.ndarray[DTYPE_t, ndim=1] ans = np.zeros([I], dtype=DTYPE)
    for i in range(I):
        upper_lim = uniques[i,0]
        dx = uniques[i,1]
        dyd = uniques[i,2]
        ans[i] = ci_frac_nnz(upper_lim, dx, dyd, su)
    return ans

def cy_reconstruct_dyd_nnz(double su,
                        np.ndarray[DTYPE_t, ndim=2] m,
                        np.ndarray[int, ndim=1] inds_dyds_nnz,
                        np.ndarray[DTYPE_t, ndim=2] unique_lims_dyds_nnz,
                        np.ndarray[int, ndim=1] inverse_inds_dyd_nnz):
    cdef np.ndarray[DTYPE_t, ndim=1] uvals = integrate_over_uniques_nnnn(su, unique_lims_dyds_nnz)
    cdef np.ndarray[DTYPE_t, ndim=1] m_ = m.reshape(-1)
    cdef int sh0 = m.shape[0]
    cdef int sh1 = m.shape[1]
    m_[inds_dyds_nnz] = uvals[inverse_inds_dyd_nnz]
    return m_.reshape(sh0,sh1)