from .ffuncs import *
import numpy as np

def matr_pd_frac_nnnn(u, x1, x2, xd, xwd, xed, yd, ywd, yed):
    pi = np.pi
    buf_if2e = None
    buf_yd0, buf_yd_nnz = {}, {}
    ans = pi/xed*(0.5*iF1(x1, x2, u, yd, ywd, yed)+\
                  iF2E(x1, x2, u, xd, xwd, xed, xed, 0, yd, ywd, yed)+\
                  0.5*xed/pi*i1F2H(buf_yd0, buf_yd_nnz, x1, x2, u, xd, xwd, xed, xed, 0, yd, ywd)+\
                  i2F2H(x1,x2,u,0,yd,ywd))
    return ans