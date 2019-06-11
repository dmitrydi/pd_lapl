from .ffuncs import *
import numpy as np

def matr_pd_frac_nnnn(u, x1, x2, xd, xwd, xed, yd, ywd, yed, buf, id1, id2):
    pi = np.pi
    ans = pi/xed*(0.5*iF1(x1, x2, u, yd, ywd, yed)+\
                  iF2E(x1, x2, u, xd, xwd, xed, xed, 0, yd, ywd, yed, buf, id1)+\
                  0.5*xed/pi*i1F2H(x1, x2, u, xd, xwd, xed, xed, 0, yd, ywd, buf, id2)+\
                  i2F2H(x1,x2,u,0,yd,ywd))
    return ans