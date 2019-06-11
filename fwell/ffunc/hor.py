from .ffuncs import *
from .frac import matr_pd_frac_nnnn
import numpy as np

def matr_pd_hor_nnnn(u, x1, x2, zd, zwd, hd, xd, xwd, xed, yd, ywd, yed, buf):
    PI = np.pi
    ans = matr_pd_frac_nnnn(u, x1, x2, xd, xwd, xed, yd, ywd, yed, buf, "hor_1", "hor_1")
    ans += PI/xed*F2(buf, u, zd, zwd, 1, hd, 0, yd, ywd, yed, "hor_2")*(x2-x1)
    ans += 2*PI/xed*ihF2E(zd, zwd, x1, x2, u, xd, xwd, xed, xed, hd, yd, ywd, yed, buf, "hor_3")
    ans -= PI/xed*F2H(u, zd, zwd, 1, hd, 0, yd, ywd)*(x2-x1)
    ans += ih1F2H(zd, zwd, x1, x2, u, xd, xwd, xed, xed, hd, yd, ywd, buf, "hor_1")
    return ans
