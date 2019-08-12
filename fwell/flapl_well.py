import numpy as np
from .ffunc.frac import matr_pd_frac_nnnn
from .helper import make_offset
from .mkcoord import *
from .buffer import Buffer

class FLaplWell():
    """class for lapl well"""
    def __init__(self, outer_bound, top_bound, bottom_bound,
        wtype, N, nwells, xwds, ywds, x_lengths = 1,
        xed = 0, yed = 0, zwds = None, hd = 0, rwd = 0, attrs = {}):
        self.xed = xed
        self.yed = yed
        self.wtype = wtype
        self.N = N
        self.nwells = nwells
        self.xd, self.xwd, self.yd, self.ywd, self.x1, self.x2 = make_coordinates_(xwds, ywds, self.N, self.nwells)
        if wtype == "hor":
            raise NotImplementedError
            self.zwd = zwds[0]*np.ones_like(self.xwd)
            self.zd = self.zwd+rwd
            self.hd = hd
        self.dum = make_dummy_matr_(self.N, self.nwells)
        self.src = make_source_matr_(self.N, self.nwells, attrs["Fcd"], self.wtype)
        self.rhv = make_dummy_rhv_(self.N, self.nwells, attrs["Fcd"], self.wtype)
        self.buf = Buffer()  
        
    def recalc(self, s):
        u = s
        self.buf.ss = s
        if self.wtype == "frac":
            self.grm = matr_pd_frac_nnnn(u, self.x1, self.x2, self.xd, self.xwd, self.xed, self.yd, self.ywd, self.yed,
                                         self.buf, "frac_1", "frac_1")
        elif self.wtype == "hor":
            self.grm = matr_pd_hor_nnnn(u, self.x1, self.x2, self.zd, self.zwd, self.hd, self.xd, self.xwd, self.xed,
                                        self.yd, self.ywd, self.yed, self.buf)
        grm_ = make_offset(self.grm)
        solution = np.linalg.solve(self.dum + self.src - grm_, self.rhv/u)
        self.pw_lapl = solution[0]
        self.qw_lapl = 1/s/s/self.pw_lapl
        self.Q_lapl = self.qw_lapl/s
        self.q_distrib  = solution[1:]
        
    def pw(self, s):
        self.recalc(s)
        return self.pw_lapl
