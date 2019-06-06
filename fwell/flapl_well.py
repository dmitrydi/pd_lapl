import numpy as np
from .ffunc.frac import matr_pd_frac_nnnn
from .helper import make_offset
from .mkcoord import *

class FLaplWell():
    """class for lapl well"""
    def __init__(self, outer_bound, top_bound, bottom_bound,
        wtype, N, nwells, xwds, ywds, x_lengths = 1,
        xed = 0, yed = 0, zwds = None, hd = 0, attrs = {}):
        self.xed = xed
        self.yed = yed
        self.wtype = wtype
        self.N = N
        self.nwells = nwells
        self.xd, self.xwd, self.yd, self.ywd, self.x1, self.x2 = make_coordinates_(xwds, ywds, self.N, self.nwells)
        self.dum = make_dummy_matr_(self.N, self.nwells)
        self.src = make_source_matr_(self.N, self.nwells, attrs["Fcd"], self.wtype)
        self.rhv = make_dummy_rhv_(self.N, self.nwells, attrs["Fcd"], self.wtype)
        self.last_s = -1
        
    def recalc(self, s):
        u = s
        if self.wtype == "frac":
            self.grm = matr_pd_frac_nnnn(u, self.x1, self.x2, self.xd, self.xwd, self.xed, self.yd, self.ywd, self.yed)
        grm_ = make_offset(self.grm)
        solution = np.linalg.solve(self.dum + self.src - grm_, self.rhv/u)
        self.pw_lapl = solution[0]
        self.qw_lapl = 1/s/s/self.pw_lapl
        self.Q_lapl = self.qw_lapl/s
        self.q_distrib  = solution[1:]
        
    def pw(self, s):
        if s != self.last_s:
            self.last_s = s
            self.recalc(s)
        return self.pw_lapl
