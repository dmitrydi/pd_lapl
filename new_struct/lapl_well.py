import numpy as np
from .sources import Sources
from .matrixizer import Matrixizer
from .integrator import *
from .helper import make_offset

class LaplWell():
    """class for lapl well"""
    def __init__(self, outer_bound, top_bound, bottom_bound,
        wtype, nseg, nwells, xwds, ywds, x_lengths = 1,
        xed = 0, yed = 0, zwds = None, hd = 0, attrs = {}):
    # attrs keeps Fcd etc.
        # check initialization
        if outer_bound != "inf":
            assert xed != 0
            assert yed != 0
        if wtype == "horiz":
            assert zwds is not None
            assert hd is not None
        assert nwells >= 1
        if nwells > 1:
            assert len(xwds) == len(ywds) == nwells
            if wtype == "horiz":
                assert len(zwds) == nwells
        ##
        self.wtype = wtype
        self.nwells = nwells
        self.xwds = np.array([xwds]).flatten()
        self.ywds = np.array([ywds]).flatten()
        if zwds is None:
            self.zwds = np.zeros_like(self.xwds, dtype = np.float)
        else:
            self.zwds = np.array(zwds)
        self.sources_ = {"sources_list": [],
            "boundaries": {"outer_bound": outer_bound, "top_bound": top_bound, "bottom_bound": bottom_bound},
            "wtype": wtype,
            "mx": None,
            "attrs": attrs}
        for xwd, ywd, zwd in zip(self.xwds, self.ywds, self.zwds):
            self.sources_["sources_list"].append(Sources(xwd, ywd, zwd,
                                                        xed, yed, hd,
                                                        wtype,
                                                        x_lengths,
                                                        outer_bound,
                                                        top_bound,
                                                        bottom_bound,
                                                        nseg))
        self.matrixizer = Matrixizer(self.sources_)
        self.last_s = -1

    def recalc(self, s):
        u = s #!
        green_matrix_ = integrate_sources_for_green_matrix(u, self.matrixizer, self.sources_)
        self.green_matrix = make_offset(green_matrix_)
        solution = np.linalg.solve(self.matrixizer.dummy_m + self.matrixizer.source_m - self.green_matrix, self.matrixizer.dummy_rhv/u)
        self.pw_lapl = solution[0]
        self.qw_lapl = 1/s/s/self.pw_lapl
        self.Q_lapl = self.qw_lapl/s
        self.q_distrib  = solution[1:]

    def pw(self, s):
        if s != self.last_s:
            self.recalc(s)
        return self.pw_lapl

    def pxy(self, s, xd, yd, cache=False):
        if s != self.last_s:
            self.recalc(s)
        if str(xd)+"_"+str(yd) not in self.matrixizer.v_cache:
            self.matrixizer.v_cache[str(xd)+"_"+str(yd)] = {"dyds_0": {}, "dyds_nnz": {}}
        v = integrate_matrix_for_xd_yd(s, xd, yd, self.matrixizer.v_cache[str(xd)+"_"+str(yd)], self.matrixizer, self.sources_)
        if not cache:
            del self.matrixizer.v_cache[str(xd)+"_"+str(yd)]
        return np.sum(v*self.q_distrib)





        
        
        