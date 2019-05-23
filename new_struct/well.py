from .helper import calc_stehf_coef, lapl_invert
from .lapl_well import LaplWell

class Well():
    def __init__(self, outer_bound, top_bound, bottom_bound,
        wtype, nseg, nwells, xwds, ywds, x_lengths = 1,
        xed = 0, yed = 0, zwds = None, hd = 0, attrs = {}):
        NSTEPS = 10
        self.stehf_coefs = calc_stehf_coef(NSTEPS)
        self.lapl_well = LaplWell(outer_bound, top_bound, bottom_bound, wtype, nseg, nwells, xwds, ywds,
            x_lengths = x_lengths, xed = xed, yed = yed, zwds = zwds, hd = hd, attrs = attrs)

    def pw(self, t):
        f = self.lapl_well.pw
        return lapl_invert(f, t, self.stehf_coefs)

