from multifrac_lapl import MultifracLapl
import numpy as np
from helper import Helper

class Multifrac():
	def __init__(self, nwells, xws, yws, outer_bound, top_bound, bottom_bound, params, n_stehf, int_type="quad", npoints=5):
		if (len(xws) != len(yws)) or (len(xws) != nwells):
			raise AttributeError("wrong init data")
		else:
			self.nwells = nwells
		self.params = params
		self.xwds = xws/self.params["ref_length"]
		self.ywds = yws/self.params["ref_length"]
		self.lapl_well = MultifracLapl(nwells, xws, yws, outer_bound, top_bound, bottom_bound, params, n_stehf, int_type, npoints)
		self.outer_bound = outer_bound
		self.top_bound = top_bound
		self.bottom_bound = bottom_bound
		self.n_stehf = n_stehf
		self.h = Helper()
		self.v = self.h.calc_stehf_coef(self.n_stehf)

	def get_pw(self, t):
		f = lambda p: self.lapl_well.pw_lapl(p)
		return self.h.lapl_invert(f, t, self.v)

	def get_source_distribution(self, t):
		def _f(p):
			self.lapl_well.recalc(p)
			return self.lapl_well.source_distrib
		return self.h.lapl_invert(_f, t, self.v)





