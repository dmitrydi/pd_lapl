from multifrac_lapl import MultifracLapl
import numpy as np
from helper import Helper

class Multifrac():
	def __init__(self, nwells, xws, yws, outer_bound, top_bound, bottom_bound, params, n_stehf):
		if (len(xws) != len(yws)) or (len(xws) != nwells):
			raise AttributeError("wrong init data")
		else:
			self.nwells = nwells
		self.params = params
		self.xwds = xws/self.params["ref_length"]
		self.ywds = yws/self.params["ref_length"]
		self.lapl_well = MultifracLapl(nwells, xws, yws, outer_bound, top_bound, bottom_bound, params, n_stehf)
		self.outer_bound = outer_bound
		self.top_bound = top_bound
		self.bottom_bound = bottom_bound
		self.n_stehf = n_stehf
		self.h = Helper()
		self.v = self.h.calc_stehf_coef(self.n_stehf)





