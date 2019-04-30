import numpy as np
from lapl_well import LaplWell

class MultifracLapl():
	def __init__(self, nwells, xws, yws, outer_bound, top_bound, bottom_bound, params, n_stehf):
		self.lapl_wells = dict()
		for i in range(nwells):
			self.lapl_wells["well_"+str(i)] = LaplWell(xws[i], yws[i], 0., outer_bound, top_bound, bottom_bound, "frac", params)
