import numpy as np
from sources import Sources
from matrixizer import *
from integrator import *

class LaplWell():
	"""class for lapl well"""
	def __init__(self, outer_bound, top_bound, bottom_bound,
		wtype, nseg, nwells, xwds, ywds, x_lengths = 1,
		xed = None, yed = None, zwds = None, hd = None, attrs = {}):
	# attrs keeps Fcd etc.
		# check initialization
		if outer_bound != "inf":
			assert xed is not None
			assert yed is not None
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
		self.sources_["mx"] = make_matrices_for_integration(self.sources_["sources_list"])
		self.dummy_matrix = make_dummy_matrix(self.sources_["sources_list"])
		self.source_matrix = make_source_matrix(self.sources_["sources_list"], self.sources_["wtype"], self.sources_["attrs"])
		self.dummy_rhv  = make_dummy_rhv(self.sources_["sources_list"], self.sources_["wtype"], self.sources_["attrs"])

	def recalc(self, s):
		u = s #!
		green_matrix_ = integrate_sources_for_green_matrix(*self.sources_["mx"],
			self.sources_["boundaries"],
			self.sources_["wtype"])
		green_matrix = make_offset(green_matrix_)
		solution = np.linalg.solve(self.dummy_matrix + self.source_matrix - green_matrix, rhv/u)
		self.pw_lapl = solution[0]
		self.qw_lapl = 1/s/s/self.pw_lapl
		self.Q_lapl = self.qw_lapl/s
		self.q_distrib  = solution[1:]




		
		
		