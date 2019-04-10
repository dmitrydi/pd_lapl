from source import LaplSource
from helper import Helper
from old_helper import OldHelper
import numpy as np

class LaplWell():
	# class for defining a well in Laplace space

	def __init__(self,
		xwd, ywd, zwd,
		outer_bound, top_bound, bottom_bound,
		wtype, params):
		self.xwd = xwd
		self.ywd = ywd
		self.zwd = zwd
		self.outer_bound = outer_bound
		self.top_bound = top_bound
		self.bottom_bound = bottom_bound
		self.wtype = wtype
		self.params = params
		self.source = LaplSource(self.outer_bound,
			self.top_bound,
			self.bottom_bound,
			self.wtype,
			self.params)
		self.source_distrib = None
		self.last_s = -1
		self.p_lapl = None
		self.q_lapl = None
		self.Q_lapl = None

	def recalc(self, s, mode="old"):
		# param 'mode' sets wheter 2*n_seg+1 matrix, or n_seg+1 matrix for source distribution calculation
		self.last_s = s
		if mode == "old":
			helper = OldHelper()
		elif mode == "new":
			helper = Helper()
		else:
			raise AttributeError("mode not specified, may be 'old' or 'new'")
		dummy_matrix = helper.get_dummy_matrix(self)
		green_matrix = helper.get_green_matrix(self, s)
		source_matrix = helper.get_source_matrix(self, s)
		right_part = helper.get_right_part(self,s)
		solution = np.linalg.solve(dummy_matrix - green_matrix + source_matrix, right_part)
		self.p_lapl = solution[0]
		self.source_distrib = solution[1:]
		self.q_lapl = 1./s/s/self.p_lapl
		self.Q_lapl = 1./s/self.p_lapl # check!

	def p_lapl_xy(self, s, xd, yd, mode="old"):
		if mode !="new":
			raise AttributeError("mode for p_lapl_xy must be 'new'")
		helper = Helper()
		if s != self.last_s:
			self.recalc(s, mode)
		sources = self.source_distrib
		green_vector = helper.get_green_vector(self, xd, yd, 0, s)
		return np.sum(sources, green_vector)



