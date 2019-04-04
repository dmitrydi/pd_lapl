from source import LaplSource
from helper import Helper
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
		self.p_lapl = None
		self.q_lapl = None
		self.Q_lapl = None

	def recalc(self, s):
		helper = Helper()
		dummy_matrix = helper.get_dummy_matrix(self)
		green_matrix = helper.get_green_matrix(self, s)
		source_matrix = helper.get_source_matrix(self, s)
		right_part = helper.get_right_part(self,s)
		solution = np.linalg.solve(dummy_matrix + green_matrix + source_matrix, right_part)
		self.p_lapl = solution[0]
		self.source_distrib = solution[1:]
		self.q_lapl = 1./s/s/self.p_lapl
		self.Q_lapl = 1./s/self.p_lapl # check!


