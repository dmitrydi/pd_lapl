from source import LaplSource
from helper import Helper
from old_helper import OldHelper
import numpy as np
from geometry_keeper import GeometryKeeper

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
		self.c = (((self.params["kx"]*self.params["ky"]*self.params["kz"])**(1./3.))/self.params["kx"])**0.5
		self.round_val = 7
		self.source_distrib = None
		self.last_s = -1
		self.p_lapl = None
		self.q_lapl = None
		self.Q_lapl = None
		self.gk = GeometryKeeper(self)
		self.source = LaplSource(self)

	def recalc(self, s):
		# recalculates source distribution in the well for Laplace parameter s
		# param 'mode' sets wheter 2*n_seg+1 matrix, or n_seg+1 matrix for source distribution calculation
		self.last_s = s # keeps the last s of recalc of the well, for speed
		helper = Helper()
		dummy_matrix = helper.get_dummy_matrix(self)
		green_matrix = helper.get_green_matrix(self, s)
		source_matrix = helper.get_source_matrix(self, s)
		right_part = helper.get_right_part(self,s)
		solution = np.linalg.solve(dummy_matrix - green_matrix + source_matrix, right_part)
		self.p_lapl = solution[0]
		self.source_distrib = solution[1:]
		self.q_lapl = 1./s/s/self.p_lapl
		self.Q_lapl = 1./s/self.p_lapl # check!

	def p_lapl_xy(self, s, xd, yd, zd):
		# calculates dimentionless pressure in Laplace space at point (xd, yd, zd)
		helper = Helper()
		if s != self.last_s:
			self.recalc(s) # if source distribution for parameter s is unknown then recalc
		if xd == self.xwd and yd == self.ywd and zd == self.zwd:
			return self.p_lapl # return bottomhole pressure
		sources = self.source_distrib
		green_vector = helper.get_green_vector(self, xd, yd, zd, s)
		#print("sources {}".format(sources))
		#print("green_vector {}".format(green_vector))
		return np.sum(sources*green_vector)



