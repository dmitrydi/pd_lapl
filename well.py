from mpmath import invertlaplace
from lapl_well import LaplWell
from source_helper import calc_stehf_coef, lapl_invert
import numpy as np

class Well():
	# class for describing a well
	# in real space
	def __init__(self, xw, yw,
		outer_bound,
		top_bound,
		bottom_bound,
		wtype,
		params,
		n_stehf, 
		zw = 0):
		self.params = params
		self.xwd = xw/self.params["ref_length"]
		self.ywd = yw/self.params["ref_length"]
		self.zwd = zw/self.params["z_ref_length"]
		self.outer_bound = outer_bound
		self.top_bound = top_bound
		self.bottom_bound = bottom_bound
		self.wtype = wtype
		self.n_stehf = n_stehf
		self.v = calc_stehf_coef(self.n_stehf)
		self.lapl_well = LaplWell(
			self.xwd,
			self.ywd,
			self.zwd,
			self.outer_bound,
			self.top_bound,
			self.bottom_bound,
			self.wtype,
			self.params)

	def get_pw(self, t):
		f = lambda p: self.lapl_well.p_lapl_xy(p, self.xwd, self.ywd, self.zwd)
		return lapl_invert(f, t, self.v)

	def get_p_xd_yd(self, t, xd, yd, zd=0):
		f = lambda p: self.lapl_well.p_lapl_xy(p, xd, yd, zd)
		return lapl_invert(f, t, self.v)

	def get_q(self, t, Pwf):
		# returns well rate Q at time t given the bottomhole pressure is Pwf
		pass

	def print_source_distribution(self, t):
		pass

