from mpmath import invertlaplace
from lapl_well import LaplWell
from source_helper import calc_stehf_coef
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

	def fp(self, p, mode):
		self.lapl_well.recalc(p, mode)
		return self.lapl_well.p_lapl

	def get_pw(self, t, mode = "old"):
		ans = 0.
		for i in range(1, self.n_stehf+1):
			p = i * np.log(2.)/t
			ans += self.fp(p, mode)*self.v[i]*p/i
		return ans

	def get_p_xd_yd(self, t, xd, yd, mode = "new"):
		ans = 0.
		for i in range(1, self.n_stehf+1):
			p = i * np.log(2.)/t
			ans += self.lapl_well.p_lapl_xy(s, xd, yd, mode)*self.v[i]*p/i
		return ans

	def get_q(self, t, Pwf):
		# returns well rate Q at time t given the bottomhole pressure is Pwf
		pass

	def print_source_distribution(self, t):
		pass

