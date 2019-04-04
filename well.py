from mpmath import invertlaplace
from lapl_well import LaplWell
from source_helper import nlinvsteh

class Well():
	# class for describing a well
	# in real space
	def __init__(self, xw, yw,
		outer_bound,
		top_bound,
		bottom_bound,
		wtype,
		params, 
		zw = 0):
		self.params = params
		self.xwd = xw/self.params["ref_length"]
		self.ywd = yw/self.params["ref_length"]
		self.zwd = zw/self.params["z_ref_length"]
		self.outer_bound = outer_bound
		self.top_bound = top_bound
		self.bottom_bound = bottom_bound
		self.wtype = wtype
		self.lapl_well = LaplWell(
			self.xwd,
			self.ywd,
			self.zwd,
			self.outer_bound,
			self.top_bound,
			self.bottom_bound,
			self.wtype,
			self.params)

	def fp(self, p):
		self.lapl_well.recalc(p)
		return self.lapl_well.p_lapl

	def get_pw(self, t):
		return nlinvsteh(self.fp, t, n = 8)

	def get_q(self, t, Pwf):
		# returns well rate Q at time t given the bottomhole pressure is Pwf
		pass

	def print_source_distribution(self, t):
		pass

