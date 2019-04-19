from scipy.special import kn, k0, k1, iti0k0
from source_helper import *

class LaplSource():
# class for a finite element (finite liquid source) of a well or fracture
# all in laplace space
	def __init__(self,
		well):
		self.well = well

	def f_s(self, s):
		w = self.well.params["omega"]
		l = self.well.params["lambda"]
		return (s*w*(1-w)+l)/(s*(1-w)+l)

	def calc_source_funcs(self, s):
		if self.well.wtype == "frac":
			fs = self.f_s(s)
			su = (s*fs)**0.5
			c = self.well.c
			uvals = 0.5/c/su*iti0k0(su*self.well.gk.ulims)[1]
		else:
			raise NotImplementedError
		return uvals

	def Green(self, s, xd, yd, zd, xwd, ywd, zwd, int_lim1, int_lim2):
		# Green's function of the source of well located in point (xwd, ywd, zwd) (response in coordinate point (xd, yd, zd)), depends on type and boundaries
		# int_lim1, int_lim2 - ends of a source along x-axis
		u = s*self.f_s(s)
		su = u**0.5
		c = self.well.c
		if self.well.wtype == "frac":
			if self.well.outer_bound == "infinite":
				if self.well.top_bound == "imp":
					if self.well.bottom_bound == "imp":
						assert int_lim1 <= int_lim2
						return frac_inf_imp_imp(xd, yd, xwd, ywd, su, c, int_lim1, int_lim2)
					else:
						raise NotImplementedError
				else:
					raise NotImplementedError
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError

	def pd(u, xd, yd, zd, xcd, ycd, zcd):
		# pressure response in space point (xd, yd, zd)
		# works only if source strength is calculated
		if self.q is None:
			raise ValueError("Source strength q in unindentified")
		pass




