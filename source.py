from scipy.special import kn, k0, k1, iti0k0
from source_helper import *
from scipy.integrate import quad

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

	def frac_source_function(self, x, s, xd, yd, xwd, ywd, c):
		fs = self.f_s(s)
		su = (s*fs)**0.5
		return 0.5*k0(su*((xd - xwd - c*x)**2 + (yd - ywd)**2)**0.5) #check 0.5!


	def calc_source_funcs(self, s):
		fs = self.f_s(s)
		su = (s*fs)**0.5
		if self.well.wtype == "frac":
			c = self.well.c
			uvals = 0.5/c/su*iti0k0(su*self.well.gk.sg["ulims"])[1]
		else:
			raise NotImplementedError
		return uvals

'''
	def calc_source_xd_yd(self, s, xd, yd):
		fs = self.f_s(s)
		su = (s*fs)**0.5
		c = self.well.c
		dx = 1./self.well.params["nseg"]
		g = lambda x: self.integrand_frac_inf_imp_imp(x, xd, self.well.xwd, yd, self.well.ywd, c, su)
		f = lambda xc: quad(g, xc-0.5*dx, xc+0.5*dx)
		return 1/c/su*f(self.well.gk.xcs) # TBD: 1 vs. 0.5

	def integrand_frac_inf_imp_imp(self, x, xd, xwd, yd, ywd, c, su):
		return k0(su*((xd-xwd-c*x)**2+(yd-ywd)**2)**0.5)

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

'''



