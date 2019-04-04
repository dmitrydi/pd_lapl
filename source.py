from scipy.special import kn, k0, k1, iti0k0
from source_helper import *

class LaplSource():
# class for a finite element (finite liquid source) of a well or fracture
# all in laplace space
	def __init__(self,
		outer_bound,
		top_bound,
		bottom_bound,
		wtype,
		x_size,
		params={}):
		self.q = None
		self.outer_bound = outer_bound
		self.top_bound = top_bound
		self.bottom_bound = bottom_bound
		self.wtype = wtype
		self.x_size = x_size # dimentionless length of a segment
		self.params = params
		self.k = (self.params["kx"]*self.params["ky"]*self.params["kz"])**(1/3)
		self.c = (self.k/self.params["kx"])**0.5

	def f_s(self, s):
		w = self.params["omega"]
		l = self.params["lambda"]
		return (s*w*(1-w)+l)/(s*(1-w)+l)

	def Green(self, s, xd, yd, zd, xwd, ywd, zwd, int_lim1, int_lim2):
		# Green's function of the source of well located in point (xwd, ywd, zwd) (response in coordinate point (xd, yd, zd)), depends on type and boundaries
		# int_lim1, int_lim2 - ends of a source along x-axis
		u = s*self.f_s(s)
		su = u**0.5
		c = self.c
		if self.wtype == "frac":
			if self.outer_bound == "infinite":
				if self.top_bound == "imp":
					if self.bottom_bound == "imp":
						assert int_lim1 <= int_lim2
						return frac_inf_imp_imp(xd, yd, xwd, ywd, su, c, int_lim1, int_lim2)
						'''
						if yd == ywd:
							if (xd - xwd - c*int_lim2) >=0:
								return 1/c/su*(iti0k0(su*(xd-xwd-c*int_lim1))[1] - iti0k0(su*(xd-xwd-c*int_lim2))[1])
							elif (xd - xwd - c*int_lim1) <=0:
								return 1/c/su*(iti0k0(su*(c*int_lim2+xwd-xd))[1] - iti0k0(su*(c*int_lim1+xwd-xd))[1])
							else:
								return 1/c/su*(iti0k0(su*(xd-xwd-c*int_lim1))[1] + iti0k0(su*(c*int_lim2+xwd-xd))[1])
						else:
							return quad(frac_inf_imp_imp, int_lim1, int_lim2, args = (xd, xwd, yd, ywd, c, su))[0]
						'''
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




