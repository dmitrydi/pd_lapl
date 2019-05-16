from source import LaplSource
from helper import Helper
from old_helper import OldHelper
import numpy as np
from geometry_keeper import GeometryKeeper
from scipy.integrate import romberg, quad
from scipy.interpolate import InterpolatedUnivariateSpline

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
		g = GeometryKeeper()
		self.gk = g.make_fracture_geometry(self, self.xwd, self.ywd)
		self.source = LaplSource(self)

	def recalc(self, s):
		# recalculates source distribution in the well for Laplace parameter s
		# param 'mode' sets wheter 2*n_seg+1 matrix, or n_seg+1 matrix for source distribution calculation
		self.last_s = s # keeps the last s of recalc of the well, for speed
		#helper = Helper()
		dummy_matrix = self.get_dummy_matrix()
		green_matrix = self.get_green_matrix(s)
		e_green_matrix = self.ext_green_matrix(green_matrix)
		source_matrix = self.get_source_matrix(s)
		right_part = self.get_right_part(s)
		solution = np.linalg.solve(dummy_matrix - e_green_matrix + source_matrix, right_part)
		self.p_lapl = solution[0]
		self.source_distrib = solution[1:]
		self.smoothed_distrib = InterpolatedUnivariateSpline(np.array(self.gk["xcs"])[0], self.source_distrib, k=4)
		self.q_lapl = 1./s/s/self.p_lapl
		self.Q_lapl = self.q_lapl/s

	def p_lapl_xy(self, s, xd, yd, zd):
		# calculates dimentionless pressure in Laplace space at point (xd, yd, zd)
		if s != self.last_s:
			self.recalc(s) # if source distribution for parameter s is unknown then recalc
		if xd == self.xwd and yd == self.ywd and zd == self.zwd:
			return self.p_lapl # return bottomhole pressure
		# sources = self.source_distrib
		# xcs = np.array(self.gk.xcs)[0]
		# fi = InterpolatedUnivariateSpline(xcs, sources, k=4)
		if self.wtype == "frac":
			g = lambda x: self.smoothed_distrib(x)*self.source.frac_source_function(x, s, xd, yd, self.xwd, self.ywd, self.c)
			if yd != self.ywd or xd == self.xwd:
				return romberg(g, self.xwd-1, self.xwd+1, rtol=1e-6)
			else:
				return quad(g, self.xwd-1, abs(xd-self.xwd), epsrel = 1e-6)[0] + quad(g, abs(xd-self.xwd), self.xwd+1, epsrel = 1e-6)[0]

	def get_dummy_matrix(self):
		N = 1+ 2*self.params["nseg"]
		m = np.zeros((N, N))
		m[:(N-1), 0] = 1
		m[-1, 1:] = 1
		return m

	def get_green_matrix(self, s):
		if self.wtype == "frac":
			#unique_integrals = self.source.calc_source_funcs(s)
			if self.outer_bound == "infinite":
				unique_integrals = self.source.integrate_source_functions(s,
					np.zeros_like(self.gk["ulims"], dtype=np.float),
					self.gk["ulims"],
					np.zeros_like(self.gk["ulims"], dtype=np.float))
				green_matrix = self.combine_green_matrix(unique_integrals)
			elif self.outer_bound == "nnnn":
				green_matrix = self.source.integrate_source_functions_bounded(s,
					self.gk["mxd"],
					self.gk["mxj"],
					self.gk["mxj1"],
					self.params["xed"],
					self.gk["myd_"],
					self.ywd,
					self.params["yed"])
		else:
			raise NotImplementedError
		return green_matrix

	def ext_green_matrix(self, green_matrix):
		N = self.params["nseg"]
		m = np.zeros((1+2*N, 1+2*N))
		m[:2*N, 1:] = green_matrix
		return m

	def combine_green_matrix(self, uints):
		# takes unique integrals of segments and make adding and substraction of values using gk.masks
		if self.wtype == "frac":
			N = self.params["nseg"]
			m = np.zeros((1+2*N, 1+2*N))
			i1 = uints[np.argwhere(self.gk["ulims"] == self.gk["alims1"].reshape(4*N*N,1))[:,1]].reshape(2*N,2*N)
			i2 = uints[np.argwhere(self.gk["ulims"] == self.gk["alims2"].reshape(4*N*N,1))[:,1]].reshape(2*N,2*N)
			integrals = np.multiply(i1, self.gk["mask_int_1"]) + np.multiply(i2, self.gk["mask_int_2"])
		else:
			raise NotImplementedError
		return integrals

	def get_source_matrix(self, s):
		N = self.params["nseg"]
		m = np.zeros((1+2*N, 1+2*N))
		m_ = np.zeros((N,N))
		dx = 1./N
		dx2_8 = 0.125*dx**2
		dx2_2 = 0.5*dx**2
		if self.wtype == "frac":
			Fcd = self.params["Fcd"]
			coef = np.pi/Fcd
			for j in range(1, N+1):
				m_[j-1, j-1] = dx2_8
				xj = dx*(j-0.5)
				for i in range(1, j):
					m_[j-1, i-1] = dx2_2 + dx*(xj - i*dx)
		else:
			raise NotImplementedError
		m[:N, 1:N+1] = np.flip(np.flip(m_, 0), 1)
		m[N:2*N, N+1:] = m_
		m *= coef
		return m

	def get_right_part(self, s):
		N = self.params["nseg"]
		dx = 1./N
		Fcd = self.params["Fcd"]
		coef = np.pi * dx/Fcd/s
		b = np.zeros((1+2*N))
		# important here should be 2
		b[-1] = 2./s/dx
		for i in range(N):
			b[N+i] = coef*(i+0.5)
			b[N-i-1] = coef*(i+0.5)
		return b




