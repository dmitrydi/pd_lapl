import numpy as np
from scipy.special import kn, k0, k1, iti0k0

class Helper():
	def __init__(self):
		pass

	def get_dummy_matrix(self, well):
		N = 1+ 2*well.params["nseg"]
		m = np.zeros((N, N))
		m[:(N-1), 0] = 1
		m[-1, 1:] = 1
		return m

	def dev_get_green_matrix(self, well, s):
		if well.wtype == "frac":
			unique_integrals = well.source.calc_source_funcs(s)
			green_matrix = self.combine_green_matrix(well, unique_integrals)
		else:
			raise NotImplementedError
		return green_matrix

	def combine_green_matrix(self, well, uints):
		if well.wtype == "frac":
			N = well.params["nseg"]
			m = np.zeros((1+2*N, 1+2*N))
			i1 = uints[np.argwhere(well.gk.ulims == well.gk.alims1.reshape(4*N*N,1))[:,1]].reshape(2*N,2*N)
			i2 = uints[np.argwhere(well.gk.ulims == well.gk.alims2.reshape(4*N*N,1))[:,1]].reshape(2*N,2*N)
			integrals = np.multiply(i1, well.gk.mask_int_1) + np.multiply(i2, well.gk.mask_int_2)
		else:
			raise NotImplementedError
		m[:2*N, 1:] = integrals
		return m

	def get_green_matrix(self, well, s):
		N = well.params["nseg"] # number of elements in half-length
		m = np.zeros((1+2*N, 1+2*N))
		dx = 1./N
		if well.wtype in ["frac", "vertical"]:
				zd = 0
		for i in range(-N, N):
			xd = well.xwd + dx*(i + 0.5)
			yd = well.ywd
			for j in range(-N, N):
				int_lim1 = well.xwd + j*dx
				int_lim2 = well.xwd + (j + 1)*dx
				# importtant that here should be 0.5
				m[i+N,1+j+N] = 0.5*well.source.Green(s, xd, yd, zd, well.xwd, well.ywd, well.zwd, int_lim1, int_lim2)
		return m

	def get_limits(self, well):
		if well.wtype != "frac":
			raise AttributeError("well type shoudl be 'frac' for use of get_limits()")
		N = well.params["nseg"] # number of elements in half-length
		m = -1*np.ones((2*N, 2*N, 2))
		dx = 1./N
		xwd = well.xwd
		c = well.c
		accu = well.round_val
		for i in range(-N, N):
			xd = xwd + dx*(i + 0.5)
			for j in range(-N, N):
				int_lim1 = xwd + j*dx
				int_lim2 = xwd + (j + 1)*dx
				if (xd - xwd - c*int_lim2) >=0:
					m[i+N,j+N,0] = (xd-xwd-c*int_lim1)
					m[i+N,j+N,1] = (xd-xwd-c*int_lim2)
				elif (xd - xwd - c*int_lim1) <0:
					m[i+N,j+N,0] = (c*int_lim2+xwd-xd)
					m[i+N,j+N,1] = (c*int_lim1+xwd-xd)
				else:
					m[i+N,j+N,0] = (xd-xwd-c*int_lim1)
					m[i+N,j+N,1] = (c*int_lim2+xwd-xd)
		m = np.around(m, decimals=accu)
		um = np.unique(m.flatten())
		return m, um

	def get_integrals(self, well, s, limits):
		fs = well.source.f_s(s)
		u = s*fs
		su = u**0.5
		v = np.zeros(len(limits))
		for i, lim in enumerate(limits):
			v[i] = iti0k0(su*lim)[1]
		return v

	def get_green_matrix_fast(self, well, s):
		N = well.params["nseg"] # number of elements in half-length
		m = np.zeros((1+2*N, 1+2*N))
		dx = 1./N
		limits = self.get_limits(well)
		integrals = self.get_integrals(well, s, limits)
		accu = well.round_val
		fs = well.source.f_s(s)
		u = s*fs
		su = u**0.5
		xwd = well.xwd
		c = well.c
		for i in range(-N, N):
			xd = well.xwd + dx*(i + 0.5)
			yd = well.ywd
			for j in range(-N, N):
				int_lim1 = well.xwd + j*dx
				int_lim2 = well.xwd + (j + 1)*dx
				if (xd - xwd - c*int_lim2) >=0:
					m[i+N,1+j+N] = 0.5/c/su*(integrals[limits==round(xd-xwd-c*int_lim1, accu)] - integrals[limits==round(xd-xwd-c*int_lim2, accu)])[0]
				elif (xd - xwd - c*int_lim1) <0:
					m[i+N,1+j+N] = 0.5/c/su*(integrals[limits==round(c*int_lim2+xwd-xd, accu)] - integrals[limits==round(c*int_lim1+xwd-xd, accu)])[0]
				else:
					m[i+N,1+j+N] = 0.5/c/su*(integrals[limits==round(xd-xwd-c*int_lim1, accu)] + integrals[limits==round(c*int_lim2+xwd-xd, accu)])[0]
		return m

	def get_green_vector(self, well, xd, yd, zd, s):
		N = well.params["nseg"]
		v = np.zeros(2*N)
		dx = 1./N
		if well.wtype in ["frac", "vertical"]:
			zd = 0
		for j in range(-N, N):
			int_lim1 = well.xwd + j*dx
			int_lim2 = well.xwd + (j + 1)*dx
			v[j+N] = 0.5*well.source.Green(s, xd, yd, zd, well.xwd, well.ywd, well.zwd, int_lim1, int_lim2)
		return v

	def get_source_matrix(self, well, s):
		N = well.params["nseg"]
		m = np.zeros((1+2*N, 1+2*N))
		m_ = np.zeros((N,N))
		dx = 1./N
		dx2_8 = 0.125*dx**2
		dx2_2 = 0.5*dx**2
		if well.wtype == "frac":
			Fcd = well.params["Fcd"]
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

	def get_right_part(self, well, s):
		N = well.params["nseg"]
		dx = 1./N
		Fcd = well.params["Fcd"]
		coef = np.pi * dx/Fcd/s
		b = np.zeros((1+2*N))
		# important here should be 2
		b[-1] = 2./s/dx
		for i in range(N):
			b[N+i] = coef*(i+0.5)
			b[N-i-1] = coef*(i+0.5)
		return b

'''
	def get_source_matrix(self, well, s):
		N = well.params["nseg"]
		m = np.zeros((1+2*N, 1+2*N))
		dx = 1./N
		if well.wtype == "frac":
			Fcd = well.params["Fcd"]
			coef = np.pi * dx* dx/Fcd
			#coef = dx* dx/Fcd
			for i in range(N):
				for j in range(i + 1):
					m[N + i, N + j+1] = i - j + 1
					m[N - i - 1, N - j ] = i - j + 1
			m *= coef
			for i in range(0, 2*N):
				m[i, i+1] = 0.25 * coef
		else:
			raise
		return m
'''

	
	

