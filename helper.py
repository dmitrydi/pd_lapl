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

	def get_green_matrix(self, well, s):
		if well.wtype == "frac":
			unique_integrals = well.source.calc_source_funcs(s)
			green_matrix = self.combine_green_matrix(well, unique_integrals)
		else:
			raise NotImplementedError
		return green_matrix

	def get_green_matrix_general(self, wells, s, outer_bound):
		if outer_bound == "infinite":
			self_matrix_ = self.get_green_matrix(wells[0], s)
			self_matrix = self_matrix_[:2*N, 1:]
			gdict = dict()
			for i in range(nwells):
				this_well = wells["well_"+str(i)]
				for j in range(nwells):
					if i != j:
						that_well = wells["well_"+str(j)]
						this_well.gk.make_mutual_geometry(gdict, that_well)
						lims1 = gdict["lims1"]
						lims2 = gdict["lims2"]
						mxd = gdict["mxd"]
						myd = gdict["myd"]


		else:
			raise NotImplementedError

	def combine_green_matrix(self, well, uints):
		# takes unique integrals of segments and make adding and substraction of values using gk.masks
		if well.wtype == "frac":
			N = well.params["nseg"]
			m = np.zeros((1+2*N, 1+2*N))
			i1 = uints[np.argwhere(well.gk.sg["ulims"] == well.gk.sg["alims1"].reshape(4*N*N,1))[:,1]].reshape(2*N,2*N)
			i2 = uints[np.argwhere(well.gk.sg["ulims"] == well.gk.sg["alims2"].reshape(4*N*N,1))[:,1]].reshape(2*N,2*N)
			integrals = np.multiply(i1, well.gk.sg["mask_int_1"]) + np.multiply(i2, well.gk.sg["mask_int_2"])
		else:
			raise NotImplementedError
		m[:2*N, 1:] = integrals
		return m

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

	def calc_stehf_coef(self, n):
		v = np.array(range(n+1), dtype = np.float)
		g = np.array(range(161), dtype = np.float)
		h = np.array(range(81), dtype = np.float)
		g[1] = 1
		NH = n // 2
		for i in range(2, n+1):
			g[i] = g[i-1]*i
		h[1] = 2./ g[NH - 1]
		for i in range(2, NH+1):
			fi = i
			if i != NH:
				h[i] = (fi ** NH)*g[2*i]/(g[NH - i]*g[i]*g[i - 1])
			else:
				h[i] = (fi ** NH)*g[2*i]/(g[i]*g[i - 1])
		SN = 2 * (NH - (NH // 2)*2) - 1
		for i in range(1, n+1):
			v[i] = 0.
			K1 = (i + 1) // 2
			K2 = i
			if K2 > NH:
				K2 = NH
			for k in range(K1, K2 + 1):
				if 2*k - i == 0:
					v[i] += h[k]/g[i - k]
				elif i == k:
					v[i] += h[k]/g[2*k - i]
				else:
					v[i] += h[k]/(g[i - k]*g[2*k - i])
			v[i] *= SN
			SN = -1*SN
		return v

	def lapl_invert(self, f, t, stehf_coefs):
		ans = 0.
		N = len(stehf_coefs)
		for i in range(1, N):
			p = i * np.log(2.)/t
			f_ = f(p)
			ans += f_*stehf_coefs[i]*p/i
		return ans


'''
NOT USED

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
'''

	
	

