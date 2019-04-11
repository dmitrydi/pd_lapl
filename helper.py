import numpy as np

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
				m[i+N,1+j+N] = well.source.Green(s, xd, yd, zd, well.xwd, well.ywd, well.zwd, int_lim1, int_lim2)
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
			v[j+N] = well.source.Green(s, xd, yd, zd, well.xwd, well.ywd, well.zwd, int_lim1, int_lim2)
		return v

	def get_source_matrix(self, well, s):
		N = well.params["nseg"]
		m = np.zeros((1+2*N, 1+2*N))
		dx = 1./N
		if well.wtype == "frac":
			Fcd = well.params["Fcd"]
			coef = np.pi * dx* dx/Fcd
			for i in range(N):
				for j in range(i + 1):
					m[N + i, N + j+1] = i - j + 1
					m[N - i - 1, N - j ] = i - j + 1
			m *= coef
			for i in range(0, 2*N):
				m[i, i+1] = 0.125 * coef
		else:
			raise
		return m

	def get_right_part(self, well, s):
		N = well.params["nseg"]
		dx = 1./N
		Fcd = well.params["Fcd"]
		coef = np.pi * dx/Fcd/s
		b = np.zeros((1+2*N))
		b[-1] = 1/s/dx
		for i in range(N):
			b[N+i] = coef*(i+0.5)
			b[N-i-1] = coef*(i+0.5)
		return b
	
	

