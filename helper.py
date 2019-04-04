import numpy as np


class Helper():
	def __init__(self):
		pass

	def get_dummy_matrix(self, well):
		N = 1+ 2*well.params["nseg"]
		m = np.zeros((N, N))
		m[:N, 0] = 1
		m[N, 1:] = 1
		return m

	def get_green_matrix(self, well, s):
		N = well.params["nseg"] # number of elements in half-length
		m = np.zeros((1+2*N, 1+2*N))
		dx = 1./N
		u = s
		for i in range(2*N):
			if i < N:
				xd = well.xwd - 1 + i*dx
			else:
				xd = well.xwd - 1 + (i+1)*dx
			yd = well.ywd
			zd = 0
			for j in range(2*N):
				int_lim1 = well.xwd - 1 + j*dx
				int_lim2 = int_lim1+dx
				m[1+i,1+j] = well.source.Green(u, xd, yd, zd, xwd, ywd, zwd, int_lim1, int_lim2)

	def get_source_matrix(self, well, s):
		N = well.params["nseg"]
		m = np.zeros((1+2*N, 1+2*N))
		dx = 1./N
		if well.wtype == "frac":
			Fcd = well.params["Fcd"]
			coef = np.pi * dx* dx/Fcd
			for i in range(N):
				for j in range(i + 1):
					m[N + i, N + j] = i - j + 1
					m[N - i - 1, N - j - 1] = i - j + 1
			m *= coef
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
			b[N+i] = coef*(i+1)
			b[N-i-1] = coef*(i+1)
		return b

	def solve(self):
		pass