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
		#dummy = np.zeros((1+2*N, 1+2*N, 3))
		dx = 1./N
		for i in range(-N, N):
			if i < 0:
				xd = well.xwd + dx*i
			else:
				xd = well.xwd + dx*(i+1)
			yd = well.ywd
			zd = 0
			for j in range(-N, N):
				int_lim1 = well.xwd + j*dx
				int_lim2 = well.xwd + (j + 1)*dx
				m[i+N,1+j+N] = well.source.Green(s, xd, yd, zd, well.xwd, well.ywd, well.zwd, int_lim1, int_lim2)
				#dummy[i+N,1+j+N, 0] = xd
				#dummy[i+N,1+j+N, 1] = int_lim1
				#dummy[i+N,1+j+N, 2] = int_lim2
		return m#, dummy

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
		else:
			raise
		return m

	def get_right_part(self, well, s):
		N = well.params["nseg"]
		dx = 1./N
		Fcd = well.params["Fcd"]
		coef = np.pi * dx/Fcd/s
		b = np.zeros((1+2*N))
		b[-1] = 0.5/s/dx
		for i in range(N):
			b[N+i] = coef*(i+1)
			b[N-i-1] = coef*(i+1)
		return b

