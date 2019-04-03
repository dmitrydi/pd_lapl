import numpy as np
from scipy.special import kn, k0, k1, iti0k0

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
			xd = well.xwd - 1 + 0.5*i*dx
			yd = well.ywd
			zd = 0
			for j in range(2*N):
				xcd = well.xwd - 1 + 0.5*j*dx
				ycd = well.ywd
				zcd = 0
				m[1+i,1+j] = well.source.Green(u, xd, yd, zd, xcd, ycd, zcd)

	def get_source_matrix(self, well, s):
		pass

	def get_right_part(self, well, s):
		pass

	def solve(self):
		pass