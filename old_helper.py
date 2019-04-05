import numpy as np
from source_helper import calc_bessel_integrals

class OldHelper():
	def __init__(self):
		pass

	def get_dummy_matrix(self, well):
		N = well.params["nseg"]
		m = np.zeros((N+1, N+1))
		m[:N, 0] = 1
		m[-1, 1:] = 1
		return m

	def get_green_matrix(self, well, s):
		N = well.params["nseg"] # number of elements in half-length
		m = np.zeros((1+N, 1+N))
		dx = 1./N
		omega = well.params["omega"]
		l = well.params["lambda"]
		bsls = calc_bessel_integrals(s, omega, l, N)
		for j in range(1, 1+N):
			for i in range(1, 1+N):
				if i <= j:
					numb = j - i + 1
				else:
					numb = i - j
				m[j-1, i] = 0.5*(bsls[numb] + bsls[j+i])
		return m

	def get_source_matrix(self, well, s):
		N = well.params["nseg"]
		m = np.zeros((1+N, 1+N))
		dx = 1./N
		if well.wtype == "frac":
			Fcd = well.params["Fcd"]
			coef = np.pi * dx* dx/Fcd
			for i in range(1, N+1):
				for j in range(1, N+1):
					if j <= i:
						m[i-1, j] = coef*(i - (j - 1))
					else:
						m[i-1, j] = 0
		else:
			raise
		return m

	def get_right_part(self, well, s):
		N = well.params["nseg"]
		dx = 1./N
		Fcd = well.params["Fcd"]
		coef = np.pi * dx/Fcd/s
		b = np.zeros((1+N))
		b[-1] = 1/s/dx
		for i in range(1,N+1):
			b[i-1] = coef*(i)
		return b

