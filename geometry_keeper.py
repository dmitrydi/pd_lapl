import numpy as np

class GeometryKeeper:
	# works with Laplwell class
	def __init__(self, well):
		self.make_geometry_matrices(well)

	def make_geometry_matrices(self, well):
		if well.wtype == "frac":
			N = well.params["nseg"]
			dx = 1./N
			xwd = well.xwd
			c = well.c
			ndigits = well.round_val
			m_xd = xwd + dx*np.matrix(np.arange(-N+0.5, N+0.5), dtype=np.float).transpose()*np.matrix(np.ones(2*N, dtype=np.float))
			m_xj1 = xwd + dx*np.matrix(np.ones(2*N, dtype=np.float)).transpose()*np.matrix(np.arange(-N, N), dtype=np.float)
			m_xj2 = dx + m_xj1
			self.lims1 = m_xd - c*m_xj1 - xwd # lower limits of integration within fracture: lims1[i, j] = dx*(i+0.5) - dx*(j-1)
			self.lims2 = m_xd - c*m_xj2 - xwd # upper limits of integration within fracture: lims2[i, j] = dx*(i+0.5) - dx*j
			self.mask_int_1 = np.matrix(np.ones((2*N, 2*N)), dtype=np.float) - 2*(self.lims1 < 0) # mask for selecting sign for integrals of lims1 and lims 2
			self.mask_int_2 = np.matrix(np.ones((2*N, 2*N)), dtype=np.float) - 2*(self.lims2 >= 0)
			# absolute values to pass to iti0k0 function, rounding needed for determining unique values
			self.alims1 = np.round(np.abs(self.lims1), decimals=ndigits) # absolute values of lims1 rounded to ndigits
			self.alims2 = np.round(np.abs(self.lims2), decimals=ndigits) # absolute values of lims2 rounded to ndigits
			self.ulims = np.unique([np.round(self.alims1, decimals=ndigits), np.round(self.alims2, decimals=ndigits)]) # unique values of limits for speedup
			self.xcs = xwd + dx*np.matrix(np.arange(-N+0.5, N+0.5), dtype=np.float) # centers of segments within fracture
			self.xends = xwd + dx*np.arange(-N, N+1, dtype = np.float)
		else:
			raise NotImplementedError