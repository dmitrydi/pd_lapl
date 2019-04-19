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
			self.lims1 = m_xd - c*m_xj1 - xwd
			self.lims2 = m_xd - c*m_xj2 - xwd
			self.mask_int_1 = np.matrix(np.ones((2*N, 2*N)), dtype=np.float) - 2*(self.lims1 < 0)
			self.mask_int_2 = np.matrix(np.ones((2*N, 2*N)), dtype=np.float) - 2*(self.lims2 >= 0)
			self.alims1 = np.round(np.abs(self.lims1), decimals=ndigits)
			self.alims2 = np.round(np.abs(self.lims2), decimals=ndigits)
			self.ulims = np.unique([np.round(self.alims1, decimals=ndigits), np.round(self.alims2, decimals=ndigits)])
		else:
			raise NotImplementedError