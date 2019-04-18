import numpy as np

class GeometryKeeper:
	# works with Laplwell class
	def __init__(self, well):
		self.well = well
		self.matrices = self.make_geometry_matrices(self.well)

	def make_geometry_matrices(self, well):
		if self.well.wtype == "frac":
			N = self.well.params["nseg"]
			dx = 1./N
			xwd = self.well.xwd
			c = self.well.c
			ndigits = self.well.round_val
			m_xd = xwd + dx*np.matrix(np.arange(-N+0.5, N+0.5), dtype=np.float).transpose()*np.matrix(np.ones(2*N, dtype=np.float))
			m_xj1 = xwd + dx*np.matrix(np.ones(2*N, dtype=np.float)).transpose()*np.matrix(np.arange(-N, N), dtype=np.float)
			m_xj2 = dx + m_xj1
			lims1 = m_xd - c*m_xj1 - xwd
			lims2 = m_xd - c*m_xj2 - xwd
			mask_int_1 = np.matrix(np.ones((2*N, 2*N)), dtype=np.float) - 2*(lims1 < 0)
			mask_int_2 = np.matrix(np.ones((2*N, 2*N)), dtype=np.float) - 2*(lims2 >= 0)
			alims1 = np.abs(lims1)
			alims2 = np.abs(lims2)
			ulims = np.unique([np.round(alims1, decimals=ndigits), np.round(alims2, decimals=ndigits)])
		else:
			raise NotImplementedError
		return {"low_lims": lims1, "high_lims": lims2, "low_mask": mask_int_1, "high_mask": mask_int_2,
						"low_alims": alims1, "high_alims": alims2, "ulims": ulims}