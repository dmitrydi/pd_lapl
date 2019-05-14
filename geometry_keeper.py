import numpy as np

class GeometryKeeper:
	# works with Laplwell class
	def __init__(self):
		pass
		#self.well = well
		#self.make_fracture_geometry(self.sg, self.well.xwd, self.well.ywd, self.well.xwd, self.well.ywd, self.well.params["nseg"], self.well.c, self.well.round_val)

	#def make_mutual_geometry(self, gdict, that_well):
	#	N = this_well.params["nseg"]
	#	c = this_well.c
	#	ndigits = this_well.round_val
	#	self.make_fracture_geometry(gdict, self.well.xwd, self.well.ywd, that_well.xwd, that_well.ywd, N, c, ndigits)

	def make_fracture_geometry(self, well, xd, yd):
		N = well.params["nseg"]
		dx = 1./N
		xwd = well.xwd
		c = well.c
		ndigits = well.round_val
		ywd = well.ywd
		gdict = dict()
		m_xd = xd + dx*np.matrix(np.arange(-N+0.5, N+0.5), dtype=np.float).transpose()*np.matrix(np.ones(2*N, dtype=np.float))
		m_xj1 = dx*np.matrix(np.ones(2*N, dtype=np.float)).transpose()*np.matrix(np.arange(-N, N), dtype=np.float)
		m_xj2 = dx + m_xj1
		# values for iti0k0 - integration limits are combined here with the point of integration - xi
		gdict["mxd"] = m_xd
		gdict["myd"] = np.abs(np.matrix((yd - ywd)*np.ones_like(m_xd)))
		gdict["lims1"] = np.round(m_xd - c*m_xj1 - xwd, decimals=ndigits) # lower limits of integration within fracture: lims1[i, j] = dx*(i+0.5) - dx*(j-1)
		gdict["lims2"] = np.round(m_xd - c*m_xj2 - xwd, decimals=ndigits) # upper limits of integration within fracture: lims2[i, j] = dx*(i+0.5) - dx*j
		gdict["mask_int_1"] = np.matrix(np.ones((2*N, 2*N)), dtype=np.float) - 2*(gdict["lims1"] < 0) # mask for selecting sign for integrals of lims1 and lims 2
		gdict["mask_int_2"] = np.matrix(np.ones((2*N, 2*N)), dtype=np.float) - 2*(gdict["lims2"] >= 0)
		# absolute values to pass to iti0k0 function, rounding needed for determining unique values
		gdict["alims1"] = np.abs(gdict["lims1"])# absolute values of lims1 rounded to ndigits
		gdict["alims2"] = np.abs(gdict["lims2"]) # absolute values of lims2 rounded to ndigits
		gdict["ulims"] = np.unique([gdict["alims1"], gdict["alims2"]]) # unique values of limits for speedup
		gdict["xcs"] = xwd + dx*np.matrix(np.arange(-N+0.5, N+0.5), dtype=np.float) # centers of segments within fracture (as array)
		gdict["xends"] = xwd + dx*np.arange(-N, N+1, dtype = np.float) # all point-coordinates of segmants
		gdict["lends"] = xwd + dx*np.arange(-N, N, dtype = np.float) # left coordinates of segments (for calculations of integrals)
		return gdict