import numpy as np

class Sources:
	def __init__(self,
		xwd, ywd, zwd,
		xed, yed, hd,
		wtype,
		x_lengths,
		outer_bound,
		top_bound,
		bottom_bound,
		nseg):
		self.xwd = xwd
		self.ywd = ywd
		self.zwd = zwd
		self.xed = xed
		self.yed = yed
		self.hd = hd
		self.typ = wtype
		self.ref_len = x_lengths
		self.outer_bound = outer_bound
		self.top_bound = top_bound
		self.bottom_bound = bottom_bound
		self.nseg = nseg
		dx = x_lengths/nseg
		# coordinates of centers of segments
		self.xis = dx*np.arange(-nseg+0.5, nseg+0.5) + self.xwd
		# left x-coordinates of segments
		self.xjs = dx*np.arange(-nseg, nseg, dtype = np.float) + self.xwd
		# right x-coordinates of segments
		self.xj1s = dx*np.arange(-nseg+1, nseg+1, dtype = np.float) + self.xwd
		self.ys = self.ywd*np.ones_like(self.xjs, dtype = np.float)
		self.zs = self.zwd*np.ones_like(self.xjs, dtype = np.float)
