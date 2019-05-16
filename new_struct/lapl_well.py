import numpy as np
from sources import Sources

class LaplWell():
	"""class for lapl well"""
	def __init__(self, outer_bound, top_bound, bottom_bound,
		wtype, nseg, nwells, xwds, ywds, x_lengths = 1,
		xed = None, yed = None, zwds = None, hd = None):
		# check initialization
		if outer_bound != "inf":
			assert xed is not None
			assert yed is not None
		if wtype == "horiz":
			assert zwds is not None
			assert hd is not None
		assert nwells >= 1
		if nwells > 1:
			assert len(xwds) == len(ywds) == nwells
			if wtype == "horiz":
				assert len(zwds) == nwells
		##
		self.wtype = wtype
		self.nwells = nwells
		self.xwds = np.array([xwds]).flatten()
		self.ywds = np.array([ywds]).flatten()
		if zwds is None:
			self.zwds = np.zeros_like(self.xwds, dtype = np.float)
		else:
			self.zwds = np.array(zwds)
		self.sources_ = []
		for xwd, ywd, zwd in zip(self.xwds, self.ywds, self.zwds):
			self.sources_.append(Sources(xwd, ywd, zwd,
														xed, yed, hd,
														wtype,
														x_lengths,
														outer_bound,
														top_bound,
														bottom_bound,
														nseg))



		
		
		