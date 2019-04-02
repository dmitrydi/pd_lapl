class Well():
	# class for describing a well
	# in real space
	def __init__(self, xw, yw,
		outer_bound,
		top_bound,
		bottom_bound,
		wtype,
		zw=0, params = {"z_ref_length":1}):
		self.L = params["ref_length"]
		self.zL = params["z_ref_length"]
		self.xwd = xw/self.L
		self.ywd = yw/self.L
		self.zwd = zw/self.zL
		self.lapl_well = LaplWell(xwd, ywd, zwd, outer_bound, top_bound, bottom_bound, wtype, params)

	def get_p(self, t, Q, x, y, z=0):
		# returns pressure response in space point (x,y,z) and time point t given well rate is Q
		pass

	def get_q(self, t, Pwf):
		# returns well rate Q at time t given the bottomhole pressure is Pwf
		pass

	def print_source_distribution(self, t):
		pass

