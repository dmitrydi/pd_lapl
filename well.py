class Well():
	# class for describing a well
	# in real space
	def __init__(self, xw, yw, type, zw=0, params = {}):
		pass

	def get_p(self, t, Q, x, y, z=0):
		# returns pressure response in space point (x,y,z) and time point t given well rate is Q
		pass

	def get_q(self, t, Pwf):
		# returns well rate Q at time t given the bottomhole pressure is Pwf
		pass

	def print_source_distribution(self, t):
		pass

