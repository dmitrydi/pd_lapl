class LaplSource():
# class for a finite element (finite liquid source) of a well or fracture
# all in laplace space
	def __init__(self,
		outer_bound,
		top_bound,
		bottom_bound,
		type,
		size,
		xc,
		yc,
		u,
		params={},
		zc=0):
		self.q = None
		self.outer_bound = outer_bound
		self.top_bound = top_bound
		pass

	def Green(self, x, y, z):
		# Green's function of the source (response in coordinate point (x,y,z)), depends on type and boundaries
		pass

	def pd(x,y,z=0):
		# pressure response in space point (x,y,z)
		# works only if source strength is calculated
		if self.q is None:
			raise ValueError("Source strength q in unindentified")
		pass




