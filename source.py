class LaplSource():
# class for a finite element (finite liquid source) of a well or fracture
# all in laplace space
	def __init__(self,
		outer_bound,
		top_bound,
		bottom_bound,
		wtype,
		x_size,
		params={}):
		self.q = None
		self.outer_bound = outer_bound
		self.top_bound = top_bound
		self.bottom_bound = bottom_bound
		self.wtype = wtype
		self.x_size = x_size # dimentionless length of a segment
		self.params = params

	def Green(self, u, xd, yd, zd, xcd, ycd, zcd):
		# Green's function of the source (response in coordinate point (xd, yd, zd)), depends on type and boundaries
		pass

	def pd(u, xd, yd, zd, xcd, ycd, zcd):
		# pressure response in space point (xd, yd, zd)
		# works only if source strength is calculated
		if self.q is None:
			raise ValueError("Source strength q in unindentified")
		pass




