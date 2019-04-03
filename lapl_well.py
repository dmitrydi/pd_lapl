class LaplWell():
	# class for defining a well in Laplace space
	from source import LaplSource
	from helper import Helper

	def __init__(self,
		xwd, ywd, zwd,
		outer_bound, top_bound, bottom_bound,
		wtype, params):
		self.xwd = xwd
		self.ywd = ywd
		self.zwd = zwd
		self.outer_bound = outer_bound
		self.top_bound = top_bound
		self.bottom_bound = bottom_bound
		self.wtype - wtype
		self.params = params
		self.source = LaplSource(self.outer_bound,
			self.top_bound,
			self.bottom_bound,
			self.wtype,
			1./self.params["nseg"],
			[])
		self.source_distrib = None
		self.p_lapl = None
		self.q_lapl = None
		self.Q_lapl = None
		

	def recalc(self, s):
		helper = Helper()
		dummy_matrix = helper.get_dummy_matrix(self)
		green_matrix = helper.get_green_matrix(self, s)
		source_matrix = helper.get_source_matrix(self, s)
		solution = helper.solve(dummy_matrix, green_matrix, source_matrix)
		self.p_lapl = solution[0]
		self.source_distrib = solution[1:]
		self.q_lapl = 1./s/s/p_lapl
		self.Q_lapl = 1./s/p_lapl # check!


