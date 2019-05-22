import numpy as np
from .integrators import frac_inf, frac_nnnn

def integrate_sources_for_green_matrix(u, matrixizer, sources):
	wtype = sources["wtype"]
	outer_bound = sources["boundaries"]["outer_bound"]
	if wtype == "frac":
		if outer_bound == "inf":
			m = frac_inf.integrate_matrix(u, matrixizer, sources)
		elif outer_bound == "nnnn":
			m = frac_nnnn.integrate_matrix(u, matrixizer, sources)
		else:
			raise NotImplementedError
	else:
		raise NotImplementedError
	return m

