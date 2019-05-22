import numpy as np
from scipy.special import k0, iti0k0
from scipy.integrate import quad

def i_frac_0(x, su):
	# x = np.array([vals])
	return 0.5/su*iti0k0(x*su)[1]

def i_frac_nnz(x, su):
	upper_lim, dx, dyd = x[0], x[1], x[-1]
	return quad(frac_fun, su*(upper_lim-dx), su*upper_lim, args=(dyd, su))[0]

def frac_fun(x, dyd, su):
	return 0.5/su*k0((x*x + su*su*dyd*dyd)**0.5)