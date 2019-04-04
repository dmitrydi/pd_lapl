from scipy.special import kn, k0, k1, iti0k0
from scipy.integrate import quad

def frac_inf_imp_imp(xd, yd, xwd, ywd, su, c, int_lim1, int_lim2):
	if yd == ywd:
		if (xd - xwd - c*int_lim2) >=0:
			return 1/c/su*(iti0k0(su*(xd-xwd-c*int_lim1))[1] - iti0k0(su*(xd-xwd-c*int_lim2))[1])
		elif (xd - xwd - c*int_lim1) <=0:
			return 1/c/su*(iti0k0(su*(c*int_lim2+xwd-xd))[1] - iti0k0(su*(c*int_lim1+xwd-xd))[1])
		else:
			return 1/c/su*(iti0k0(su*(xd-xwd-c*int_lim1))[1] + iti0k0(su*(c*int_lim2+xwd-xd))[1])
	else:
		return quad(integrand_frac_inf_imp_imp, int_lim1, int_lim2, args = (xd, xwd, yd, ywd, c, su))[0]

def integrand_frac_inf_imp_imp(x, xd, xwd, yd, ywd, c, su):
	return k0(su*((xd-xwd-c*x)**2+(yd-ywd)**2)**0.5)

