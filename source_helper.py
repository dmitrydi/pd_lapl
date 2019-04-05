from scipy.special import kn, k0, k1, iti0k0
from scipy.integrate import quad
import scipy.misc
fact = scipy.misc.factorial
import numpy as np

def frac_inf_imp_imp(xd, yd, xwd, ywd, su, c, int_lim1, int_lim2):
	if yd == ywd:
		if (xd - xwd - c*int_lim2) >=0:
			try:
				return 1/c/su*(iti0k0(su*abs(xd-xwd-c*int_lim1))[1] - iti0k0(su*abs(xd-xwd-c*int_lim2))[1])
			except:
				print(type(su*abs(xd-xwd-c*int_lim1)), type(su*abs(xd-xwd-c*int_lim2)))
				raise
		elif (xd - xwd - c*int_lim1) <0:
			try:
				return 1/c/su*(iti0k0(su*abs(c*int_lim2+xwd-xd))[1] - iti0k0(su*abs(c*int_lim1+xwd-xd))[1])
			except:
				print(type(su*abs(c*int_lim2+xwd-xd)), type(su*abs(c*int_lim1+xwd-xd)))
				raise
		else:
			try:
				return 1/c/su*(iti0k0(su*abs(xd-xwd-c*int_lim1))[1] + iti0k0(su*abs(c*int_lim2+xwd-xd))[1])
			except:
				print(type(su*abs(xd-xwd-c*int_lim1)), type(su*abs(c*int_lim2+xwd-xd)))
				raise
	else:
		return quad(integrand_frac_inf_imp_imp, int_lim1, int_lim2, args = (xd, xwd, yd, ywd, c, su))[0]

def integrand_frac_inf_imp_imp(x, xd, xwd, yd, ywd, c, su):
	return k0(su*((xd-xwd-c*x)**2+(yd-ywd)**2)**0.5)

def csteh(n, i):
    acc = 0.0
    for k in range(int(np.floor((i+1)/2.0)), int(min(i, n/2.0))+1):
        num = k**(n/2.0) * fact(2 * k)
        den = fact(i - k) * fact(k -1) * fact(k) * fact(2*k - i) * fact(n/2.0 - k)
        acc += (num /den)
    expo = i+n/2.0
    term = np.power(-1+0.0j,expo)
    res = term * acc
    return res.real

def nlinvsteh(F, t, n):
    acc = 0.0
    lton2 = np.log(2) / t
    for i in range(1, n+1):
        a = csteh(n, i)
        b = F(i * lton2)
        acc += (a * b)
    return lton2 * acc

def f_s(s, w, l):
	return (s*w*(1-w)+l)/(s*(1-w)+l)

def calc_bessel_integrals(s, omega, l, N):
	u = f_s(s, omega, l)
	su = u**0.5
	dx = 1./N
	r = np.zeros(2*N+1)
	bsls = np.zeros(2*N+1)
	for i in range(1, 2*N+1):
		r[i] = 1/su * iti0k0(i*dx*su)[1]
		bsls[i] = r[i] - r[i-1]
	return bsls
