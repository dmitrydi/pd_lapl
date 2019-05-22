from scipy.special import kn, k0, k1, iti0k0
from source_helper import *
from scipy.integrate import quad, fixed_quad

class LaplSource():
# class for a finite element (finite liquid source) of a well or fracture
# all in laplace space
	def __init__(self,
		well):
		self.well = well

	def f_s(self, s):
		w = self.well.params["omega"]
		l = self.well.params["lambda"]
		return (s*w*(1-w)+l)/(s*(1-w)+l)

	def frac_source_function(self, x, s, xd, yd, xwd, ywd, c):
		fs = self.f_s(s)
		su = (s*fs)**0.5
		return 0.5*k0(su*((xd - xwd - c*x)**2 + (yd - ywd)**2)**0.5) #check 0.5!

	def frac_source_func(self, x, dyd, su):
		if self.well.outer_bound == "infinite":
			return 0.5*k0(su*(x*x + dyd*dyd)**0.5)
		else:
			raise NotImplementedError

	def integrate_source_functions_bounded(self, s, xis, xjs, xj1s, xed, yds, ywd, yed):
		dyds = np.abs(yds - ywd)
		return self.ifb1(s, xis, xjs, xj1s, xed, yds, ywd, yed) + self.ifb2(s, xis, xjs, xj1s, xed, yds, ywd, yed) + self.ifb3(s, xis, xjs, xj1s, xed, dyds)

	def ifb1(self, s, xis, xjs, xj1s, xed, yds, ywd, yed):
		u = s*self.f_s(s)
		su = u**0.5
		sexp = self.calc_sexp(su, yed)
		yd1_w = yed - np.abs(yds-ywd)
		yd2_w = yed - (yds+ywd)
		p1 = 0.5*np.pi/xed/su*(np.exp(-su*(yds+ywd)) + np.exp(-su*(yed+yd2_w)) + np.exp(-su*np.abs(yds-ywd)) + np.exp(-su*(yed+yd1_w)))
		p1 *= (1 + sexp)
		p1 *= xj1s - xjs
		return p1

	def ifb2(self, s, xis, xjs, xj1s, xed, yds, ywd, yed, debug = False):
		blk_size = 10
		MAXITER = 1000
		TINY = 1e-20
		EPS = 1e-12
		sum_ = np.zeros_like(xis)
		u = s*self.f_s(s)
		for i in range(MAXITER):
			blk = np.arange(1+i*blk_size, 1+(i+1)*blk_size)
			d = np.zeros_like(xis)
			for k in blk:
				d += self.ifb2_k(k, u, xis, xjs, xj1s, xed, yds, ywd, yed)
			sum_ += d
			if np.mean(np.abs(d))/(np.mean(np.abs(sum_))+TINY)  < EPS and i > 3:
				if debug:
					return sum_, i*blk_size
				else:
					return sum_
		raise RuntimeWarning("ifb2 did not converge in {} steps".format(MAXITER))

	def ifb2_k(self, k, s, xis, xjs, xj1s, xed, yds, ywd, yed):
		ek = (s + k*k*np.pi*np.pi/xed/xed)**0.5
		yd1_w = yed - np.abs(yds-ywd)
		yd2_w = yed - (yds+ywd)
		sexp = self.calc_sexp(ek, yed)
		p1 = 2./k/ek*np.cos(k*np.pi*xis/xed)
		p1 *= np.sin(k*np.pi/2./xed*(xj1s-xjs))
		p1 *= np.cos(k*np.pi/2./xed*(2*xis - (xjs + xj1s)))
		p1 *= (np.exp(-ek*(yds+ywd)) + np.exp(-ek*(yed+yd1_w)) + np.exp(-ek*(yed+yd2_w)))*(1 + sexp) + np.exp(-ek*np.abs(yds-ywd))*sexp
		return p1

	def calc_sexp(self, ek, yed):
		ek_ = ek*yed
		MAXITER = 300
		blk_size = 3
		TINY = 1e-20
		EPS = 1e-12
		sum_ = 0.
		for i in range(1, MAXITER):
			blk = np.arange(1+i*blk_size, 1+(i+1)*blk_size)
			d = np.sum(np.exp(-2*blk*ek_))
			sum_ += d
			if d/(sum_ + TINY) < EPS:
				return sum_
		raise RuntimeWarning("calc_sexp did not converge in {} steps".format(MAXITER))

	def ifb3(self, s, xis, xjs, xj1s, xed, dyds):
		return self.ifb3_1(s, xis, xjs, xj1s, xed, dyds) + self.ifb3_2(s, xjs, xj1s, xed, dyds)

	def ifb3_1(self, s, xis, xjs, xj1s, xed, dyds, debug=False):
		KMAX = 100
		EPS = 1e-12
		TINY = 1e-20
		sum_ = self.ifb3_k(s, 0, xis, xjs, xj1s, xed, dyds)
		for k in range(1, KMAX):
			d = self.ifb3_k(s, k, xis, xjs, xj1s, xed, dyds)
			d += self.ifb3_k(s, -k, xis, xjs, xj1s, xed, dyds)
			sum_ += d
			if np.max(d)/(np.min(sum_) + TINY) < EPS:
				if debug:
					return sum_, k
				else:
					return sum_
		raise RuntimeWarning("ifb3_1 did not converge in {} steps".format(KMAX))

	def ifb3_2(self, s, xjs, xj1s, xed, dyds):
		su = (s*self.f_s(s))**0.5
		return -0.5*np.pi/xed/su*(xj1s - xjs)*np.exp(-su*dyds)

	def ifb3_k(self, s, k, xis, xjs, xj1s, xed, dyds):
		su = (s*self.f_s(s))**0.5
		orig_shape = xis.shape
		fxis, fxjs, fxj1s, fdyds = xis.reshape(-1), xjs.reshape(-1), xj1s.reshape(-1), dyds.reshape(-1)
		ans = np.zeros_like(fxis)
		inds_0 = np.argwhere(np.isclose(fdyds, 0.)).flatten()
		inds_nnz = np.argwhere(fdyds != 0.).flatten()
		xis_0, xjs_0, xj1s_0 = fxis[inds_0], fxjs[inds_0], fxj1s[inds_0]
		xis_nnz, xjs_nnz, xj1s_nnz = fxis[inds_nnz], fxjs[inds_nnz], fxj1s[inds_nnz]
		if len(inds_0) > 0:
			a1_ = su*(xjs_0 + xis_0 - 2*k*xed)
			b1_ = su*(xj1s_0 + xis_0 - 2*k*xed)
			a2_ = su*(xjs_0 - xis_0 - 2*k*xed)
			b2_ = su*(xj1s_0 - xis_0 - 2*k*xed)
			s1 = 1 - 2*(a1_ > 0.)
			s2 = 1 - 2*(b1_ < 0.)
			s3 = 1 - 2*(a2_ > 0.)
			s4 = 1 - 2*(b2_ < 0.)
			a1 = np.round(np.abs(a1_), decimals=6)
			b1 = np.round(np.abs(b1_), decimals=6)
			a2 = np.round(np.abs(a2_), decimals=6)
			b2 = np.round(np.abs(b2_), decimals=6)
			u1, i1 = np.unique(a1, return_inverse = True)
			u2, i2 = np.unique(b1, return_inverse = True)
			u3, i3 = np.unique(a2, return_inverse = True)
			u4, i4 = np.unique(b2, return_inverse = True)
			ua = np.concatenate((u1, u2, u3, u4))
			u, i = np.unique(ua, return_inverse = True)
			va = 0.5/su*iti0k0(u)[1][i]
			v1 = va[:len(u1)][i1]
			v2 = va[len(u1):len(u1)+len(u2)][i2]
			v3 = va[len(u1)+len(u2):len(u1)+len(u2)+len(u3)][i3]
			v4 = va[len(u1)+len(u2)+len(u3):][i4]
			ans[inds_0] = v1*s1 + v2*s2 + v3*s3 + v4*s4
		if len(inds_nnz) > 0:
			raise NotImplementedError
		return ans.reshape(orig_shape)

	def integrate_source_functions(self, s, lims1, lims2, dyds, int_type = "quad", npoints = 5):
		fs = self.f_s(s)
		su = (s*fs)**0.5
		c = self.well.c
		assert len(lims1) == len(lims2) == len(dyds)
		ans = np.zeros_like(lims1)
		if self.well.wtype == "frac":
			if self.well.outer_bound == "infinite":
				dyds_0_inds = np.argwhere(dyds==0.).flatten()
				dyds_nnz_inds = np.argwhere(dyds!=0.).flatten()
				# calcculate for zero dyds:
				if len(dyds_0_inds) > 0:
					ints_dyd_0_lims1 = 0.5/c/su*iti0k0(su*lims1[dyds_0_inds])[1]
					ints_dyd_0_lims2 = 0.5/c/su*iti0k0(su*lims2[dyds_0_inds])[1]
					ints_dyd_0 = ints_dyd_0_lims2 - ints_dyd_0_lims1
					ans[dyds_0_inds] = ints_dyd_0
				# calculate for nonzero dyds:
				if len(dyds_nnz_inds) > 0:
					if int_type == "quad":
						g = lambda x: quad(self.frac_source_func, x[0], x[1], args=(x[2], su))[0]
					elif int_type == "fixed_quad":
						g = lambda x: fixed_quad(self.frac_source_func, x[0], x[1], args=(x[2], su), n=npoints)[0]
					else:
						raise ArgumentError("int type should be 'quad' or 'fixed_quad'")
					nnz_items = np.hstack([lims1[dyds_nnz_inds].reshape(-1,1), lims2[dyds_nnz_inds].reshape(-1,1), dyds[dyds_nnz_inds].reshape(-1,1)])
					ans[dyds_nnz_inds] = np.apply_along_axis(g, 1, nnz_items)
			elif self.well.outer_bound== "nnnn":
				raise NotImplementedError
		else:
			raise NotImplementedError
		return ans






'''
	def calc_source_xd_yd(self, s, xd, yd):
		fs = self.f_s(s)
		su = (s*fs)**0.5
		c = self.well.c
		dx = 1./self.well.params["nseg"]
		g = lambda x: self.integrand_frac_inf_imp_imp(x, xd, self.well.xwd, yd, self.well.ywd, c, su)
		f = lambda xc: quad(g, xc-0.5*dx, xc+0.5*dx)
		return 1/c/su*f(self.well.gk.xcs) # TBD: 1 vs. 0.5

	def integrand_frac_inf_imp_imp(self, x, xd, xwd, yd, ywd, c, su):
		return k0(su*((xd-xwd-c*x)**2+(yd-ywd)**2)**0.5)

	def Green(self, s, xd, yd, zd, xwd, ywd, zwd, int_lim1, int_lim2):
		# Green's function of the source of well located in point (xwd, ywd, zwd) (response in coordinate point (xd, yd, zd)), depends on type and boundaries
		# int_lim1, int_lim2 - ends of a source along x-axis
		u = s*self.f_s(s)
		su = u**0.5
		c = self.well.c
		if self.well.wtype == "frac":
			if self.well.outer_bound == "infinite":
				if self.well.top_bound == "imp":
					if self.well.bottom_bound == "imp":
						assert int_lim1 <= int_lim2
						return frac_inf_imp_imp(xd, yd, xwd, ywd, su, c, int_lim1, int_lim2)
					else:
						raise NotImplementedError
				else:
					raise NotImplementedError
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError

'''



