import numpy as np
from scipy.linalg import block_diag

def make_matrices_for_integration(sources_list):
	M = len(sources_list)
	N = sources_list[0].nseg
	xis, xjs, xj1s, yws, yds, zws, zds = None, None, None, None, None, None, None
	for src in sources_list:
		# make xis
		xis_ = np.tile(src.xis, (2*N, 1)).T
		xis_ = np.tile(xis_, M)
		if xis is None:
			xis = xis_.copy()
		else:
			xis = np.vstack([xis, xis_])
		# make xjs
		xjs_ = np.tile(src.xjs, (2*N, 1))
		xjs_ = np.tile(xjs_, (M, 1))
		if xjs is None:
			xjs = xjs_.copy()
		else:
			xjs = np.hstack([xjs, xjs_])
		# make xj1s:
		xj1s_ = np.tile(src.xj1s, (2*N, 1))
		xj1s_ = np.tile(xj1s_, (M, 1))
		if xj1s is None:
			xj1s = xj1s_.copy()
		else:
			xj1s = np.hstack([xj1s, xj1s_])
		# make yds
		yds_ = np.tile(src.ys, (2*N, 1))
		yds_ = np.tile(yds_, M)
		if yds is None:
			yds = yds_.copy()
		else:
			yds = np.vstack([yds, yds_])
		# make yws
		yws = yds.T
		# make zds
		zds_ = np.tile(src.zs, (2*N, 1))
		zds_ = np.tile(zds_, M)
		if zds is None:
			zds = zds_.copy()
		else:
			zds = np.vstack([zds, zds_])
		# make zws
		zws = zds.T
	return xis, xjs, xj1s, yws, yds, zws, zds

def make_dummy_matrix(sources_list):
	N = sources_list[0].nseg
	M = len(sources_list)
	m = np.zeros((1+2*N*M, 1+2*N*M), dtype = np.float)
	m[:-1,0] = 1.
	m[-1, 1:] = 1.
	return m

def make_source_matrix(sources_list, wtype, attrs):
	N = sources_list[0].nseg
	M = len(sources_list)
	m = np.zeros((1+2*N*M, 1+2*N*M))
	m_ = np.zeros((N,N))
	dx = 1./N
	dx2_8 = 0.125*dx**2
	dx2_2 = 0.5*dx**2
	if wtype == "frac":
		Fcd = attrs["Fcd"]
		coef = np.pi/Fcd
		for j in range(1, N+1):
			m_[j-1, j-1] = dx2_8
			xj = dx*(j-0.5)
			for i in range(1, j):
				m_[j-1, i-1] = dx2_2 + dx*(xj - i*dx)
		m_ = block_diag(np.flip(np.flip(m_, 0), 1), m_)
		dum_list = []
		for _ in range(M):
			dum_list.append(m_)
		m_ = block_diag(*dum_list)
		m[:-1, 1:] = m_
		m *= coef
	else:
		raise NotImplementedError
	return m

def make_dummy_rhv(sources_list, wtype, attrs):
	N = sources_list[0].nseg
	M = len(sources_list)
	b_ = np.zeros((2*N))
	dx = 1./N
	rhv = np.array([], dtype = np.float)
	if wtype == "frac":
		Fcd = attrs["Fcd"]
		coef = np.pi * dx/Fcd/M
		for i in range(N):
			b_[N+i] = (i+0.5)
			b_[N-i-1] = (i+0.5)
		b_ *= coef
		for _ in range(M):
			rhv = np.append(rhv, b_)
		rhv = np.append(rhv, 2./dx)
	return rhv