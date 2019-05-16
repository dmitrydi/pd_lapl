import numpy as np

def make_matrix_for_integration(sources_list):
	M = len(sources_list)
	N = sources_list[0].nseg
	xis, xjs, xj1s, yws, yds = None, None, None, None, None
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
		yws = yds.T

	return xis, xjs, xj1s, yws, yds