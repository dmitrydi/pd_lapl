import numpy as np
from lapl_well import LaplWell
from geometry_keeper import GeometryKeeper
from scipy.linalg import block_diag

class MultifracLapl():
	def __init__(self, nwells, xws, yws, outer_bound, top_bound, bottom_bound, params, n_stehf):
		self.wells = dict()
		for i in range(nwells):
			self.wells["well_"+str(i)] = LaplWell(xws[i], yws[i], 0., outer_bound, top_bound, bottom_bound, "frac", params)
		g = GeometryKeeper()
		self.N = self.wells["well_0"].params["nseg"]
		self.nwells = nwells
		self.geo = np.zeros((nwells*2*self.N, nwells*2*self.N, 5))
		for i in range(nwells):
			this_well = self.wells["well_" + str(i)]
			for j in range(nwells):
				that_well = self.wells["well_" + str(j)]
				gdict = g.make_fracture_geometry(that_well, this_well.xwd, this_well.ywd)
				self.geo[i*2*self.N:(i+1)*2*self.N, j*2*self.N:(j+1)*2*self.N, 0] = gdict["alims1"]
				self.geo[i*2*self.N:(i+1)*2*self.N, j*2*self.N:(j+1)*2*self.N, 1] = gdict["alims2"]
				self.geo[i*2*self.N:(i+1)*2*self.N, j*2*self.N:(j+1)*2*self.N, 2] = gdict["myd"]
				self.geo[i*2*self.N:(i+1)*2*self.N, j*2*self.N:(j+1)*2*self.N, 3] = gdict["mask_int_1"]
				self.geo[i*2*self.N:(i+1)*2*self.N, j*2*self.N:(j+1)*2*self.N, 4] = gdict["mask_int_2"]
		self.geo = self.geo.reshape(-1, 5)
		self.last_s = -1.

	def pw_lapl(self, s):
		if s != self.last_s:
			self.recalc(s)
		return self.p_lapl

	def recalc(self, s):
		self.last_s = s # keeps the last s of recalc of the well, for speed
		#helper = Helper()
		green_matrix = self.get_green_matrix(s)
		source_matrix = self.get_source_matrix(s)
		matrix = self.combine_matrix(green_matrix, source_matrix)
		right_part = self.get_right_part(s)
		solution = np.linalg.solve(matrix, right_part)
		self.p_lapl = solution[0]

	def get_green_matrix(self, s):
		orig_shape = (self.nwells*self.N*2, self.nwells*self.N*2) # original shape of resultant Green matrix
		green_matrix = np.zeros(self.geo.shape[0])
		# find indices values where dyd == 0:
		dyd_0_inds = np.argwhere(self.geo[:, 2] == 0.).flatten()
		items_dyd_0 = self.geo[dyd_0_inds]
		# select unique values for using iti0k0:
		lims1 = items_dyd_0[:,0]
		lims2 = items_dyd_0[:,1]
		lims = np.append(lims1, lims2)
		mask_1 = items_dyd_0[:,-2]
		mask_2 = items_dyd_0[:,-1]
		n_lims1 = len(lims1)
		ulims, inds = np.unique(lims, return_inverse = True)
		uvals = self.wells["well_0"].source.integrate_source_functions(s,
				np.zeros_like(ulims, dtype=np.float),
				ulims,
				np.zeros_like(ulims, dtype=np.float))
		vals = uvals[inds]
		vals1 = vals[:n_lims1]
		vals2 = vals[n_lims1:]
		vals = vals1*mask_1 + vals2*mask_2
		green_matrix[dyd_0_inds] = vals
		# find indices where dyd != 0
		dyd_nnz_inds = np.argwhere(self.geo[:, 2] != 0.).flatten()
		items_dyd_nnz = self.geo[dyd_nnz_inds][:,:3]
		lims = items_dyd_nnz[:,:2]
		upper_lims = np.max(lims, axis=1).reshape(-1, 1)
		upper_lims_dyds = np.hstack([upper_lims, items_dyd_nnz[:,-1].reshape(-1, 1)])
		ulims_dyd_nnz, inds_dyd_nnz = np.unique(upper_lims_dyds, axis=0, return_inverse=True)
		uvals_dyd_nnz = self.wells["well_0"].source.integrate_source_functions(s,
				ulims_dyd_nnz[:, 0] - 1./self.N,
				ulims_dyd_nnz[:, 0],
				ulims_dyd_nnz[:, 1])
		vals_dyd_nnz = uvals_dyd_nnz[inds_dyd_nnz]
		green_matrix[dyd_nnz_inds] = vals_dyd_nnz
		# done
		return green_matrix.reshape(orig_shape)

	def get_source_matrix(self, s):
		single_source_matrix = self.wells["well_0"].get_source_matrix(s)[:-1, 1:]
		l = []
		for _ in range(self.nwells):
			l.append(single_source_matrix)
		return block_diag(*l)

	def get_right_part(self, s):
		single_right_part = self.wells["well_0"].get_right_part(s)
		rp =  np.array([], dtype=np.float)
		for _ in range(self.nwells):
			rp = np.append(rp, single_right_part[:-1]/self.nwells) # note here division by number of wells
		rp = np.append(rp, single_right_part[-1])
		return rp

	def combine_matrix(self, green_matrix, source_matrix):
		N = self.N
		M = self.nwells
		ans = np.zeros((2*N*M+1, 2*N*M+1), dtype = np.float)
		ans[:-1,0] = 1.
		ans[-1, 1:] = 1.
		ans[:-1, 1:] = source_matrix - green_matrix
		return ans



