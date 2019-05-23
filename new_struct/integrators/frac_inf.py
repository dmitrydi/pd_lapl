import numpy as np
from .common import i_frac_0, i_frac_nnz

def integrate_matrix(u, matrixizer, sources):
    N = sources["sources_list"][0].nseg
    M = len(sources["sources_list"])
    orig_shape = (2*N*M, 2*N*M)
    m = np.zeros(orig_shape, dtype=np.float)
    if "0-" not in matrixizer.m_cache["dyds_0"].keys():
        matrixizer.make_calc_matr(sources, "-", 0)
    inds_dyds_0, unique_lims_dyd_0, inverse_inds_dyd_0, len_alims1, mask1, mask2 = matrixizer.m_cache["dyds_0"]["0-"]
    inds_dyds_nnz, unique_lims_dyd_nnz, inverse_inds_dyd_nnz = matrixizer.m_cache["dyds_nnz"]["0-"]
    su = np.sqrt(u)
    if inds_dyds_0 is not None:
        vals_dyds_0 = i_frac_0(unique_lims_dyd_0, su)
        matrixizer.reconstruct_dyds_0(m, inds_dyds_0, vals_dyds_0, inverse_inds_dyd_0, len_alims1, mask1, mask2)
    if inds_dyds_nnz is not None:
        f = lambda t: i_frac_nnz(t, su)
        vals_dyds_nnz = np.apply_along_axis(f, 1, unique_lims_dyd_nnz)
        matrixizer.reconstruct_dyds_nnz(m, inds_dyds_nnz, vals_dyds_nnz, inverse_inds_dyd_nnz)
    return m

