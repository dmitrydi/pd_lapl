import numpy as np
from .common import i_frac_0, i_frac_nnz, make_calc_matr_

# def integrate_matrix(u, matrixizer, sources):
#     N = sources["sources_list"][0].nseg
#     M = len(sources["sources_list"])
#     orig_shape = (2*N*M, 2*N*M)
#     m = np.zeros(orig_shape, dtype=np.float)
#     if "0-" not in matrixizer.m_cache["dyds_0"].keys():
#         matrixizer.make_calc_matr(sources, "-", 0)
#     inds_dyds_0, unique_lims_dyd_0, inverse_inds_dyd_0, len_alims1, mask1, mask2 = matrixizer.m_cache["dyds_0"]["0-"]
#     inds_dyds_nnz, unique_lims_dyd_nnz, inverse_inds_dyd_nnz = matrixizer.m_cache["dyds_nnz"]["0-"]
#     su = np.sqrt(u)
#     if inds_dyds_0 is not None:
#         vals_dyds_0 = i_frac_0(unique_lims_dyd_0, su)
#         matrixizer.reconstruct_dyds_0(m, inds_dyds_0, vals_dyds_0, inverse_inds_dyd_0, len_alims1, mask1, mask2)
#     if inds_dyds_nnz is not None:
#         f = lambda t: i_frac_nnz(t, su)
#         vals_dyds_nnz = np.apply_along_axis(f, 1, unique_lims_dyd_nnz)
#         matrixizer.reconstruct_dyds_nnz(m, inds_dyds_nnz, vals_dyds_nnz, inverse_inds_dyd_nnz)
#     return m

def integrate_matrix(u, matrixizer, sources):
    xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_ = matrixizer.raw
    m = integrate_matrix_(u, matrixizer.m_cache, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_)
    return m

def integrate_matrix_(u, buf, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_):
    if "0-" not in buf["dyds_0"].keys():
        make_calc_matr_(buf, 0, 0, '-', 0, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_)
    inds_dyds_0, unique_lims_dyd_0, inverse_inds_dyd_0, len_alims1, mask1, mask2 = buf["dyds_0"]["0-"]
    inds_dyds_nnz, unique_lims_dyd_nnz, inverse_inds_dyd_nnz = buf["dyds_nnz"]["0-"]
    su = np.sqrt(u)
    orig_shape = xis_.shape
    m = np.zeros(orig_shape, dtype=np.float)
    if inds_dyds_0 is not None:
        vals_dyds_0 = i_frac_0(unique_lims_dyd_0, su)
        m = m.reshape(-1)
        v = vals_dyds_0[inverse_inds_dyd_0]
        v1 = v[:len_alims1]
        v2 = v[len_alims1:]
        m[inds_dyds_0] = v1*mask1 + v2*mask2
        m = m.reshape(orig_shape)
    if inds_dyds_nnz is not None:
        f = lambda t: i_frac_nnz(t, su)
        vals_dyds_nnz = np.apply_along_axis(f, 1, unique_lims_dyd_nnz)
        vals_dyds_nnz = np.apply_along_axis(f, 1, unique_lims_dyd_nnz)
        m = m.reshape(-1)
        m[inds_dyds_nnz] = vals_dyds_nnz[inverse_inds_dyd_nnz]
        m = m.reshape(orig_shape)
    return m


