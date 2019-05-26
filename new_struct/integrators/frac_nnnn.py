import numpy as np
from .common import i_frac_0, i_frac_nnz
from .cy.integrate_dyd_nnz import cy_reconstruct_dyd_nnz 

def integrate_matrix(u, matrixizer, sources, version='old'):
    xed = sources["sources_list"][0].xed
    yed = sources["sources_list"][0].yed
    if "fb_1_2_nnnn" not in matrixizer.m_cache.keys():
        matrixizer.make_matr_for_fb_1_2_nnnn(sources)
    arg_x_0, arg_x_1, arg_x_2, arg_x_3, arg_y_1, arg_y_2, arg_y_3, arg_y_4 = matrixizer.m_cache["fb_1_2_nnnn"]
    # (x1js - xjs), np.pi*xis/xed, np.pi/2./xed*arg_x_0, np.pi/2./xed*(2*xis - (xjs + xj1s)), yds+yws, yed+yd1_w, yed+yd2_w, np.abs(yds-ywd)
    i1 = ifb1(u, arg_x_0, arg_y_1, arg_y_2, arg_y_3, arg_y_4, xed, yed)
    i3 = ifb3(u, matrixizer, sources, arg_x_0, arg_y_4, xed, version)
    dlm = np.min([np.linalg.norm(i1), np.linalg.norm(i3)])
    i2 = ifb2(u, arg_x_1, arg_x_2, arg_x_3, arg_y_1, arg_y_2, arg_y_3, arg_y_4, xed, yed, dlm = dlm)
    
    return i1 + i2 + i3

def ifb1(u, arg_x_0, arg_y_1, arg_y_2, arg_y_3, arg_y_4, xed, yed):
    su = u**0.5
    sexp = calc_sexp(su, yed)
    p1 = 0.5*np.pi/xed/su*(np.exp(-su*arg_y_1) + np.exp(-su*arg_y_2) + np.exp(-su*arg_y_3) + np.exp(-su*arg_y_4))
    p1 *= (1 + sexp)
    p1 *= arg_x_0
    return p1

def ifb2(u, arg_x_1, arg_x_2, arg_x_3, arg_y_1, arg_y_2, arg_y_3, arg_y_4, xed, yed, dlm = None, debug = False):
    blk_size = 10
    MAXITER = 1000
    TINY = 1e-20
    EPS = 1e-12
    sum_ = np.zeros_like(arg_x_1)
    if dlm is None:
        dlm = 0.
    for i in range(MAXITER):
        blk = np.arange(1+i*blk_size, 1+(i+1)*blk_size)
        d = np.zeros_like(arg_x_1)
        for k in blk:
            d += ifb2_k(k, u, arg_x_1, arg_x_2, arg_x_3, arg_y_1, arg_y_2, arg_y_3, arg_y_4, xed, yed)
        sum_ += d
        if np.linalg.norm(d)/(np.linalg.norm(sum_)+ dlm + TINY) < EPS:
            if debug:
                return sum_, i*blk_size
            else:
                return sum_
    raise RuntimeWarning("ifb2 did not converge in {} steps".format(MAXITER))

def ifb2_k(k, u, arg_x_1, arg_x_2, arg_x_3, arg_y_1, arg_y_2, arg_y_3, arg_y_4, xed, yed):
    ek = (u + k*k*np.pi*np.pi/xed/xed)**0.5
    sexp = calc_sexp(ek, yed)
    p1 = 2./k/ek*np.cos(k*arg_x_1) # arg_x_1
    p1 *= np.sin(k*arg_x_2) # arg_x_2
    p1 *= np.cos(k*arg_x_3) # arg_x_3
    p1 *= (np.exp(-ek*arg_y_1) + np.exp(-ek*arg_y_2) + np.exp(-ek*arg_y_3))*(1 + sexp) + np.exp(-ek*arg_y_4)*sexp # arg_y_1, arg_y_2, arg_y3, arg_y_4
    return p1

def ifb3(u, matrixizer, sources, arg_x_0, arg_y_4, xed, version='old'):
    a = ifb3_1(u, matrixizer, sources, debug = False, version=version)
    b = ifb3_2(u, arg_x_0, arg_y_4, xed)
    return a + b

def ifb3_1(u, matrixizer, sources, debug=False, version='old'):
    KMAX = 100
    EPS = 1e-12
    TINY = 1e-20
    sum_ = ifb3_k(u, 0, "-", matrixizer, sources, version)
    sum_ += ifb3_k(u, 0, "+", matrixizer, sources, version)
    for k in range(1, KMAX):
        d = ifb3_k(u, k, "-", matrixizer, sources, version)
        d += ifb3_k(u, k, "+", matrixizer, sources, version)
        d += ifb3_k(u, -k, "-", matrixizer, sources, version)
        d += ifb3_k(u, -k, "+", matrixizer, sources, version)
        sum_ += d
        if np.linalg.norm(d)/(np.linalg.norm(sum_) + TINY) < EPS:
            if debug:
                return sum_, k
            else:
                return sum_
    raise RuntimeWarning("ifb3_1 did not converge in {} steps".format(KMAX))

def ifb3_k(u, k, sign, matrixizer, sources, version='old'):
    N = sources["sources_list"][0].nseg
    M = len(sources["sources_list"])
    orig_shape = (2*N*M, 2*N*M)
    m = np.zeros(orig_shape, dtype=np.float)
    if str(k)+sign not in matrixizer.m_cache["dyds_0"].keys():
        matrixizer.make_calc_matr(sources, sign, k)
    inds_dyds_0, unique_lims_dyd_0, inverse_inds_dyd_0, len_alims1, mask1, mask2 = matrixizer.m_cache["dyds_0"][str(k)+sign]
    inds_dyds_nnz, unique_lims_dyd_nnz, inverse_inds_dyd_nnz = matrixizer.m_cache["dyds_nnz"][str(k)+sign]
    su = np.sqrt(u)
    if inds_dyds_0 is not None:
        vals_dyds_0 = i_frac_0(unique_lims_dyd_0, su)
        matrixizer.reconstruct_dyds_0(m, inds_dyds_0, vals_dyds_0, inverse_inds_dyd_0, len_alims1, mask1, mask2)
    if inds_dyds_nnz is not None:
        if version == 'old':
            f = lambda t: i_frac_nnz(t, su)
            vals_dyds_nnz = np.apply_along_axis(f, 1, unique_lims_dyd_nnz)
            matrixizer.reconstruct_dyds_nnz(m, inds_dyds_nnz, vals_dyds_nnz, inverse_inds_dyd_nnz)
        elif version == 'new':
            m = cy_reconstruct_dyd_nnz(su, m, inds_dyds_nnz, unique_lims_dyd_nnz, inverse_inds_dyd_nnz)
    return m

def ifb3_2(u, arg_x_0, arg_y_4, xed):
    su = u**0.5
    m = -0.5*np.pi/xed/su*arg_x_0*np.exp(-su*arg_y_4)
    return m

def calc_sexp(ek, yed):
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