import numpy as np
from .common import i_frac_0, i_frac_nnz, make_calc_matr_


def integrate_matrix(u, matrixizer, sources):
    xed = sources["sources_list"][0].xed
    yed = sources["sources_list"][0].yed
    xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_ = matrixizer.raw
    m = integrate_matrix_(u, matrixizer.m_cache, matrixizer.m_cache, xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_id='')
    return m

def integrate_matrix_(u, buf_k, buf_fb_1_2, xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_id=''):
    if "fb_1_2_nnnn"+xd_id not in buf_fb_1_2.keys():
        make_matr_for_fb_1_2_nnnn_(buf_fb_1_2, xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_id)
    arg_x_0, arg_x_1, arg_x_2, arg_x_3, arg_y_1, arg_y_2, arg_y_3, arg_y_4 = buf_fb_1_2["fb_1_2_nnnn"+xd_id]
    i1 = ifb1(u, arg_x_0, arg_y_1, arg_y_2, arg_y_3, arg_y_4, xed, yed)
    i2 = ifb2(u, arg_x_1, arg_x_2, arg_x_3, arg_y_1, arg_y_2, arg_y_3, arg_y_4, xed, yed)
    i3 = ifb3_(buf_k, u, xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, arg_x_0, arg_y_4, xd_id)
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

def ifb3_(buf, u, xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, arg_x_0, arg_y_4, xd_yd_id=''):
    a = ifb3_1_(buf, u, xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_yd_id)
    b = ifb3_2(u, arg_x_0, arg_y_4, xed)
    return a + b

def ifb3_1_(buf, u, xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_yd_id = '', debug=False):
    KMAX = 100
    EPS = 1e-12
    TINY = 1e-20
    sum_ = ifb3_k_(buf, u, 0, "-", xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_yd_id)
    sum_ += ifb3_k_(buf, u, 0, "+", xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_yd_id)
    for k in range(1, KMAX):
        d = ifb3_k_(buf, u, k, "-", xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_yd_id)
        d += ifb3_k_(buf, u, k, "+", xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_yd_id)
        d += ifb3_k_(buf, u, -k, "-", xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_yd_id)
        d += ifb3_k_(buf, u, -k, "+", xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_yd_id)
        sum_ += d
        if np.linalg.norm(d)/(np.linalg.norm(sum_) + TINY) < EPS:
            if debug:
                return sum_, k
            else:
                return sum_
    raise RuntimeWarning("ifb3_1 did not converge in {} steps".format(KMAX))

def ifb3_k_(buf, u, k, sign, xed, yed, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_yd_id=''):
    orig_shape = xis_.shape
    m = np.zeros(orig_shape, dtype=np.float)
    if str(k)+sign not in buf["dyds_0"].keys():
        make_calc_matr_(buf, xed, yed, sign, k, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_, xd_yd_id)
    inds_dyds_0, unique_lims_dyd_0, inverse_inds_dyd_0, len_alims1, mask1, mask2 = buf["dyds_0"][str(k)+sign]
    inds_dyds_nnz, unique_lims_dyd_nnz, inverse_inds_dyd_nnz = buf["dyds_nnz"][str(k)+sign]
    su = np.sqrt(u)
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
        m = m.reshape(-1)
        m[inds_dyds_nnz] = vals_dyds_nnz[inverse_inds_dyd_nnz]
        m = m.reshape(orig_shape)
    return m

def ifb3_2(u, arg_x_0, arg_y_4, xed):
    su = u**0.5
    return -0.5*np.pi/xed/su*arg_x_0*np.exp(-su*arg_y_4)

def make_matr_for_fb_1_2_nnnn_(buf, xed, yed, xis, xjs, xj1s, yws, yds, zws, zds, xd_id=''):
    yd1_w = yed - np.abs(yds-yws)
    yd2_w = yed - (yds+yws)
    arg_x_0 = xj1s - xjs
    arg_x_1 = np.pi*xis/xed
    arg_x_2 = np.pi/2./xed*arg_x_0
    arg_x_3 = np.pi/2./xed*(2*xis - (xjs + xj1s))
    arg_y_1 = yds+yws
    arg_y_2 = yed+yd1_w
    arg_y_3 = yed+yd2_w
    arg_y_4 = np.abs(yds-yws)
    buf["fb_1_2_nnnn"+xd_id] = (arg_x_0, arg_x_1, arg_x_2, arg_x_3, arg_y_1, arg_y_2, arg_y_3, arg_y_4)

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