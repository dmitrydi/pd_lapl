import numpy as np
from scipy.special import k0, iti0k0
from scipy.integrate import quad
NDIGITS = 6

def i_frac_0(x, su):
    # x = np.array([vals])
    return 0.5/su*iti0k0(x*su)[1]

def i_frac_nnz(x, su):
    upper_lim, dx, dyd = x[0], x[1], x[-1]
    return quad(frac_fun, su*(upper_lim-dx), su*upper_lim, args=(dyd, su))[0]

def frac_fun(x, dyd, su):
    return 0.5/su*k0((x*x + su*su*dyd*dyd)**0.5)

def make_calc_matr_(buf, xed, yed, sign, k, xis_, xjs_, xj1s_, yws_, yds_, zws_, zds_):
    if sign == "+":
        s_ = 1
    elif sign == "-":
        s_ = -1
    else:
        raise ValueError
    orig_shape = xis_.shape
    xis = xis_.flatten()
    xjs = xjs_.flatten()
    xj1s = xj1s_.flatten()
    yws = yws_.flatten()
    yds = yds_.flatten()
    zws = zws_.flatten()
    zds = zds_.flatten()
    dyds = np.round(np.abs(yds - yws), decimals = NDIGITS)
    lims1 = np.round(xjs + s_*xis - 2*k*xed, decimals = NDIGITS)
    lims2 = np.round(xj1s + s_*xis - 2*k*xed, decimals = NDIGITS)
    #dlims = lims2 - lims1
    alims1 = np.abs(lims1)
    alims2 = np.abs(lims2)
    inds_dyds_nnz = np.argwhere(dyds != 0.).flatten()
    inds_dyds_0 = np.argwhere(dyds == 0.).flatten()
    # deal with dyds == 0:
    if len(inds_dyds_0) > 0:
        mask1 = 1 - 2*(lims1[inds_dyds_0] > 0.)
        mask2 = 1 - 2*(lims2[inds_dyds_0] < 0.)
        len_alims1 = len(alims1[inds_dyds_0])
        all_lims_dyd_0 = np.concatenate((alims1[inds_dyds_0], alims2[inds_dyds_0]))
        unique_lims_dyd_0, inverse_inds_dyd_0 = np.unique(all_lims_dyd_0, return_inverse = True)
        buf["dyds_0"][str(k) + sign] = (inds_dyds_0, unique_lims_dyd_0, inverse_inds_dyd_0, len_alims1, mask1, mask2)
    else:
        buf["dyds_0"][str(k) + sign] = (None, None, None, None, None, None)
    # deal with dyds != 0:
    if len(inds_dyds_nnz) > 0:
        alims1_nnz, alims2_nnz = alims1[inds_dyds_nnz], alims2[inds_dyds_nnz]
        all_lims_dyd_nnz = np.vstack([alims1_nnz, alims2_nnz, lims2[inds_dyds_nnz] - lims1[inds_dyds_nnz]]).T # (|lims1|, |lims2|, lims2 - lims1)
        upper_lims = np.max(all_lims_dyd_nnz[:,:2], axis=1).reshape(-1, 1)
        upper_lims_dyds = np.hstack([upper_lims, all_lims_dyd_nnz[:, -1].reshape(-1,1), dyds[inds_dyds_nnz].reshape(-1, 1)]) #(x_upper, dx, dyd)
        unique_lims_dyd_nnz, inverse_inds_dyd_nnz = np.unique(upper_lims_dyds, axis=0, return_inverse=True)
        buf["dyds_nnz"][str(k) + sign] = (inds_dyds_nnz, unique_lims_dyd_nnz, inverse_inds_dyd_nnz)
    else:
        buf["dyds_nnz"][str(k) + sign] = (None, None, None)