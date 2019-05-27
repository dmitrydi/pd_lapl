import numpy as np
from scipy.linalg import block_diag

class Matrixizer():
    def __init__(self, sources):
        self.raw = self.make_raw_matr(sources)
        self.dummy_m = self.make_dummy_matr(sources)
        self.source_m = self.make_source_matr(sources)
        self.dummy_rhv = self.make_dummy_rhv(sources)
        self.m_cache = {"dyds_0": {}, "dyds_nnz": {}}
        self.v_cache = {}

    def make_raw_matr(self, sources):
        sources_list = sources["sources_list"]
        wtype = sources["wtype"]
        outer_bound = sources["boundaries"]["outer_bound"]
        M = len(sources_list)
        N = sources_list[0].nseg
        orig_shape = 2*N*M
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

    def make_dummy_matr(self, sources):
        N = sources["sources_list"][0].nseg
        M = len(sources["sources_list"])
        m = np.zeros((1+2*N*M, 1+2*N*M), dtype = np.float)
        m[:-1,0] = 1.
        m[-1, 1:] = 1.
        return m

    def make_source_matr(self, sources):
        sources_list = sources["sources_list"]
        wtype = sources["wtype"]
        attrs = sources["attrs"]
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

    def make_dummy_rhv(self, sources):
        sources_list = sources["sources_list"]
        wtype = sources["wtype"]
        attrs = sources["attrs"]
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

    # def make_calc_matr(self, sources, sign, k):
    #     if sign == "+":
    #         s_ = 1
    #     elif sign == "-":
    #         s_ = -1
    #     else:
    #         raise ValueError
    #     xed = sources["sources_list"][0].xed
    #     yed = sources["sources_list"][0].yed
    #     wtype = sources["wtype"]
    #     xis, xjs, xj1s, yws, yds, zws, zds = self.raw
    #     orig_shape = xis.shape
    #     xis = xis.flatten()
    #     xjs = xjs.flatten()
    #     xj1s = xj1s.flatten()
    #     yws = yws.flatten()
    #     yds = yds.flatten()
    #     zws = zws.flatten()
    #     zds = zds.flatten()
    #     if wtype == "frac":
    #         dyds = np.round(np.abs(yds - yws), decimals = NDIGITS)
    #         lims1 = np.round(xjs + s_*xis - 2*k*xed, decimals = NDIGITS)
    #         lims2 = np.round(xj1s + s_*xis - 2*k*xed, decimals = NDIGITS)
    #         #dlims = lims2 - lims1
    #         alims1 = np.abs(lims1)
    #         alims2 = np.abs(lims2)
    #         inds_dyds_nnz = np.argwhere(dyds != 0.).flatten()
    #         inds_dyds_0 = np.argwhere(dyds == 0.).flatten()
    #         # deal with dyds == 0:
    #         if len(inds_dyds_0) > 0:
    #             mask1 = 1 - 2*(lims1[inds_dyds_0] > 0.)
    #             mask2 = 1 - 2*(lims2[inds_dyds_0] < 0.)
    #             len_alims1 = len(alims1[inds_dyds_0])
    #             all_lims_dyd_0 = np.concatenate((alims1[inds_dyds_0], alims2[inds_dyds_0]))
    #             unique_lims_dyd_0, inverse_inds_dyd_0 = np.unique(all_lims_dyd_0, return_inverse = True)
    #             self.m_cache["dyds_0"][str(k) + sign] = (inds_dyds_0, unique_lims_dyd_0, inverse_inds_dyd_0, len_alims1, mask1, mask2)
    #         else:
    #             self.m_cache["dyds_0"][str(k) + sign] = (None, None, None, None, None, None)
    #         # deal with dyds != 0:
    #         if len(inds_dyds_nnz) > 0:
    #             alims1_nnz, alims2_nnz = alims1[inds_dyds_nnz], alims2[inds_dyds_nnz]
    #             all_lims_dyd_nnz = np.vstack([alims1_nnz, alims2_nnz, lims2[inds_dyds_nnz] - lims1[inds_dyds_nnz]]).T # (|lims1|, |lims2|, lims2 - lims1)
    #             upper_lims = np.max(all_lims_dyd_nnz[:,:2], axis=1).reshape(-1, 1)
    #             upper_lims_dyds = np.hstack([upper_lims, all_lims_dyd_nnz[:, -1].reshape(-1,1), dyds[inds_dyds_nnz].reshape(-1, 1)]) #(x_upper, dx, dyd)
    #             unique_lims_dyd_nnz, inverse_inds_dyd_nnz = np.unique(upper_lims_dyds, axis=0, return_inverse=True)
    #             self.m_cache["dyds_nnz"][str(k) + sign] = (inds_dyds_nnz, unique_lims_dyd_nnz, inverse_inds_dyd_nnz)
    #         else:
    #             self.m_cache["dyds_nnz"][str(k) + sign] = (None, None, None)
    #     else:
    #             raise NotImplementedError

    # def reconstruct_dyds_0(self, m, inds_dyds_0, unique_vals, inverse_inds_dyd_0, len_alims1, mask1, mask2):
    #     orig_shape = m.shape
    #     m = m.reshape(-1)
    #     v = unique_vals[inverse_inds_dyd_0]
    #     v1 = v[:len_alims1]
    #     v2 = v[len_alims1:]
    #     m[inds_dyds_0] = v1*mask1 + v2*mask2
    #     m = m.reshape(orig_shape)

    # def reconstruct_dyds_nnz(self, m, inds_dyds_nnz, unique_vals, inverse_inds_dyd_nnz):
    #     orig_shape = m.shape
    #     m = m.reshape(-1)
    #     m[inds_dyds_nnz] = unique_vals[inverse_inds_dyd_nnz]
    #     m = m.reshape(orig_shape)

    # def make_matr_for_fb_1_2_nnnn(self, sources):
    #     xed = sources["sources_list"][0].xed
    #     yed = sources["sources_list"][0].yed
    #     xis, xjs, xj1s, yws, yds, zws, zds = self.raw
    #     yd1_w = yed - np.abs(yds-yws)
    #     yd2_w = yed - (yds+yws)
    #     arg_x_0 = xj1s - xjs
    #     arg_x_1 = np.pi*xis/xed
    #     arg_x_2 = np.pi/2./xed*arg_x_0
    #     arg_x_3 = np.pi/2./xed*(2*xis - (xjs + xj1s))
    #     arg_y_1 = yds+yws
    #     arg_y_2 = yed+yd1_w
    #     arg_y_3 = yed+yd2_w
    #     arg_y_4 = np.abs(yds-yws)
    #     self.m_cache["fb_1_2_nnnn"] = (arg_x_0, arg_x_1, arg_x_2, arg_x_3, arg_y_1, arg_y_2, arg_y_3, arg_y_4)



