import numpy as np
from .integrators import frac_inf, frac_nnnn

def integrate_sources_for_green_matrix(u, matrixizer, sources):
    wtype = sources["wtype"]
    outer_bound = sources["boundaries"]["outer_bound"]
    if wtype == "frac":
        if outer_bound == "inf":
            m = frac_inf.integrate_matrix(u, matrixizer, sources)
        elif outer_bound == "nnnn":
            m = frac_nnnn.integrate_matrix(u, matrixizer, sources)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return m

def integrate_matrix_for_xd_yd(u, xd, yd, buf, matrixizer, sources):
    wtype = sources["wtype"]
    outer_bound = sources["boundaries"]["outer_bound"]
    _, xjs_, xj1s_, yws_, _, _zws, _zds = matrixizer.raw
    xjs = xjs_[0,...]
    xj1s = xj1s_[0,...]
    yws = yws_[0,...]
    xis = xd*np.ones_like(xjs, dtype=np.float)
    yds = yd*np.ones_like(xjs, dtype=np.float)
    if wtype == "frac":
        if outer_bound == "inf":
            m = frac_inf.integrate_matrix_(u, buf, xis, xjs, xj1s, yws, yds, _zws, _zds)
        elif outer_bound == "nnnn":
            xed = sources["sources_list"][0].xed
            yed = sources["sources_list"][0].yed
            m = frac_nnnn.integrate_matrix_(u, buf, buf,
                xed, yed, xis, xjs, xj1s, yws, yds, _zws, _zds)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return m

