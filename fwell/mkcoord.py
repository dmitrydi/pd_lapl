import numpy as np
from scipy.linalg import block_diag

def make_coordinates_(xwds, ywds, N, nwells):
    xd = None
    for xwd_ in xwds:
        if xd is None:
            xd = np.tile(np.linspace(xwd_-1+0.5/N, xwd_+1-0.5/N, 2*N).reshape(-1,1), (1,2*N*nwells))
        else:
            xd = np.vstack([xd, np.tile(np.linspace(xwd_-1+0.5/N, xwd_+1-0.5/N, 2*N).reshape(-1,1), (1,2*N*nwells))])
    xwd = None
    for xwd_ in xwds:
        if xwd is None:
            xwd = xwd_*np.ones((2*N*nwells, 2*N))
        else:
            xwd = np.hstack([xwd, xwd_*np.ones((2*N*nwells, 2*N))])
    ywd = None
    for ywd_ in ywds:
        if ywd is None:
            ywd = ywd_*np.ones((2*N*nwells, 2*N))
        else:
            ywd = np.hstack([ywd, ywd_*np.ones((2*N*nwells, 2*N))])
    yd = ywd.T
    x1 = np.tile(np.arange(-N,N)/N, (2*N*nwells,nwells))
    x2 = x1 + 1/N
    return xd, xwd, yd, ywd, x1, x2

def make_dummy_matr_(N, M):
    m = np.zeros((1+2*N*M, 1+2*N*M), dtype = np.float)
    m[:-1,0] = 1.
    m[-1, 1:] = 1.
    return m

def make_source_matr_(N,M,Fcd,wtype):
    m = np.zeros((1+2*N*M, 1+2*N*M))
    m_ = np.zeros((N,N))
    dx = 1./N
    dx2_8 = 0.125*dx**2
    dx2_2 = 0.5*dx**2
    if wtype == "frac":
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

def make_dummy_rhv_(N,M,Fcd,wtype):
    b_ = np.zeros((2*N))
    dx = 1./N
    rhv = np.array([], dtype = np.float)
    if wtype == "frac":
        coef = np.pi * dx/Fcd/M
        for i in range(N):
            b_[N+i] = (i+0.5)
            b_[N-i-1] = (i+0.5)
        b_ *= coef
        for _ in range(M):
            rhv = np.append(rhv, b_)
        rhv = np.append(rhv, 2./dx)
    return rhv
