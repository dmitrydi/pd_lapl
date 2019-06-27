import numpy as np

def calc_stehf_coef(n):
    v = np.array(range(n+1), dtype = np.float)
    g = np.array(range(161), dtype = np.float)
    h = np.array(range(81), dtype = np.float)
    g[1] = 1
    NH = n // 2
    for i in range(2, n+1):
        g[i] = g[i-1]*i
    h[1] = 2./ g[NH - 1]
    for i in range(2, NH+1):
        fi = i
        if i != NH:
            h[i] = (fi ** NH)*g[2*i]/(g[NH - i]*g[i]*g[i - 1])
        else:
            h[i] = (fi ** NH)*g[2*i]/(g[i]*g[i - 1])
    SN = 2 * (NH - (NH // 2)*2) - 1
    for i in range(1, n+1):
        v[i] = 0.
        K1 = (i + 1) // 2
        K2 = i
        if K2 > NH:
            K2 = NH
        for k in range(K1, K2 + 1):
            if 2*k - i == 0:
                v[i] += h[k]/g[i - k]
            elif i == k:
                v[i] += h[k]/g[2*k - i]
            else:
                v[i] += h[k]/(g[i - k]*g[2*k - i])
        v[i] *= SN
        SN = -1*SN
    return v

def lapl_invert(f, t, stehf_coefs):
    ans = 0.
    N = len(stehf_coefs)
    for i in range(1, N):
        p = i * np.log(2.)/t
        f_ = f(p)
        ans += f_*stehf_coefs[i]*p/i
    return ans

def make_offset(matr):
    sh = matr.shape
    m = np.zeros((1+sh[0], 1+sh[1]))
    m[:-1, 1:] = matr
    return m
