import numpy as np, time, sys, pickle
from core import *

LF_GRID = [0, 16, 32, 48, 64, 96, 128, 160, 192, 256]

def design(S, drp, P, on, post, mode, matching='hung'):
    if mode == 'cc':
        rs, us = samples(S, post, 'cc'); eN, eF = S.epsNF, S.epsFF
    elif mode == 'wc':
        rs, us = samples(S, post, 'wc'); eN, eF = 2.0/rs.shape[1], 1.0/us.shape[1]
    elif mode == 'nom':
        rs, us = samples(S, post, 'nom'); eN, eF = 1.0, 1.0
    elif mode == 'truth':
        rs, us = samples(S, post, 'truth'); eN, eF = 1.0, 1.0
    Hg = gains_nf(S, P, rs, drp['uNF']); G = gains_ff(S, P, drp['rk'], us)
    if matching == 'hung':
        qN, qF = (0.0, 0.0) if mode == 'wc' else (min(eN, 0.5), min(eF, 0.5))
        asg = match_hungarian(proxy_matrix(S, Hg, G, qN, qF, on))
    else:
        asg = match_ding(S, Hg, G, on)
    inn = Inner(S, Hg, G, asg, eN, eF, S.beta, on)
    return inn.solve()

def outer_search(S, drp, P, on, mode='cc', LN_set=None, LF_set=None):
    """predict-then-acquire: choose (LN,LF) from prior (statistical) information only"""
    LN_set = [0, S.M] if LN_set is None else LN_set
    LF_set = LF_GRID if LF_set is None else LF_set
    M, K = S.M, S.K
    best = None; table = []
    zN, zF = (S.zN, S.zF) if mode == 'cc' else (np.linspace(-3, 3, 41), np.linspace(-3, 3, 61))
    qN, qF = (S.epsNF, S.epsFF) if mode == 'cc' else (0.0, 0.0)
    cacheN = {}
    for LN in LN_set:
        sdR = np.array([post_sd_r(S, drp['rbar'][m], drp['uNF'][m], LN) for m in range(M)])
        rs = drp['rbar'][:, None] + sdR[:, None]*zN[None, :]
        cacheN[LN] = gains_nf(S, P, rs, drp['uNF'])
    for LF in LF_set:
        if 0 < LF < K: continue
        sdU = np.array([post_sd_u(S, drp['rk'][k], drp['ubar'][k], LF) for k in range(K)])
        us = drp['ubar'][:, None] + sdU[:, None]*zF[None, :]
        G = gains_ff(S, P, drp['rk'], us)
        for LN in LN_set:
            Rp = proxy_matrix(S, cacheN[LN], G, qN, qF, on)
            asg = match_hungarian(Rp)
            Rhat = np.maximum(Rp[asg, np.arange(K)], 0).sum()
            L = LN + LF
            val = (1 - L/S.T)*(Rhat + on.sum()*S.R) - on.sum()*S.R
            table.append((LN, LF, val))
            if best is None or val > best[2]: best = (LN, LF, val)
    return best, table

def run_drop(S, seed, schemes, Lfull=128):
    rng = np.random.default_rng(seed)
    drp = S.drop(rng)
    P = S.zf_beams(drp['rbar'], drp['uNF']); on = legacy_mask(S, P, drp)
    out = {}
    for sch in schemes:
        r2 = np.random.default_rng(seed*7919 + 13)    # common random numbers across schemes
        t0 = time.time()
        if sch == 'genie':
            post = acquire(S, drp, 0, 0, r2, P)
            d = design(S, drp, P, on, post, 'truth')
            ev = evaluate(S, drp, post, d, P, r2, truth=True); L = 0
        else:
            if sch in ('prop', 'wc'):
                (LN, LF, _), _ = outer_search(S, drp, P, on, 'cc' if sch == 'prop' else 'wc')
            elif sch in ('nom', 'ccfix'):
                LN, LF = S.M, Lfull
            elif sch == 'stat':
                LN, LF = 0, 0
            elif sch == 'nomopt':     # nominal design but with optimised pilots (ablation)
                (LN, LF, _), _ = outer_search(S, drp, P, on, 'cc')
            post = acquire(S, drp, LN, LF, r2, P)
            mode = {'prop': 'cc', 'wc': 'wc', 'nom': 'nom', 'ccfix': 'cc', 'stat': 'cc', 'nomopt': 'nom'}[sch]
            d = design(S, drp, P, on, post, mode, 'ding' if sch == 'nom' else 'hung')
            ev = evaluate(S, drp, post, d, P, r2); L = LN + LF
        out[sch] = dict(net=net_gain(S, ev, L, drp), L=L, Rff=ev['Rff'], outFF=ev['outFF'],
                        outNF=ev['outNF'], nfrate=ev['nfrate'], non=int(on.sum()), fsic=ev['fsic'],
                        it=len(d['hist'])-1, t=time.time()-t0, hist=d['hist'],
                        nadm=int((d['x'] > 1e-6).sum()))
    return out

if __name__ == '__main__':
    S = Sys()
    for seed in range(3):
        t = time.time(); o = run_drop(S, seed, ['genie', 'prop', 'wc', 'nom', 'ccfix', 'stat'])
        for k, v in o.items(): print(seed, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != 'hist'})
        print('time', time.time()-t)
