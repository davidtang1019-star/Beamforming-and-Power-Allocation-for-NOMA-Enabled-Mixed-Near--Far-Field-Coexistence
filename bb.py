import os
os.environ['OMP_NUM_THREADS'] = '1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np, time, pickle, sys
from run import *

def bb_opt(inn, K, tol=0.02, maxit=400):
    """global optimum of the CVaR-SAA problem over x (monotonic BB, Ding-style)"""
    f = lambda x: np.sum(np.log2(1+x))
    G = inn.G; asg = inn.asg
    xmax = np.array([G[k][:, asg[k]].max()*1.0 for k in range(K)])
    act = np.ones(K, bool)
    def feas(x):
        if np.all(x <= 0): return True
        a = x > 0
        return inn.feasible(a, 1/x[a]) is not None
    boxes = [(np.zeros(K), xmax, f(np.zeros(K)), f(xmax))]
    LB = 0.0; xbest = np.zeros(K); it = 0; nlp = 0
    while it < maxit:
        UB = max(b[3] for b in boxes)
        if UB - LB <= tol: break
        i = int(np.argmax([b[3] for b in boxes])); a, b, _, _ = boxes.pop(i)
        e = int(np.argmax(b - a)); mid = 0.5*(a[e] + b[e])
        b1 = b.copy(); b1[e] = mid
        a2 = a.copy(); a2[e] = mid
        boxes.append((a, b1, f(a), f(b1)))
        nlp += 1
        if feas(a2):
            boxes.append((a2, b, f(a2), f(b)))
            if f(a2) > LB: LB = f(a2); xbest = a2.copy()
        boxes = [bx for bx in boxes if bx[3] >= LB - 1e-12]
        it += 1
    UB = max([b[3] for b in boxes] + [LB])
    return LB, UB, xbest, it, nlp

def one(seed, K=2):
    S = Sys(K=K)
    rng = np.random.default_rng(seed); drp = S.drop(rng)
    P = S.zf_beams(drp['rbar'], drp['uNF']); on = legacy_mask(S, P, drp)
    (LN, LF, _), _ = outer_search(S, drp, P, on, 'cc')
    post = acquire(S, drp, LN, LF, np.random.default_rng(seed+99), P)
    rs, us = samples(S, post, 'cc')
    Hg = gains_nf(S, P, rs, drp['uNF']); G = gains_ff(S, P, drp['rk'], us)
    asg = match_hungarian(proxy_matrix(S, Hg, G, S.epsNF, S.epsFF, on))
    inn = Inner(S, Hg, G, asg, S.epsNF, S.epsFF, S.beta, on)
    t = time.time(); d = inn.solve(); tsca = time.time()-t
    t = time.time(); LB, UB, xb, it, nlp = bb_opt(inn, K); tbb = time.time()-t
    return dict(sca=d['hist'][-1], hist=d['hist'], LB=LB, UB=UB, tsca=tsca, tbb=tbb, it=it, nlp=nlp)

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    res = []
    for s in range(n):
        r = one(1000+s); res.append(r); print(s, {k: (np.round(v, 3) if k != 'hist' else len(v)) for k, v in r.items()}, flush=True)
        pickle.dump(res, open('data/bb.pkl', 'wb'))
