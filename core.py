"""Partial-CSI robust NOMA for mixed near-/far-field coexistence.
Core models: geometry, channels, legacy ZF beams, CRB-based partial CSI,
CVaR (chance-constraint) SCA design, proxy-based pilot/matching layers, evaluation.
Units: powers normalised by Pmax, gains normalised by noise power (sigma^2 = 1)."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
import cvxpy as cp

C0 = 3e8

class Sys:
    def __init__(self, N=64, fc=28e9, M=36, K=4, Pmax_dBm=20.0, noise_dBm=-80.0,
                 R=0.1, beta=0.0, epsNF=0.01, epsFF=0.05, T=500,
                 pu_dBm=0.0, pnf_dBm=10.0, sig_r0=0.2, sig_u0=0.03, gam_th_dB=14.0,
                 rNF=(5.0, 15.0), phiNF=np.pi/3, rFF=(100.0, 150.0), phiFF=np.pi/3,
                 SN=50, SF=60):
        self.__dict__.update(locals()); del self.__dict__['self']
        self.lam = C0/fc; self.k0 = 2*np.pi/self.lam; self.d = self.lam/2
        self.delta = (np.arange(N)-(N-1)/2)*self.d
        self.D = (N-1)*self.d; self.dR = 2*self.D**2/self.lam
        self.sig2 = 10**((noise_dBm-30)/10); self.Pmax = 10**((Pmax_dBm-30)/10)
        self.snr0 = self.Pmax/self.sig2          # gain normalisation
        self.epsR = 2**R-1
        self.gam_th = 10**(gam_th_dB/10)
        self.zN = norm.ppf((np.arange(SN)+0.5)/SN)   # stratified std-normal samples
        self.zF = norm.ppf((np.arange(SF)+0.5)/SF)

    # ---------- channels ----------
    def steer(self, r, u):
        """exact spherical-wave response (unit modulus) and path loss; r,u arrays (...)"""
        r = np.asarray(r, float)[..., None]; u = np.asarray(u, float)[..., None]
        dist = np.sqrt(r**2 + self.delta**2 - 2*r*self.delta*u)
        return np.exp(-1j*self.k0*dist), (self.lam/(4*np.pi*r[..., 0]))

    def chan(self, r, u):
        a, al = self.steer(r, u)
        return al[..., None]*a

    def zf_beams(self, r, u):
        H = self.chan(r, u).T            # N x M
        G = np.linalg.inv(H.conj().T @ H)
        P = H @ G @ np.diag(1/np.sqrt(np.real(np.diag(G))))
        return P

    # ---------- partial CSI (CRB based) ----------
    def fim_nf(self, r, u, rho):
        """per-pilot Fisher information for (r,u) with unknown complex gain"""
        a, _ = self.steer(r, u)
        dist = np.sqrt(r**2 + self.delta**2 - 2*r*self.delta*u)
        dr = (r - self.delta*u)/dist; du = -r*self.delta/dist
        Dm = np.stack([-1j*self.k0*dr*a, -1j*self.k0*du*a], 1)
        Pp = np.eye(self.N) - np.outer(a, a.conj())/self.N
        return 2*rho*np.real(Dm.conj().T @ Pp @ Dm)

    def crb_u_ff(self, r, u, rho):
        F = self.fim_nf(r, u, rho)
        return 1.0/F[1, 1]               # u only (r known from large-scale info)

    def crb_r_nf(self, r, u, rho):
        F = self.fim_nf(r, u, rho)
        return np.linalg.inv(F)[0, 0]

    # ---------- drop ----------
    def drop(self, rng):
        M, K = self.M, self.K
        r = rng.uniform(*self.rNF, M); ph = rng.uniform(-self.phiNF, self.phiNF, M)
        rk = rng.uniform(*self.rFF, K); phk = rng.uniform(-self.phiFF, self.phiFF, K)
        return dict(rbar=r, uNF=np.sin(ph), rk=rk, ubar=np.sin(phk))

def legacy_mask(S, P, drp, thr=0.1):
    """legacy-feasible NF users (Ding et al.): nominal own-QoS power <= thr*Pmax"""
    h = np.abs(np.sum(S.chan(drp['rbar'], drp['uNF']).conj()*P.T, 1))**2*S.snr0
    drp['on'] = S.epsR/h <= thr
    return drp['on']

def gains_nf(S, P, rs, uNF):
    """rs: (M,Ss) samples of NF distance. returns Hg (M,Ss,M) = |h_m(r_s)^H p_j|^2 * snr0"""
    M, Ss = rs.shape
    out = np.empty((M, Ss, P.shape[1]))
    for m in range(M):
        h = S.chan(rs[m], np.full(Ss, uNF[m]))          # Ss x N
        out[m] = np.abs(h.conj() @ P)**2*S.snr0
    return out

def gains_ff(S, P, rk, us):
    """us: (K,Ss) samples of FF u. returns G (K,Ss,M)"""
    K, Ss = us.shape
    out = np.empty((K, Ss, P.shape[1]))
    for k in range(K):
        g = S.chan(np.full(Ss, rk[k]), np.clip(us[k], -0.999, 0.999))
        out[k] = np.abs(g.conj() @ P)**2*S.snr0
    return out

# ---------------- posterior / CSI acquisition ----------------
def acquire(S, drp, LN, LF, rng, P):
    """returns posterior means/stds for NF distances and FF directions, plus truth0"""
    M, K = S.M, S.K
    r0 = drp['rbar'] + S.sig_r0*rng.standard_normal(M)        # true NF distances
    u0 = np.clip(drp['ubar'] + S.sig_u0*rng.standard_normal(K), -0.95, 0.95)
    # NF
    muR = np.empty(M); sdR = np.empty(M)
    for m in range(M):
        if LN > 0:
            al2 = (S.lam/(4*np.pi*drp['rbar'][m]))**2
            rho = 10**((S.pnf_dBm-30)/10)*al2/S.sig2
            v1 = S.crb_r_nf(drp['rbar'][m], drp['uNF'][m], rho)/LN
            est = r0[m] + np.sqrt(v1)*rng.standard_normal()
            pv = 1/(1/S.sig_r0**2 + 1/v1)
            muR[m] = pv*(drp['rbar'][m]/S.sig_r0**2 + est/v1); sdR[m] = np.sqrt(pv)
        else:
            muR[m] = drp['rbar'][m]; sdR[m] = S.sig_r0
    muU = np.empty(K); sdU = np.empty(K)
    for k in range(K):
        v1 = ff_pilot_var(S, drp['rk'][k], drp['ubar'][k], LF)
        if np.isfinite(v1):
            est = u0[k] + np.sqrt(v1)*rng.standard_normal()
            pv = 1/(1/S.sig_u0**2 + 1/v1)
            muU[k] = pv*(drp['ubar'][k]/S.sig_u0**2 + est/v1); sdU[k] = np.sqrt(pv)
        else:
            muU[k] = drp['ubar'][k]; sdU[k] = S.sig_u0
    return dict(muR=muR, sdR=sdR, muU=np.clip(muU, -0.95, 0.95), sdU=sdU, r0=r0, u0=u0)

def ff_rho(S, rk):
    al2 = (S.lam/(4*np.pi*rk))**2
    return 10**((S.pu_dBm-30)/10)*al2/S.sig2

def ff_pilot_var(S, rk, u, LF):
    """variance of pilot-based u estimate; inf if below ML threshold"""
    if LF <= 0: return np.inf
    rho = ff_rho(S, rk)
    if LF*S.N*rho < S.gam_th: return np.inf
    return S.crb_u_ff(rk, u, rho)/LF

def post_sd_u(S, rk, u, LF):
    v1 = ff_pilot_var(S, rk, u, LF)
    return np.sqrt(1/(1/S.sig_u0**2 + (0 if not np.isfinite(v1) else 1/v1)))

def post_sd_r(S, r, u, LN):
    if LN <= 0: return S.sig_r0
    al2 = (S.lam/(4*np.pi*r))**2
    rho = 10**((S.pnf_dBm-30)/10)*al2/S.sig2
    v1 = S.crb_r_nf(r, u, rho)/LN
    return np.sqrt(1/(1/S.sig_r0**2 + 1/v1))

def samples(S, post, robust='cc', box=3.0):
    """sample sets for NF distance and FF direction. robust in {'cc','wc','nom','truth'}"""
    if robust == 'cc':
        zN, zF = S.zN, S.zF
    elif robust == 'wc':
        zN = np.linspace(-box, box, 41); zF = np.linspace(-box, box, 61)
    else:
        zN = np.zeros(1); zF = np.zeros(1)
    rs = post['muR'][:, None] + post['sdR'][:, None]*zN[None, :]
    us = post['muU'][:, None] + post['sdU'][:, None]*zF[None, :]
    if robust == 'truth':
        rs = post['r0'][:, None]; us = post['u0'][:, None]
    return rs, us

# ---------------- proxy (middle/outer layers) ----------------
def proxy_matrix(S, Hg, G, epsNF, epsFF, on=None):
    """proxy rate R[m,k] for FF user k riding beam m, based on quantiles"""
    M = Hg.shape[0]; K = G.shape[0]
    hsel = np.array([Hg[m, :, m] for m in range(M)])                 # M x Ss
    hq = np.quantile(hsel, epsNF, axis=1)
    leak = np.array([Hg[m].sum(1) - Hg[m, :, m] for m in range(M)])   # M x Ss (all beams full power)
    Iq = np.quantile(leak*(K/max(M-1, 1)), 1-epsNF, axis=1)
    xsic = hq/(1 + Iq)/(1+S.epsR)
    gq = np.quantile(G, epsFF, axis=1).T                               # M x K
    Rp = np.log2(1 + np.minimum(gq, xsic[:, None]))
    if on is not None: Rp[~on, :] = -1e3
    return Rp

def match_hungarian(Rp):
    rows, cols = linear_sum_assignment(-Rp.T)     # K x M
    asg = np.empty(Rp.shape[1], int); asg[rows] = cols
    return asg

def match_ding(S, Hg, G, on):
    """Ding's Remark-2 greedy criterion on nominal (single-sample) gains"""
    M = Hg.shape[0]; K = G.shape[0]
    h = np.array([Hg[m, 0, m] for m in range(M)])
    avail = [m for m in range(M) if on[m]]; asg = np.empty(K, int)
    for k in range(K):
        g = G[k, 0, avail]; hh = h[avail]
        sc = np.minimum(g/g.max(), hh/hh.max())
        j = avail[int(np.argmax(sc))]; asg[k] = j; avail.remove(j)
    return asg

# ---------------- inner CVaR-SCA ----------------
import scipy.sparse as sp

class Inner:
    """CVaR-approximated chance-constrained design, reciprocal-SINR SCA.
    Variables: P (M), z (K), v_k = 1/x_k, q_k <= z_k v_k (concave minorant)."""
    def __init__(self, S, Hg, G, asg, epsNF, epsFF, beta, on):
        self.S = S; M = Hg.shape[0]; K = G.shape[0]; self.M, self.K = M, K
        self.Hg, self.G, self.asg, self.on = Hg, G, asg, on
        self.epsNF, self.epsFF, self.beta = epsNF, epsFF, beta
        A = np.zeros((M, K)); A[asg, np.arange(K)] = 1; self.A = A
        hself = np.array([np.median(Hg[m, :, m]) for m in range(M)])
        self.Ps = min(1.0, 10*S.epsR/hself[on].min())

    def _build(self, act, vfix=None):
        """act: boolean mask of admitted FF users. vfix: fixed v (feasibility LP)"""
        S, M, K, A, Ps = self.S, self.M, self.K, self.A, self.Ps
        Hg, G, asg = self.Hg, self.G, self.asg
        Pt = cp.Variable(M, nonneg=True); z = cp.Variable(K, nonneg=True)
        P = Ps*Pt; Z = A @ z
        cons = [P + Z <= 1]
        if (~self.on).any(): cons += [Pt[~self.on] == 0]
        if (~act).any(): cons += [z[~act] == 0]
        ka = np.where(act)[0]; Ka = len(ka)
        if vfix is None:
            v = cp.Variable(Ka, pos=True); q = cp.Variable(Ka)
            self.al = cp.Parameter(Ka, pos=True); self.ali = cp.Parameter(Ka, pos=True)
            self.p1 = cp.Parameter(Ka, nonneg=True); self.p2 = cp.Parameter(Ka, nonneg=True)
            self.s0sq = cp.Parameter(Ka, nonneg=True)
            za = z[ka] if Ka else None
            if Ka:
                a_ = cp.multiply(self.al, za); b_ = cp.multiply(self.ali, v)
                cons += [q <= 0.5*(cp.multiply(self.p1, za) + cp.multiply(self.p2, v)) - 0.25*self.s0sq - 0.25*cp.square(a_ - b_)]
            qk = {k: q[i] for i, k in enumerate(ka)}
        else:
            v = vfix; qk = {k: z[k]*vfix[i] for i, k in enumerate(ka)}
        # ---- NF QoS, vectorised CVaR over (m,s)
        mon = np.where(self.on)[0]; Ss = Hg.shape[1]
        rows_B = []; hv = []; scl = []
        for m in mon:
            B = Hg[m].copy(); hm = B[:, m].copy(); B[:, m] = 0
            rows_B.append(B); hv.append(hm); scl.append(np.full(Ss, 1/S.epsR))
        Bb = sp.csr_matrix(np.vstack(rows_B)); hb = np.concatenate(hv); sc = np.concatenate(scl)
        E = sp.csr_matrix((np.ones(len(mon)*Ss), (np.arange(len(mon)*Ss), np.repeat(mon, Ss))), shape=(len(mon)*Ss, M))
        F = cp.multiply(sc, S.epsR*(1 + Bb @ (P + Z) + self.beta*cp.multiply(hb, E @ Z)) - cp.multiply(hb, E @ P))
        tQ = cp.Variable(len(mon)); eQ = cp.Variable(len(mon)*Ss, nonneg=True)
        Rep = sp.csr_matrix((np.ones(len(mon)*Ss), (np.arange(len(mon)*Ss), np.repeat(np.arange(len(mon)), Ss))))
        cons += [eQ >= F - Rep @ tQ, tQ + (Rep.T @ eQ)/(0.5*self.epsNF*Ss) <= 0]
        # ---- SIC and FF decoding, per admitted user
        for k in ka:
            m = asg[k]
            B = Hg[m].copy(); hm = B[:, m].copy(); B[:, m] = 0
            sc1 = 1/(1 + np.sqrt(np.median(hm)))
            Fs = sc1*(1 + cp.multiply(hm, P[m]) + B @ (P + Z) - hm*qk[k])
            t = cp.Variable(); e = cp.Variable(Ss, nonneg=True)
            cons += [e >= Fs - t, t + cp.sum(e)/(0.5*self.epsNF*Ss) <= 0]
            Gk = G[k]; Sf = Gk.shape[0]; gk = Gk[:, m]
            others = [i for i in ka if i != k]
            sc2 = 1/(1 + np.sqrt(max(np.median(gk), 0)))
            interf = 1 + Gk @ P
            if others: interf = interf + Gk[:, asg[others]] @ z[others]
            Ff = sc2*(interf - gk*qk[k])
            t = cp.Variable(); e = cp.Variable(Sf, nonneg=True)
            cons += [e >= Ff - t, t + cp.sum(e)/(self.epsFF*Sf) <= 0]
        return v, Pt, z, cons, ka

    @staticmethod
    def _cvar(F, eps):
        S_ = len(F); best = np.inf
        for t in F:
            best = min(best, t + np.maximum(F - t, 0).sum()/(eps*S_))
        return best

    def verify(self, P, z, x, tol=1e-6):
        """check all sample-CVaR constraints of the design (noise-normalised units)"""
        S, M, K, A, Hg, G, asg = self.S, self.M, self.K, self.A, self.Hg, self.G, self.asg
        Z = A @ z
        if np.any(P + Z > 1 + 1e-6): return False
        for m in np.where(self.on)[0]:
            B = Hg[m].copy(); hm = B[:, m].copy(); B[:, m] = 0
            F = S.epsR*(1 + B @ (P + Z) + self.beta*hm*Z[m]) - hm*P[m]
            if self._cvar(F, 0.5*self.epsNF) > tol*S.epsR*(1 + np.mean(B @ (P + Z))): return False
        for k in range(K):
            if x[k] <= 0: continue
            m = asg[k]; B = Hg[m].copy(); hm = B[:, m].copy(); B[:, m] = 0
            den = 1 + hm*P[m] + B @ (P + Z)
            if self._cvar(x[k]*den - hm*z[k], 0.5*self.epsNF) > tol*x[k]*np.mean(den): return False
            Gk = G[k]; den = 1 + Gk @ P + sum(Gk[:, asg[i]]*z[i] for i in range(K) if i != k and x[i] > 0)
            if self._cvar(x[k]*den - Gk[:, m]*z[k], self.epsFF) > tol*x[k]*np.mean(den): return False
        return True

    def feasible(self, act, vfix):
        v, Pt, z, cons, ka = self._build(act, vfix=vfix)
        pr = cp.Problem(cp.Minimize(cp.sum(Pt) - 1e-3*cp.sum(z)), cons)
        try: pr.solve(solver=cp.CLARABEL, max_iter=300, time_limit=30.0)
        except Exception: return None
        if pr.status != 'optimal' or Pt.value is None: return None
        Pv, zv = np.maximum(Pt.value, 0), np.maximum(z.value, 0)
        xv = np.zeros(self.K); xv[act] = 1/np.asarray(vfix)
        if not self.verify(Pv*self.Ps, zv, xv*(1-1e-6), tol=1e-5): return None
        return Pv, zv

    def solve(self, x0=0.05, maxit=30, tol=1e-4):
        K = self.K
        act = np.ones(K, bool)
        # admission control: drop users (lowest index first among infeasible) until feasible
        sol = self.feasible(act, np.full(K, 1/x0))
        if sol is None:
            x0 = x0/20
            sol = self.feasible(act, np.full(K, 1/x0))
        if sol is None:
            act[:] = False
            for k in range(K):
                a2 = act.copy(); a2[k] = True
                if self.feasible(a2, np.full(a2.sum(), 1/x0)) is not None: act[k] = True
            if act.any():
                sol = self.feasible(act, np.full(act.sum(), 1/x0))
                while sol is None and act.any():
                    act[np.where(act)[0][-1]] = False
                    sol = self.feasible(act, np.full(act.sum(), 1/x0))
        x = np.zeros(K)
        if not act.any() or sol is None:
            sol = self.feasible(np.zeros(K, bool), np.zeros(0))
            Pt0 = sol[0] if sol is not None else np.zeros(self.M)
            return dict(x=x, P=Pt0*self.Ps, z=np.zeros(K), asg=self.asg, hist=[0.0], act=act)
        Pt0, z0 = sol
        ka = np.where(act)[0]
        v0 = np.full(len(ka), 1/x0)
        v, Pt, z, cons, _ = self._build(act)
        wpar = cp.Parameter(len(ka), nonneg=True)
        pr = cp.Problem(cp.Minimize(wpar @ v), cons)
        f = lambda vv: np.sum(np.log2(1 + 1/vv))
        hist = [f(v0)]
        for it in range(maxit):
            zz = np.maximum(z0[ka], 1e-12)
            al = np.sqrt(v0/zz); a0 = al*zz; s0 = 2*a0
            self.al.value = al; self.ali.value = 1/al; self.p1.value = s0*al; self.p2.value = s0/al; self.s0sq.value = s0**2
            wpar.value = (1/(v0*(v0+1)))/np.log(2)          # -f'(v0)
            try: pr.solve(solver=cp.CLARABEL, max_iter=300, time_limit=60.0)
            except Exception: break
            if pr.status not in ('optimal', 'optimal_inaccurate') or v.value is None: break
            vn = np.maximum(v.value, 1e-6); val = f(vn)
            if val < hist[-1] - 1e-6: break
            xt = np.zeros(K); xt[ka] = 1/vn
            if not self.verify(np.maximum(Pt.value, 0)*self.Ps, np.maximum(z.value, 0), xt*(1-1e-6), tol=1e-5):
                break
            v0, Pt0, z0 = vn, np.maximum(Pt.value, 0), np.maximum(z.value, 0)
            hist.append(val)
            if abs(hist[-1]-hist[-2]) <= tol*max(1, abs(hist[-1])): break
        x[ka] = 1/v0
        return dict(x=x, P=Pt0*self.Ps, z=z0, asg=self.asg, hist=hist, act=act)

# ---------------- evaluation ----------------
def evaluate(S, drp, post, des, P, rng, ndraw=200, truth=False):
    """expected FF sum-rate and NF QoS satisfaction over posterior draws."""
    M, K = S.M, S.K
    if truth:
        rs = post['r0'][:, None]; us = post['u0'][:, None]; nd = 1
    else:
        rs = post['muR'][:, None] + post['sdR'][:, None]*rng.standard_normal((M, ndraw))
        us = post['muU'][:, None] + post['sdU'][:, None]*rng.standard_normal((K, ndraw))
        nd = ndraw
    Hg = gains_nf(S, P, rs, drp['uNF']); G = gains_ff(S, P, drp['rk'], us)
    x, Pm, z, asg = des['x'], des['P'], des['z'], des['asg']
    A = np.zeros((M, K)); A[asg, np.arange(K)] = 1; Z = A @ z
    tol = 1 - 1e-3
    hself = np.stack([Hg[m, :, m] for m in range(M)])                 # M x nd
    I = np.stack([Hg[m] @ (Pm + Z) - Hg[m, :, m]*(Pm[m]+Z[m]) for m in range(M)])
    qos = Pm[:, None]*hself/(1 + I + S.beta*hself*Z[:, None]) >= S.epsR*tol
    ffok = np.zeros((K, nd), bool); sicok = np.ones((K, nd), bool)
    for k in range(K):
        m = asg[k]
        sic = z[k]*hself[m]/(1 + Pm[m]*hself[m] + I[m])
        sicok[k] = sic >= x[k]*tol
        sig = G[k][:, m]*z[k]
        intf = 1 + G[k] @ Pm + sum(G[k][:, asg[i]]*z[i] for i in range(K) if i != k)
        ffok[k] = sig/intf >= x[k]*tol
        qos[m] &= sicok[k] | (x[k] <= 0)
    bott = []
    for k in range(K):
        if x[k] > 1e-6 and nd > 1:
            m = asg[k]
            sic = z[k]*hself[m]/(1 + Pm[m]*hself[m] + I[m])
            sig = G[k][:, m]*z[k]
            intf = 1 + G[k] @ Pm + sum(G[k][:, asg[i]]*z[i] for i in range(K) if i != k)
            bott.append(np.quantile(sic, S.epsNF) < np.quantile(sig/intf, S.epsFF))
    qos = qos[drp['on']]
    rate = (np.log2(1+x)[:, None]*(ffok & sicok)).sum(0)                # nd
    return dict(Rff=rate.mean(), outFF=1-((ffok & sicok) | (x[:, None] <= 1e-6)).mean(),
                outNF=1-qos.mean(), nfrate=(qos.sum(0)*S.R).mean(),
                fsic=(np.mean(bott) if bott else np.nan))

def net_gain(S, ev, L, drp):
    return (1 - L/S.T)*(ev['Rff'] + ev['nfrate']) - drp['on'].sum()*S.R
