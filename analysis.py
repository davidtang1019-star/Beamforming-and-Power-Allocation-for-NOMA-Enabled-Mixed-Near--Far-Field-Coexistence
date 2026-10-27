import numpy as np
from scipy.special import fresnel
from scipy.stats import norm
from style import *
from core import Sys

lam = 3e8/28e9; k0 = 2*np.pi/lam

def E(t):
    S_, C_ = fresnel(t); return C_ - 1j*S_

def G_fresnel(nu, F, N):
    return N/(2*F)*np.abs(E(np.sqrt(2*F)*(nu+0.5)) - E(np.sqrt(2*F)*(nu-0.5)))**2

def G_exact(N, r, phi, u, rk=1e5):
    """|a(u)^H pbar|^2 with exact spherical NF beam and (planar-limit) FF steering"""
    d = lam/2; dl = (np.arange(N)-(N-1)/2)*d
    up = np.sin(phi)
    h = np.exp(-1j*k0*np.sqrt(r**2 + dl**2 - 2*r*dl*up)); p = h/np.linalg.norm(h)
    a = np.exp(1j*k0*np.outer(u, dl))      # planar FF response e^{+jk delta u}
    return np.abs(a.conj() @ p)**2

KAP = 0.78
def Fstar(N, q, s):
    """edge-corrected optimal focusing index (Corollary 1)"""
    return (KAP/np.sqrt(2) + np.sqrt(KAP**2/2 + N*q*s))**2

def Fidx(N, r, phi=0.0):
    return N**2*lam*np.cos(phi)**2/(4*r)

# ---------- Fig: leakage pattern ----------
def fig_leak():
    N = 128; phi = 0.0
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    nu = np.linspace(-1.5, 1.5, 3001)
    for i, Fv in enumerate([1, 4, 16]):
        r = N**2*lam/(4*Fv); Wd = N*lam/(2*r)
        ge = G_exact(N, r, phi, nu*Wd)
        ax.plot(nu, 10*np.log10(ge/N), color=COL[i], lw=1.0, label=fr'Exact, $F={Fv}$ ($r={r:.1f}$ m)')
        ax.plot(nu[::60], 10*np.log10(G_fresnel(nu[::60], Fv, N)/N), 'o', mfc='none', color=COL[i], ms=3.2)
        ax.plot([-0.5, -0.5, 0.5, 0.5], [-45, -10*np.log10(Fv), -10*np.log10(Fv), -45], '--', color=COL[i], lw=0.8)
    ax.plot([], [], 'ko', mfc='none', ms=3.2, label='Lemma 1 (Fresnel)')
    ax.plot([], [], 'k--', lw=0.8, label='Proposition 1 (flat-top)')
    ax.set_xlabel(r'Normalised angular offset $\nu=(u-u_{\phi})/W$'); ax.set_ylabel(r'$\mathcal{G}(u)/N$ (dB)')
    ax.set_ylim(-30, 2); ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-34, 12); ax.legend(loc='upper center', ncol=2, fontsize=5.8, columnspacing=0.8)
    fig.savefig('figs/fig_leak.pdf'); plt.close(fig)

# ---------- robust single-beam rate vs focusing index ----------
def robust_rate(N, r, s, eps=0.05, Pmax_dBm=20, rk=120.0, nsamp=4000, mu_off=0.0, phi=0.0):
    sig2 = 1e-11; P = 10**((Pmax_dBm-30)/10)
    alk2 = (lam/(4*np.pi*rk))**2
    z = norm.ppf((np.arange(nsamp)+0.5)/nsamp)
    u = np.sin(phi) + mu_off + s*z
    g = G_exact(N, r, phi, u)
    ge = np.quantile(g, eps)
    alm2 = (lam/(4*np.pi*r))**2; h = alm2*N
    Pst = sig2*(2**0.1-1)/h
    xF = (P-Pst)*alk2*ge/(sig2 + Pst*alk2*ge); xS = (P-Pst)*h/(sig2 + Pst*h)
    return np.log2(1+min(xF, xS)), np.log2(1+xF) <= np.log2(1+xS)

def fig_sharp():
    N = 128; eps = 0.05; q = norm.isf(eps/2)
    Fs = np.logspace(np.log10(0.6), np.log10(25), 70)
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    sig2 = 1e-11; P = 0.1; alk2 = (lam/(4*np.pi*120))**2; kap = P*alk2/sig2
    for i, s in enumerate([0.005, 0.01, 0.02, 0.04]):
        rr = [robust_rate(N, N**2*lam/(4*Fv), s)[0] for Fv in Fs]
        ax.plot(Fs, rr, color=COL[i], label=fr'$s_u={s}$')
        Fst = Fstar(N, q, s)
        ana = np.where(Fs >= Fst, np.log2(1+kap*N/Fs), 0)
        ax.plot(Fs, ana, '--', color=COL[i], lw=0.8)
        Fm = Fs[::4]; th = []
        for Fv in Fm:
            nu = np.linspace(0, q*s*N/(2*Fv), 300); th.append(np.log2(1+kap*G_fresnel(nu, Fv, N).min()))
        ax.plot(Fm, th, 'o', color=COL[i], mfc='none', ms=3)
    ax.plot([], [], 'k-', label='Exact quantile'); ax.plot([], [], 'ko', mfc='none', ms=3, label='Theorem 1 bound')
    ax.plot([], [], 'k--', lw=0.8, label='Corollary 1 (flat-top)')
    ax.set_xscale('log'); ax.set_xlabel(r'Focusing index $F$'); ax.set_ylabel(r'Robust FF rate $R_\epsilon$ (bit/s/Hz)')
    ax.set_xlim(Fs[0], Fs[-1]); ax.set_ylim(0, None)
    ax.set_ylim(0, 6.2); ax.legend(fontsize=6.0, loc='upper left', ncol=2)
    fig.savefig('figs/fig_sharp.pdf'); plt.close(fig)

# ---------- CRB vs ML periodogram ----------
def fig_crb():
    rng = np.random.default_rng(0)
    rk = 100.0; alk2 = (lam/(4*np.pi*rk))**2; rho = 1e-3*alk2/1e-11   # pu = 0 dBm
    Ls = np.unique(np.round(np.logspace(0, np.log10(400), 16)).astype(int))
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    for i, N in enumerate([64, 128]):
        n = np.arange(N); u0 = 0.2
        rm = []; cr = []
        for L in Ls:
            snr = L*rho
            errs = []
            for t in range(400):
                y = np.exp(1j*np.pi*n*u0) + (rng.standard_normal(N)+1j*rng.standard_normal(N))/np.sqrt(2*snr)
                Nf = 16*N; spec = np.abs(np.fft.fft(y, Nf))**2   # peak at m/Nf = u0/2
                m = np.argmax(spec); ug = 2*m/Nf; ug = (ug+1) % 2 - 1
                # local refinement
                grid = ug + np.linspace(-1/Nf*2, 1/Nf*2, 41)
                val = np.abs(np.exp(-1j*np.pi*np.outer(grid, n)) @ y)**2
                uh = grid[np.argmax(val)]
                e = (uh - u0 + 1) % 2 - 1; errs.append(e)
            rm.append(np.sqrt(np.mean(np.square(errs))))
            cr.append(np.sqrt(6/(np.pi**2*snr*N*(N**2-1))))
        ax.loglog(Ls, rm, 'o-', color=COL[i], mfc='none', label=fr'ML, $N={N}$')
        ax.loglog(Ls, cr, '--', color=COL[i], label=fr'$\sqrt{{\mathrm{{CRB}}}}$, $N={N}$')
        Lth = 10**1.4/(N*rho); ax.axvline(Lth, color=COL[i], ls=':', lw=0.8)
    ax.set_xlabel(r'FF pilot length $L_{\rm F}$ (symbols)'); ax.set_ylabel(r'RMSE of $\hat u$')
    ax.legend(fontsize=6.3); fig.savefig('figs/fig_crb.pdf'); plt.close(fig)
    print('rho dB', 10*np.log10(rho))

# ---------- phase diagram: robust gain over (F, s) ----------
def fig_phase():
    N = 128; eps = 0.05; q = norm.isf(eps/2)
    sig2 = 1e-11; P = 0.1; alk2 = (lam/(4*np.pi*120))**2; kap = P*alk2/sig2
    Fs = np.logspace(np.log10(0.7), np.log10(25), 45); ss = np.logspace(-3, np.log10(0.08), 45)
    Z = np.zeros((len(ss), len(Fs)))
    for j, Fv in enumerate(Fs):
        r = N**2*lam/(4*Fv)
        for i, sv in enumerate(ss):
            Z[i, j] = robust_rate(N, r, sv, nsamp=1500)[0]
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    pc = ax.pcolormesh(Fs, ss, Z, cmap='viridis', shading='auto')
    cb = fig.colorbar(pc, ax=ax, pad=0.02); cb.set_label(r'Robust FF rate $R_\epsilon$ (bit/s/Hz)')
    ax.plot([Fstar(N, q, sv) for sv in ss], ss, 'w--', lw=1.2, label=r'$F^\star(s_u)$, Corollary 1')
    ax.plot(N*q*ss, ss, ':', color='orange', lw=1.1, label=r'$Nq_\epsilon s_u$ (flat-top limit)')
    jb = np.argmax(Z, axis=1); ax.plot(Fs[jb], ss, 'wo', ms=2.2, mfc='none', mew=0.6, label=r'$\arg\max_F R_\epsilon$ (exact)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Focusing index $F$')
    ax.set_ylabel(r'Posterior std. $s_u$ of $u$'); ax.set_xlim(Fs[0], Fs[-1]); ax.set_ylim(ss[0], ss[-1])
    ax.legend(fontsize=5.8, loc='lower right')
    fig.savefig('figs/fig_phase.pdf'); plt.close(fig)

# ---------- net gain vs pilot length ----------
def fig_pilot():
    N = 128; T = 500; M0R = 3.6; eps = 0.05; q = norm.isf(eps/2); s0 = 0.03
    Fv = 4.0; r = N**2*lam/(4*Fv); W = 2*Fv/N
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    Ls = np.arange(0, 301, 2)
    for i, pu in enumerate([-10, -5, 0, 5]):
        rho = 10**((pu-30)/10)*(lam/(4*np.pi*120))**2/1e-11
        J = np.pi**2*rho*N*(N**2-1)/6; Lth = 10**1.4/(N*rho)
        val = []
        for L in Ls:
            sv = 1/np.sqrt(1/s0**2 + (L*J if L >= Lth else 0))
            R = robust_rate(N, r, sv, nsamp=1500)[0]
            val.append((1-L/T)*(R+M0R)-M0R)
        ax.plot(Ls, val, color=COL[i], label=fr'$p_{{\rm u}}={pu}$ dBm ($L_{{\rm th}}={Lth:.0f}$)')
        Lmin = max(Lth, (4*q**2/W**2 - 1/s0**2)/J)
        ax.axvline(Lmin, color=COL[i], ls=':', lw=0.8)
        ax.plot(Ls[int(np.argmax(val))], max(val), '*', color=COL[i], ms=6)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel(r'FF pilot length $L_{\rm F}$ (symbols)'); ax.set_ylabel(r'Net NOMA gain $\Delta R$ (bit/s/Hz)')
    ax.legend(fontsize=6.0, loc='upper right'); ax.set_xlim(0, 300)
    fig.savefig('figs/fig_pilot.pdf'); plt.close(fig)

if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]: globals()[f]()
