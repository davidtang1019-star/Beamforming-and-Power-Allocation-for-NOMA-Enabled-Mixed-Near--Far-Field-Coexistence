import numpy as np, pickle, sys
from style import *
from agg import load
LAB = {'genie': 'Perfect CSI, no overhead [6]', 'prop': 'Proposed (CC, optimized pilots)',
       'wc': 'Worst-case robust, optimized pilots', 'ccfix': 'CC, fixed pilots ($L_{\\rm N}{=}M$, $L_{\\rm F}{=}128$)',
       'nom': 'Estimate-then-design, nominal [6]', 'stat': 'CC, statistical CSI only'}
STY = {'genie': dict(color='k', ls='--', marker='', lw=1.0), 'prop': dict(color=COL[1], ls='-', marker='o'),
       'wc': dict(color=COL[0], ls='-', marker='s'), 'ccfix': dict(color=COL[2], ls='-', marker='^'),
       'nom': dict(color=COL[3], ls='-.', marker='v'), 'stat': dict(color=COL[5], ls=':', marker='x')}

def series(out, sch, key, f=np.mean):
    xs = sorted(out.keys()); y = []; c = []
    for x in xs:
        d = out[x][0]
        if sch not in d: y.append(np.nan); c.append(0); continue
        v = np.array(d[sch][key]); y.append(np.nanmean(v)); c.append(1.96*np.nanstd(v)/np.sqrt(np.sum(~np.isnan(v))))
    return np.array(xs, float), np.array(y), np.array(c)

def fig_pmax():
    out = load('pmax')
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    for sch in ['genie', 'prop', 'wc', 'ccfix', 'nom', 'stat']:
        x, y, c = series(out, sch, 'net')
        ax.errorbar(x, y, yerr=c, label=LAB[sch], capsize=2, ms=4, mfc='none', **STY[sch])
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel(r'Per-beam power budget $\bar P$ (dBm)'); ax.set_ylabel(r'Net gain $\Delta R$ (bit/s/Hz)')
    ax.legend(fontsize=5.9, loc='upper left'); ax.set_xticks([0, 10, 20, 30])
    fig.savefig('figs/fig_pmax.pdf'); plt.close(fig)

def fig_pu():
    out = load('pu')
    fig, axs = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.08))
    for sch in ['prop', 'wc', 'ccfix', 'nom', 'stat']:
        x, y, c = series(out, sch, 'net')
        axs[0].errorbar(x, y, yerr=c, label=LAB[sch], capsize=2, mfc='none', **STY[sch])
    axs[0].axhline(0, color='gray', lw=0.6); axs[0].set_ylabel(r'Net gain $\Delta R$ (bit/s/Hz)')
    axs[0].legend(fontsize=5.9, loc='upper left')
    for sch in ['prop', 'wc']:
        x, y, c = series(out, sch, 'L')
        axs[1].plot(x, y, mfc='none', label={'prop': 'Proposed', 'wc': 'Worst-case'}[sch], **STY[sch])
    axs[1].set_ylabel(r'Mean $L_{\rm N}+L_{\rm F}$'); axs[1].set_xlabel(r'FF uplink pilot power $p_{\rm u}$ (dBm)')
    axs[1].legend(fontsize=5.9)
    fig.savefig('figs/fig_pu.pdf'); plt.close(fig)

def fig_N():
    out = load('N')
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    for sch in ['genie', 'prop', 'nom', 'stat']:
        x, y, c = series(out, sch, 'net')
        ax.errorbar(x, y, yerr=c, label=LAB[sch], capsize=2, mfc='none', **STY[sch])
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel('Number of BS antennas $N$'); ax.set_ylabel(r'Net gain $\Delta R$ (bit/s/Hz)')
    ax.legend(fontsize=5.9, loc='upper left'); ax.set_ylim(-0.5, 15.5); ax.set_xticks([48, 64, 96, 128])
    fig.savefig('figs/fig_N.pdf'); plt.close(fig)

def fig_T():
    out = load('T')
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.0), gridspec_kw=dict(wspace=0.42))
    for sch in ['prop', 'ccfix']:
        x, y, c = series(out, sch, 'net')
        if sch == 'ccfix': y = np.where(x > 164, y, np.nan)
        axs[0].errorbar(x, y, yerr=c, label=LAB[sch].split('(')[0].strip(), capsize=2, mfc='none', **STY[sch])
        x, y, c = series(out, sch, 'L')
        axs[1].plot(x, y, mfc='none', **STY[sch])
    for a in axs: a.set_xscale('log'); a.set_xlabel('Block length $T$')
    axs[0].set_ylabel(r'Net gain $\Delta R$'); axs[1].set_ylabel(r'Mean $L_{\rm N}+L_{\rm F}$')
    axs[0].set_ylim(-2, 8.5)
    axs[0].legend(fontsize=5.6, loc='upper left')
    fig.savefig('figs/fig_T.pdf'); plt.close(fig)

def fig_nf():
    res = load('nf')
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.0), gridspec_kw=dict(wspace=0.45))
    for j, b in enumerate([0.0, 1e-3]):
        xs = [k[0] for k in res if k[1] == b]
        r = [np.mean(res[(sr, b)][0]['prop']['Rff']) for sr in xs]
        f = [np.nanmean(res[(sr, b)][0]['prop']['fsic']) for sr in xs]
        lab = r'$\beta=0$' if b == 0 else r'$\beta=-30$ dB'
        axs[0].plot(xs, r, marker='o', mfc='none', color=COL[j], label=lab)
        axs[1].plot(xs, f, marker='s', mfc='none', color=COL[j], label=lab)
    for a in axs: a.set_xscale('log'); a.set_xlabel(r'Legacy distance error $\sigma_{r,0}$ (m)')
    axs[0].set_ylabel(r'FF sum rate (bit/s/Hz)'); axs[1].set_ylabel('Fraction SIC-limited')
    axs[0].legend(fontsize=6)
    fig.savefig('figs/fig_nf.pdf'); plt.close(fig)

def fig_K():
    out = load('K')
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for sch in ['genie', 'prop']:
        x, y, c = series(out, sch, 'net')
        ax.errorbar(x, y, yerr=c, label=LAB[sch], capsize=2, mfc='none', **STY[sch])
    ax.set_xlabel('Number of FF users $K$'); ax.set_ylabel(r'Net gain $\Delta R$ (bit/s/Hz)'); ax.legend(fontsize=6)
    fig.savefig('figs/fig_K.pdf'); plt.close(fig)

def fig_conv():
    res = pickle.load(open('data/pmax.pkl', 'rb'))
    bb = pickle.load(open('data/bb.pkl', 'rb'))
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.0), gridspec_kw=dict(wspace=0.45))
    for i, (s, o) in enumerate(res[20][:8]):
        h = o['prop']['hist']; axs[0].plot(range(len(h)), h, marker='.', ms=3, color=COL[i % len(COL)], lw=0.8)
    axs[0].set_xlabel('SCA iteration $t$'); axs[0].set_ylabel(r'$f(\mathbf{v}^{(t)})$ (bit/s/Hz)')
    sca = np.array([b['sca'] for b in bb]); lb = np.array([b['LB'] for b in bb]); ub = np.array([b['UB'] for b in bb])
    axs[1].plot(ub, sca, 'o', mfc='none', color=COL[1], ms=3.5)
    m = max(ub.max(), sca.max())*1.05; axs[1].plot([0, m], [0, m], 'k--', lw=0.7)
    axs[1].set_xlabel('BB optimum (bit/s/Hz)'); axs[1].set_ylabel('Algorithm 1 (bit/s/Hz)')
    fig.savefig('figs/fig_conv.pdf'); plt.close(fig)

if __name__ == '__main__':
    for f in sys.argv[1:]: globals()[f]()
