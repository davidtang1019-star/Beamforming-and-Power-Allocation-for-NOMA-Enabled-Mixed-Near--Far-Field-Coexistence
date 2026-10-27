import pickle, numpy as np, sys
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
name = sys.argv[1]; vals = eval(sys.argv[2]); base = sys.argv[3].split(',')
res = pickle.load(open(f'data/{name}.pkl', 'rb'))
rows = []
for v in vals:
    lst = [o for s, o in res[v] if 'err' not in o]
    for b in base:
        p = np.array([o['prop']['net'] for o in lst]); q = np.array([o[b]['net'] for o in lst])
        d = p - q; st = wilcoxon(p, q)
        rows.append((v, b, d.mean(), 1.96*d.std(ddof=1)/np.sqrt(len(d)), np.mean(d > 0), st.pvalue))
padj = multipletests([r[-1] for r in rows], method='holm')[1]
for r, pa in zip(rows, padj):
    print(f"{r[0]} prop-{r[1]}: mean diff {r[2]:.3f} ±{r[3]:.3f}, prop better in {100*r[4]:.0f}% drops, p={r[5]:.2e}, Holm p={pa:.2e}")
