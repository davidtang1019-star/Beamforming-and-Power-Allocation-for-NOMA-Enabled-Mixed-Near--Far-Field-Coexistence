import pickle, numpy as np, sys
def load(name):
    res = pickle.load(open(f'data/{name}.pkl', 'rb'))
    out = {}
    for val, lst in res.items():
        d = {}
        errs = sum(1 for s, o in lst if 'err' in o)
        for s, o in lst:
            if 'err' in o: continue
            for sch, v in o.items():
                for key in ['net', 'Rff', 'outFF', 'outNF', 'L', 'fsic', 'it', 't', 'nadm', 'non']:
                    d.setdefault(sch, {}).setdefault(key, []).append(float(v[key]))
        out[val] = (d, errs)
    return out
if __name__ == '__main__':
    out = load(sys.argv[1])
    for val, (d, errs) in out.items():
        print('==', val, 'errors', errs)
        for sch, v in d.items():
            n = len(v['net']); m = np.mean(v['net']); ci = 1.96*np.std(v['net'])/np.sqrt(n)
            print(f"  {sch:6s} n={n} net={m:.3f}±{ci:.3f} Rff={np.mean(v['Rff']):.3f} outFF={np.mean(v['outFF']):.3f} outNF={np.mean(v['outNF']):.4f} L={np.mean(v['L']):.0f} fsic={np.nanmean(v['fsic']):.2f} it={np.mean(v['it']):.1f} t={np.mean(v['t']):.1f}")
