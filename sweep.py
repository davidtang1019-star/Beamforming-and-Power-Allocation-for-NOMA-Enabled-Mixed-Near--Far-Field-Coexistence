import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pickle, sys, time
from multiprocessing import Pool
from run import *

ND = int(os.environ.get('ND', 60))
EXPS = {
 'pmax': [('Pmax_dBm', v, dict(Pmax_dBm=v), ['genie', 'prop', 'wc', 'nom', 'ccfix', 'stat']) for v in [0, 10, 20, 30]],
 'pu':   [('pu_dBm', v, dict(pu_dBm=v), ['prop', 'wc', 'nom', 'ccfix', 'stat']) for v in [-10, -5, 0, 5, 10]],
 'N':    [('N', v, dict(N=v), ['genie', 'prop', 'nom', 'stat']) for v in [48, 64, 96, 128]],
 'T':    [('T', v, dict(T=v), ['prop', 'ccfix']) for v in [100, 200, 500, 1000, 2000]],
 'nf':   [('sr_beta', (sr, b), dict(sig_r0=sr, beta=b), ['prop']) for b in [0.0, 1e-3] for sr in [0.05, 0.1, 0.2, 0.4]],
 'K':    [('K', v, dict(K=v), ['genie', 'prop']) for v in [1, 2, 4, 6, 8]],
}

def job(args):
    kw, seed, schemes = args
    import signal
    def handler(sig, frm): raise TimeoutError('drop timeout')
    signal.signal(signal.SIGALRM, handler); signal.alarm(400)
    S = Sys(**kw)
    try:
        r = seed, run_drop(S, seed, schemes)
    except Exception as e:
        r = seed, {'err': repr(e)}
    signal.alarm(0)
    return r

if __name__ == '__main__':
    name = sys.argv[1]
    res = {}
    t0 = time.time()
    with Pool(2) as pool:
        for (pname, val, kw, schemes) in EXPS[name]:
            out = pool.map(job, [(kw, s, schemes) for s in range(ND)])
            res[val] = out
            print(name, val, 'done', round(time.time()-t0), flush=True)
            pickle.dump(res, open(f'data/{name}.pkl', 'wb'))
