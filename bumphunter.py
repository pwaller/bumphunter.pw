#! /usr/bin/env python

from math import log

from numpy import array, random
from scipy.stats import poisson

def evaluate_statistic(data, mc, verbose=False, edges=None):
    # Get search range (first bin with data, last bin with data)
    nzi, = mc.nonzero() # nzi = non-zero indices
    search_lo, search_hi = nzi[0], nzi[-1]
    
    def all_windows():
        "Iterator returning [lo, hi) for all windows"
        # Try windows from one bin in width up to half of the full range
        min_win_size, max_win_size = 1, (search_hi - search_lo) // 2
        for binwidth in xrange(min_win_size, max_win_size):
            if verbose: print " --- binwidth = ", binwidth
            step = max(1, binwidth // 2) # Step size <- half binwidth
            for pos in xrange(search_lo, search_hi - binwidth, step):
                yield pos, pos + binwidth

    def pvalue(lo, hi):
        "Compute p value in window [lo, hi)"
        d, m = data[lo:hi].sum(), mc[lo:hi].sum()
        if m == 0:
            # MC prediction is zero. Not sure what then..
            assert d == 0, "Data = {0} where the prediction is zero..".format(d)
            return 1
        if d < m: return 1 # "Dips" get ignored.
            
        # P(d >= m)
        p = 1 - poisson.cdf(d-1, m)
        
        if verbose and edges:
            print "{0:2} {1:2} [{2:8.3f}, {3:8.3f}] {4:7.0f} {5:7.3f} {6:.5f} {7:.2f}".format(
                lo, hi, edges[lo], edges[hi], d, m, p, -log(p))
                
        return p
    
    min_pvalue, (lo, hi) = min((pvalue(lo, hi), (lo, hi))
                               for lo, hi in all_windows())
    
    return -log(min_pvalue), (lo, hi)

def make_toys(prediction, n):
    "fluctuate `prediction` input distribution `n` times"
    return random.mtrand.poisson(prediction, size=(n, len(prediction)))
    
def bumphunter(hdata, hmc, n):
    "Compute the bumphunter statistic and run `n` pseudo-experiments"
    data = array(hdata[i] for i in xrange(1, hdata.GetNbinsX()))
    mc   = array(hmc[i]   for i in xrange(1, hmc.GetNbinsX()))
    
    pseudo_experiments = [evaluate_statistic(pe, mc)[0]
                          for pe in make_toys(mc, n)]
    
    measurement, (lo, hi) = evaluate_statistic(data, mc)
    
    return measurement, (lo, hi), pseudo_experiments

