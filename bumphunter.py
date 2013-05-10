#! /usr/bin/env python

from math import log

from numpy import array, random
from scipy.special import gammainc
from scipy.stats import poisson

def evaluate_statistic(data, mc, verbose=False, edges=None):
    # Get search range (first bin with data, last bin with data)
    # nz = non-zero indices
    nzi, = mc.nonzero()
    search_lo, search_hi = nzi[0], nzi[-1]    
    
    def all_windows():
        "Iterator returning [lo, hi) for all windows"        
        min_win_size, max_win_size = 1, (search_hi - search_lo) // 2
        for binwidth in xrange(min_win_size, max_win_size):
            if verbose: print " --- binwidth = ", binwidth
            step = max(1, binwidth // 2)
            for pos in xrange(search_lo, search_hi - binwidth, step):
                yield pos, pos+binwidth

    def pvalue(lo, hi):
        "Compute p value in window [lo, hi)"
        d, m = data[lo:hi].sum(), mc[lo:hi].sum()
        # MC prediction is zero.
        if m == 0: return 1
        # "Dips" get ignored.
        if d < m: return 1
            
        #v = poisson.cdf(d, m)
        v = gammainc(d, m)
        if verbose and edges:
            print "{0:2} {1:2} [{2:8.3f}, {3:8.3f}] {4:7.0f} {5:7.3f} {6:.5f} {7:.2f}".format(
                lo, hi, edges[lo], edges[hi], d, m, v, -log(v))
                
        return v

    min_pvalue, (lo, hi) = min((pvalue(lo, hi), (lo, hi))
                               for lo, hi in all_windows())
        
    return -log(min_pvalue), (lo, hi)

def make_toys(prediction, n):
    "fluctuate `prediction` n times"
    return random.mtrand.poisson(prediction, size=(n, len(prediction)))
    
def bumphunter(hdata, hmc, n):
    "Compute the bumphunter statistic and run `n` pseudo-experiments"
    data = array(hdata[i] for i in xrange(1, hdata.GetNbinsX()))
    mc   = array(hmc[i]   for i in xrange(1, hmc.GetNbinsX()))
    
    pseudo_experiments = [evaluate_statistic(data, pe)[0]
                          for pe in make_toys(mc, n)]
    
    measurement, (lo, hi) = evaluate_statistic(data, mc)
    
    return measurement, (lo, hi), pseudo_experiments

