#!/usr/bin/env python3
"""Large-request certificate for every x in [1/10, 1].

The shared potential table is supplied by verify.py. All kernel integrals
use finite endpoint sums. Run this file for the large-request part only.
"""
import sys
import time

from flint import arb


def lower(value):
    assert value.is_finite(), value
    return value.lower()


def score(x, rho, y, lam):
    """Positive-first-age win probability; at y=1 use its continuous limit."""
    if x == 1:
        return arb(1)
    if y <= rho:
        base = 1 - lam*x*y
    elif y >= rho:
        base = 1 + rho - y + y*y.log()
    else:
        # A rounded argument can straddle the analytic branch boundary.
        base = (1-lam*x*y).union(1+rho-y+y*y.log())
    assert base > 0
    return (base.log() * ((1-x)/x)).exp()


def certify_interval(index, age_cells, kernel_cells, table, lam, mu):
    left, right = arb(16)/(index+1), arb(16)/index
    rl, rr = (-lam*left).exp(), (-lam*right).exp()
    dl, dr = (1-rl)/kernel_cells, (1-rr)/kernel_cells
    vl = [rl + (1-rl)*arb(k)/kernel_cells for k in range(kernel_cells+1)]
    vr = [rr + (1-rr)*arb(k)/kernel_cells for k in range(kernel_cells+1)]
    h = [table.reset_bounds(rr/v)[0] for v in vr[:-1]] if not (right == 1) else []
    assert all(v >= 0 for v in h)
    running, total = None, arb(0)
    for j in range(age_cells):
        lo, hi = arb(j)/age_cells, arb(j+1)/age_cells
        cell = mu*left - table.gamma_bounds(lo)[1]
        gain, remaining = arb(0), arb(0)
        for k in range(kernel_cells):
            gain += lower(score(left, rl, hi*vl[k+1], lam))
            if not (right == 1):
                loss = max(arb(0), lower(1-score(right, rr, lo*vr[k], lam)))
                remaining += loss*h[k]
        cell += dl*gain
        if not (right == 1):
            cell += (1-right)*((rr-mu*right)*table.gamma_bounds(hi*rr)[0] + dr*remaining)
        cell = lower(cell)
        running = cell if running is None else min(running, cell)
        total += running/age_cells
    return lower(total)



def verify(table, lam, mu, age_cells=512, kernel_cells=256):
    """Recompute all 144 large-request bounds from the supplied table."""
    assert min(age_cells, kernel_cells) > 0
    started = time.perf_counter()
    reports, bounds = [], []
    for index in range(159, 15, -1):
        tick = time.perf_counter()
        bound = certify_interval(index, age_cells, kernel_cells, table, lam, mu)
        assert bound >= 0, ('large-request certificate failure', index, bound)
        bounds.append(bound)
        reports.append({
            'index': index, 'left': f'16/{index+1}', 'right': f'16/{index}',
            'lower_bound': str(bound), 'passes': True,
            'seconds': time.perf_counter()-tick,
        })
        if len(reports) == 1 or len(reports) % 12 == 0:
            print(f'Large requests: {len(reports)}/144 passed; '
                  f'minimum {min(bounds)}', flush=True)
    assert {r['index'] for r in reports} == set(range(16, 160))
    return {
        'complete': True,
        'claim': 'nu R_x >= 0 for every reachable nu and 1/10 <= x <= 1',
        'method': 'finite monotone endpoint sums; no potential series truncation',
        'certified_quantity': 'lower bound on B_age_cells(C_left,right)',
        'request_intervals': len(reports), 'age_cells': age_cells,
        'kernel_integration_cells': kernel_cells,
        'age_rectangles': len(reports)*age_cells,
        'minimum_lower_bound': str(min(bounds)),
        'certificate_seconds': time.perf_counter()-started,
        'intervals': reports,
    }


if __name__ == '__main__':
    from verify import main
    main(['--only', 'large', *sys.argv[1:]])
