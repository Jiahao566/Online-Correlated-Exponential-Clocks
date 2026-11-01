#!/usr/bin/env python3
"""Small-request certificate for every x in [0, 1/10].

The shared potential table is supplied by verify.py. All u integrals use
finite interval rectangles. Run this file for the small-request part only.
"""
import sys
import time

from flint import arb


def interval(lo, hi):
    return lo.union(hi)


def finite_lower(value):
    assert value.is_finite(), value
    return value.lower()


class RequestBox:
    """Constants shared by all z cells of one complete request interval."""

    def __init__(self, x, u_cells, table, lam, mu):
        self.x, self.table, self.lam = x, table, lam
        self.r = (-lam*x).exp()
        a = self.r-mu*x
        a1, a2 = -lam*self.r-mu, lam*lam*self.r
        self.base_constant = -a2
        self.base_g = (1-x)*a2-2*a1
        self.base_gd = -2*lam*(1-x)*a1+2*lam*a+lam*lam*(1-x)*a
        self.base_gdd = lam*lam*(1-x)*a
        self.terms = []
        c = arb(20)/33
        for k in range(u_cells):
            u = interval(arb(k)/u_cells, arb(k+1)/u_cells)
            t = lam*u
            e = (-t*x).exp()
            w = (-lam*x*(1-u)).exp()
            h, hd, hdd = table.reset(w)
            hx = -lam*(1-u)*w*hd
            hxx = lam*lam*(1-u)**2*(w*hd+w*w*hdd)
            f = 1-(1-x)*h
            f1, f2 = h-(1-x)*hx, 2*hx-(1-x)*hxx
            c0 = (-2+2*x*t)*f1-x*f2+(2*t-x*t*t)*f
            c1 = (2-2*x*t)*f+2*x*f1
            c2 = x*f
            # p, p_x and p_xx are cubics in v=lambda*z*exp(-lambda*x*u).
            p = (arb(1), x/2, c*x*x)
            p1 = (-t, arb(1)/2-t*x, c*(2*x-3*t*x*x))
            p2 = (t*t, -2*t+2*t*t*x, c*(2-12*t*x+9*t*t*x*x))
            ell = tuple(-(1-x)*a for a in p)
            ell1 = tuple(a-(1-x)*b for a,b in zip(p,p1))
            ell2 = tuple(2*a-(1-x)*b for a,b in zip(p1,p2))
            self.terms.append((lam*e, ell, ell1, ell2,
                               c0*lam*e/u_cells,
                               c1*lam*e/u_cells,
                               c2*lam*e/u_cells))

    def base(self, z):
        y = z*self.r
        g, gd, gdd = self.table.gamma(y)
        return (self.base_constant+self.base_g*g
                +self.base_gd*y*gd+self.base_gdd*y*y*gdd)

    def cell(self, z):
        value = self.base(z)
        for ve, ell, ell1, ell2, c0, c1, c2 in self.terms:
            v = ve*z
            l = v*(ell[0]+v*(ell[1]+v*ell[2]))
            l1 = v*(ell1[0]+v*(ell1[1]+v*ell1[2]))
            l2 = v*(ell2[0]+v*(ell2[1]+v*ell2[2]))
            d = l.exp()
            value += (1-d)*c0+d*(l1*c1+(l1*l1+l2)*c2)
        return finite_lower(value)


def certify(x, age_cells, u_cells, table, lam, mu):
    box = RequestBox(x, u_cells, table, lam, mu)
    running, total = None, arb(0)
    for k in range(age_cells):
        z = interval(arb(k)/age_cells, arb(k+1)/age_cells)
        bound = box.cell(z)
        running = bound if running is None else min(running, bound)
        total += running/age_cells
    return finite_lower(total)



def verify(table, lam, mu, request_cells=128, age_cells=1024, u_cells=256):
    """Recompute all small-request bounds; never read a saved certificate."""
    assert min(request_cells, age_cells, u_cells) > 0
    started = time.perf_counter()
    reports, bounds = [], []
    for index in range(request_cells-1, -1, -1):
        tick = time.perf_counter()
        x = interval(arb(index)/(10*request_cells), arb(index+1)/(10*request_cells))
        bound = certify(x, age_cells, u_cells, table, lam, mu)
        assert bound >= 0, ('small-request certificate failure', index, bound)
        bounds.append(bound)
        reports.append({
            'index': index, 'left': f'{index}/{10*request_cells}',
            'right': f'{index+1}/{10*request_cells}',
            'lower_bound': str(bound), 'passes': True,
            'seconds': time.perf_counter()-tick,
        })
        if len(reports) == 1 or len(reports) % 16 == 0 or len(reports) == request_cells:
            print(f'Small requests: {len(reports)}/{request_cells} passed; '
                  f'minimum {min(bounds)}', flush=True)
    assert {r['index'] for r in reports} == set(range(request_cells))
    return {
        'complete': True,
        'claim': 'nu R_x >= 0 for every reachable nu and 0 <= x <= 1/10',
        'method': 'finite potential endpoint sums and full-interval u rectangles',
        'certified_quantity': 'lower bound on B_age_cells(partial_x^2 Rlower_x)',
        'request_cells': request_cells, 'age_cells': age_cells, 'u_cells': u_cells,
        'age_rectangles': request_cells*age_cells,
        'integrand_boxes': request_cells*age_cells*u_cells,
        'minimum_lower_bound': str(min(bounds)),
        'certificate_seconds': time.perf_counter()-started,
        'intervals': reports,
    }


if __name__ == '__main__':
    from verify import main
    main(['--only', 'small', *sys.argv[1:]])
