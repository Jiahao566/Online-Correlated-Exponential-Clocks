#!/usr/bin/env python3
"""Certify the paper's AdWords 53/100 potential using interval masses.

Run with --output NEW_DIRECTORY to recompute all numerical enclosures and
then audit the exact witnesses and complete covers. Add --audit-existing
for a read-only audit of saved results, without repeating integrations.

All accounting, kernel, LP, curvature, and audit helpers are in this file.
The only other entry point is verify_score.py; the candidate is read from
candidate.json beside this script. Mesh/multiplier hints and the initial request-mass grid
are embedded as exact rational data and never serve as proof by themselves.
See ../README.md for the mathematics, dependencies, and reproduction commands.
"""
# Self-contained accounting core for adwords/verify.py.
# All functions use exact Fractions or outward Arb enclosures for acceptance.
# Floating-point quadrature and LP optimization only propose checked witnesses.
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction as Q
from functools import lru_cache
from math import comb, factorial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import gzip
import hashlib
import json
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from flint import arb, arb_series, ctx

HERE = Path(__file__).resolve().parent
BASE = HERE
ctx.prec = 384
LAM = Q(9, 2)
KNOTS = list(map(Q, ['0', '1/20', '1/10', '3/20', '1/5', '1/4',
                    '3/10', '2/5', '1/2', '5/8', '3/4']))
# Nonconstant coefficients of the degree-26 Ein expansion. Every accounting
# use replaces its constant by an outward upper bound for the ANALYTIC score.
H = [-Q(263341615608130387, 250000000000000000)] + [
    (-1)**(k+1)*LAM**k/(k*factorial(k)) for k in range(1, 27)]
FAMILIES = {f'{j}:{b}' for j in range(11) for b in ('1/2', '3/4', '1')} | {'continuation'}


def ball(value):
    """Convert an exact rational or an existing enclosure to an Arb ball."""
    if isinstance(value, Q):
        return arb(value.numerator) / value.denominator
    return arb(value)


# Rational candidate parsing, Bernstein conversions, and shape/load conditions.

def to_power(b):
    n = len(b) - 1
    return [comb(n, k) * sum(((-1) ** (k - j) * comb(k, j) * b[j] for j in range(k + 1))) for k in range(n + 1)]

def local_bern(p, l, u, M):
    d = len(p) - 1
    w = u - l
    a = [w ** k * sum((p[j] * comb(j, k) * l ** (j - k) for j in range(k, d + 1))) for k in range(d + 1)]
    return [sum((a[k] * Q(comb(i, k), comb(M, k)) for k in range(min(i, d) + 1))) for i in range(M + 1)]

def local_edges(p, edges, M):
    return sum([local_bern(p, l, u, M) for l, u in zip(edges, edges[1:])], [])

def add(p, q):
    return [(p[k] if k < len(p) else 0) + (q[k] if k < len(q) else 0) for k in range(max(len(p), len(q)))]

def scale(p, c):
    return [c * x for x in p]

def load_prices(path):
    """Load the exact rational F,G,h data; internal formulas use f=G,g=F-G."""
    data = json.loads(Path(path).read_text())
    if data['lambda'] != '9/2' or data['ratio'] != '53/100' or data['degree'] != 12:
        raise ValueError('Unexpected certificate parameters')
    if list(map(Q, data['knots'])) != KNOTS:
        raise ValueError('Unexpected load knots')
    G = [[Q(x) for x in row] for row in data['G']]
    F = [[Q(x) for x in row] for row in data['F']]
    h = [Q(x) for x in data['h']]
    if len(F) != 11 or len(G) != 11 or any((len(row) != 13 for row in F + G)) or (len(h) != 13):
        raise ValueError('Unexpected coefficient dimensions')
    return (G, [[a - b for a, b in zip(fr, gr)] for fr, gr in zip(F, G)], h)

def small_costs(f, g, h):
    for j in range(10):
        dt = KNOTS[j + 1] - KNOTS[j]
        df = scale(add(f[j + 1], scale(f[j], -1)), 1 / dt)
        dg = scale(add(g[j + 1], scale(g[j], -1)), 1 / dt)
        for e in [j, j + 1]:
            yield (f'load_{j}_{e}', add([1 + 2 * KNOTS[e]], add(scale(add(f[e], g[e]), -3), add(df, dg))), add([Q(2)], add(scale(f[e], -3), df)))
    yield ('continuation_start', add([1 + 2 * KNOTS[-1]], add(scale(add(f[-1], g[-1]), -3), h)), add([Q(2)], scale(f[-1], -3)))
    yield ('continuation_slope', add([Q(2)], scale(h, -3)), [Q(0)])

def shape(f, g, h):
    out = {}
    for j in range(11):
        for name, b in [('one', f[j]), ('zero_minus_half', add(g[j], scale(f[j], Q(1, 2))))]:
            out[f'{name}_{j}_nonnegative'] = min(b)
            out[f'{name}_{j}_age'] = min((b[k] - b[k + 1] for k in range(12)))
    out['h_nonnegative'] = min(h)
    out['h_age'] = min((h[k] - h[k + 1] for k in range(12)))
    for j in range(10):
        for name, vec in [('one', f), ('zero', [add(a, b) for a, b in zip(f, g)])]:
            deriv = scale(add(vec[j + 1], scale(vec[j], -1)), 1 / (KNOTS[j + 1] - KNOTS[j]))
            for e in [j, j + 1]:
                out[f'{name}_{j}_{e}_load'] = min(add(scale(vec[e], 2), scale(deriv, -1)))
    out['continuation_load'] = min(add(scale(add(f[-1], g[-1]), 2), scale(h, -1)))
    return out


# Numerical kernel fits: proposals only, certified below on entire intervals.
_PROPOSAL_L = 4.5
_PROPOSAL_MU = np.exp(-_PROPOSAL_L)
_DEGREE = 12
_BASIS_SIZE = 13

def bern(z, n=_DEGREE):
    z = np.asarray(z)
    return np.array([comb(n, j) * z ** j * (1 - z) ** (n - j) for j in range(n + 1)]).T

def bernprime(z, n=_DEGREE):
    p = bern(z, n - 1)
    q = np.zeros((len(z), n + 1))
    q[:, 1:] += n * p
    q[:, :-1] -= n * p
    return q

def numerical_kernel(z, x, nquad=32):
    P = bern(z)
    if x == 0:
        LM = -_PROPOSAL_L * z[:, None] * bernprime(z) - (_PROPOSAL_L + _PROPOSAL_MU) * P
        LM[:, -1] += _PROPOSAL_L * (1 - np.exp(-_PROPOSAL_L * z))
        LG = -_PROPOSAL_L * z[:, None] * bernprime(z) - _PROPOSAL_L * P
        LG[:, -1] += _PROPOSAL_L
        return (LM, LG, _PROPOSAL_MU + _PROPOSAL_L * np.exp(-_PROPOSAL_L * z))
    r = np.exp(-_PROPOSAL_L * x)
    G = r * bern(r * z) + np.cumsum(bern(np.array([r]), _DEGREE + 1)[0])[0:_BASIS_SIZE] / (_DEGREE + 1)
    if x == 1:
        return (np.zeros_like(P), G, np.ones(len(z)))
    beta = (1 - x) / x
    nodes, weights = np.polynomial.legendre.leggauss(nquad)
    split = np.minimum(1, r / np.maximum(z, 1e-100))
    split = np.maximum(r, split)
    out = (r - _PROPOSAL_MU * x) * bern(r * z)
    for lo, hi in [(np.full_like(z, r), split), (split, np.ones_like(z))]:
        v = lo[:, None] + (hi - lo)[:, None] * (nodes + 1) / 2
        y = z[:, None] * v
        base = np.where(y <= r, 1 - _PROPOSAL_L * x * y, 1 + r - y + y * np.log(np.maximum(y, 1e-100)))
        miss = -np.expm1(beta * np.log(base))
        t = r / v
        BP = bern(t.ravel()).reshape(t.shape + (_BASIS_SIZE,))
        BP1 = bern(t.ravel(), _DEGREE + 1).reshape(t.shape + (_BASIS_SIZE + 1,))
        gap = t[:, :, None] * BP + np.cumsum(BP1, axis=2)[:, :, :_BASIS_SIZE] / (_DEGREE + 1)
        out += np.einsum('ij,ijk,j,i->ik', miss, gap, weights, (hi - lo) / 2)
    return (out, G, 1 - out.sum(axis=1))


# Rigorous finite-request kernel evaluation and full-cell approximation errors.
_KERNEL_L = arb(9) / 2

def series_value(coef, z):
    out = coef[-1]
    for c in coef[-2::-1]:
        out = out * z + c
    return out

def _kernel_hull(a, b):
    return arb((a + b) / 2, ((b - a) / 2).abs_upper())

def interval_max(a, b):
    """Enclose max of the exact numbers represented by two balls."""
    return _kernel_hull(max(a.lower(), b.lower()), max(a.upper(), b.upper()))

def interval_min(a, b):
    return _kernel_hull(min(a.lower(), b.lower()), min(a.upper(), b.upper()))

def score_segments(x, degree=20, tolerance='1e-16'):
    """(l,u,global power coefficients,uniform error) for D_x(y)."""
    ctx.cap = max(ctx.cap, degree + 2)
    x = arb(x)
    r = (-_KERNEL_L * x).exp()
    beta = (1 - x) / x
    tol = arb(tolerance)
    d = [arb(1)]
    for k in range(1, 42):
        d.append(-_KERNEL_L * r * (1 - k * x) / k * d[-1])
    rem = d[41].abs_upper() / (1 - (-arb(1)).exp())
    ans = [(arb(0), r, [d[k] / r ** k for k in range(41)], rem)]

    def recurse(l, u, depth=0):
        m = (l + u) / 2
        h = (u - l) / 2

        def D(t):
            return ((1 + r - t + t * t.log()).log() * beta).exp()
        t = arb_series([_kernel_hull(l, u), 1], prec=degree + 2)
        bound = D(t)[degree + 1].abs_upper() * h.abs_upper() ** (degree + 1)
        if not bound.is_finite() or not bound < tol:
            assert depth < 30, (x, l, u, bound)
            recurse(l, m, depth + 1)
            recurse(m, u, depth + 1)
            return
        tay = D(arb_series([m, 1], prec=degree + 1))
        power = [arb(0)] * (degree + 1)
        for j in range(degree + 1):
            for k in range(j + 1):
                power[k] += tay[j] * comb(j, k) * (-m) ** (j - k)
        ans.append((l, u, power, bound))
    recurse(r, arb(1))
    return ans

class KernelExpression:
    """P(z)+C/z+J(z)log(z), with rigorous Arb coefficients."""

    def __init__(self):
        self.p = []
        self.c = arb(0)
        self.j = []

    def copy(self):
        out = KernelExpression()
        out.p = self.p.copy()
        out.c = self.c
        out.j = self.j.copy()
        return out

    def term(self, k, c, log=False):
        if k == -1:
            assert not log
            self.c += c
            return
        a = self.j if log else self.p
        if len(a) <= k:
            a.extend([arb(0)] * (k + 1 - len(a)))
        a[k] += c

    def add_integral(self, score, gap, lower, upper):
        for k, dk in enumerate(score):
            for p, gp in gap.items():
                pref = dk * gp
                e = k - p + 1
                for sign, (kind, b) in [(1, upper), (-1, lower)]:
                    t = sign * pref
                    if e:
                        self.term(k if kind == 'constant' else p - 1, t * b ** e / e)
                    else:
                        self.term(k, t * b.log())
                        if kind == 'divided':
                            self.term(k, -t, log=True)

    def value(self, z):
        val = series_value(self.p, z) if self.p else z * 0
        if self.c != 0:
            val += self.c / z
        if self.j:
            val += series_value(self.j, z) * z.log()
        return val

    def local_taylor(self, l, u, degree=12):
        """Taylor coefficients in (z-midpoint), and uniform remainder."""
        ctx.cap = max(ctx.cap, degree + 2)
        m = (l + u) / 2
        h = (u - l) / 2
        if not self.j and self.c == 0:
            n = len(self.p) - 1
            t = [arb(0)] * (n + 1)
            for j, c in enumerate(self.p):
                for k in range(j + 1):
                    t[k] += c * comb(j, k) * m ** (j - k)
            err = sum((t[k].abs_upper() * h.abs_upper() ** k for k in range(degree + 1, n + 1)))
            return ((t + [arb(0)] * (degree + 1))[:degree + 1], arb(err))
        t = self.value(arb_series([m, 1], prec=degree + 1))
        derivative = self.value(arb_series([_kernel_hull(l, u), 1], prec=degree + 2))[degree + 1]
        err = derivative.abs_upper() * h.abs_upper() ** (degree + 1)
        return ([t[k] for k in range(degree + 1)], err)

def kernel_segments(x, coef, score=None, pnorm=None):
    """Return (l,u,expression,error) covering z in [0,1].

    coef is in global POWER basis, and must specify exact rationals or
    enclosing Arb balls. pnorm, if supplied, is a proved sup-norm upper bound.
    """
    x = arb(x)
    coef = [arb(c) for c in coef]
    r = (-_KERNEL_L * x).exp()
    mu = (-_KERNEL_L).exp()
    if score is None:
        score = score_segments(x)
    if pnorm is None:
        pnorm = sum((c.abs_upper() for c in coef))
    eps = max((e.upper() for l, u, d, e in score)) * (1 - r) * pnorm
    gap = {0: sum((c / (j + 1) for j, c in enumerate(coef)))}
    gap.update({j + 1: coef[j] * j / (j + 1) * r ** (j + 1) for j in range(1, len(coef))})
    out = KernelExpression()
    for j, c in enumerate(coef):
        out.term(j, (r - mu * x) * c * r ** j)

    def residual(d):
        return [1 - d[0]] + [-c for c in d[1:]]
    old = residual(score[0][2])
    first = out.copy()
    first.add_integral(old, gap, ('constant', r), ('constant', arb(1)))
    ans = [(arb(0), r, first, eps)]
    out.add_integral(old, gap, ('constant', r), ('divided', r))
    for l, u, d, e in score[1:]:
        dd = residual(d)
        this = out.copy()
        this.add_integral(dd, gap, ('divided', l), ('constant', arb(1)))
        ans.append((l, u, this, eps))
        out.add_integral(dd, gap, ('divided', l), ('divided', u))
    return ans

def taylor_to_bernstein(t, l, u):
    """Taylor in z-(l+u)/2 -> Bernstein on [l,u], exactly with balls."""
    n = len(t) - 1
    w = u - l
    power = [arb(0)] * (n + 1)
    for j, c in enumerate(t):
        for k in range(j + 1):
            power[k] += c * comb(j, k) * w ** j * (-arb(1) / 2) ** (j - k)
    return [sum((power[k] * arb(comb(i, k)) / comb(n, k) for k in range(i + 1))) for i in range(n + 1)]

def bernstein_to_power(b, l, u):
    """Global power coefficients of a local Bernstein polynomial."""
    b = [arb(c) for c in b]
    n = len(b) - 1
    w = u - l
    local = [sum((b[i] * comb(n, i) * comb(n - i, k - i) * (-1) ** (k - i) for i in range(k + 1))) for k in range(n + 1)]
    out = [arb(0)] * (n + 1)
    for k, c in enumerate(local):
        for j in range(k + 1):
            out[j] += c / w ** k * comb(k, j) * (-l) ** (k - j)
    return out

def approximation_error(x, coef, approximations, score=None, pnorm=None, degree=12):
    """Certify supplied local Bernstein approximations on arbitrary cells.

    approximations is [(rational_l,rational_u,Bernstein_coefficients),...].
    Return one rigorous absolute error bound per supplied cell. Every score
    branch and the entire cell interior is included, not just sample nodes.
    """
    segs = kernel_segments(x, coef, score, pnorm)
    answers = []
    for l, u, b in approximations:
        l, u = (arb(l), arb(u))
        pp = bernstein_to_power(b, l, u)
        bounds = []
        for a, c, expr, eps in segs:
            if u <= a or c <= l:
                continue
            lo = interval_max(l, a)
            hi = interval_min(u, c)
            diff = expr.copy()
            for j, v in enumerate(pp):
                diff.term(j, -v)

            def certify(v, w, depth=0):
                ts, err = diff.local_taylor(v, w, degree)
                if (not err.is_finite() or err > arb('1e-12')) and depth < 20:
                    m = (v + w) / 2
                    certify(v, m, depth + 1)
                    certify(m, w, depth + 1)
                    return
                bb = taylor_to_bernstein(ts, v, w)
                bounds.append(max((q.abs_upper() for q in bb)) + err + eps)
            certify(lo, hi)
        assert bounds, (l, u)
        answers.append(max((b.upper() for b in bounds)))
    return answers


# Outward interval minima of polynomial plus exponential costs.

def _hull_bounds(lower, upper):
    """Enclose [lower,upper], where both inputs are rigorous exact endpoints."""
    return arb((lower + upper) / 2, ((upper - lower) / 2).abs_upper())

def _coefficient_range(coefficients):
    if not coefficients:
        return arb(0)
    return _hull_bounds(min((c.lower() for c in coefficients)), max((c.upper() for c in coefficients)))

def _split(coefficients, t):
    """De Casteljau subdivision at t; both lists include their shared endpoint."""
    work = list(coefficients)
    left, right = ([work[0]], [work[-1]])
    omt = 1 - t
    for remaining in range(len(work) - 1, 0, -1):
        work = [omt * work[j] + t * work[j + 1] for j in range(remaining)]
        left.append(work[0])
        right.append(work[-1])
    right.reverse()
    return (left, right)

def _restrict(coefficients, left, right):
    """Global Bernstein coefficients -> Bernstein coefficients on [left,right]."""
    if left == 0 and right == 1:
        return list(coefficients)
    if right == 1:
        return _split(coefficients, left)[1]
    prefix = _split(coefficients, right)[0]
    if left == 0:
        return prefix
    return _split(prefix, left / right)[1]

class PolynomialExponential:
    """Reusable rigorous interval-minimum calculator for one fixed function."""

    def __init__(self, power_coefficients, exp_coefficient=0, lam=Q(9, 2)):
        self.power = list(map(ball, power_coefficients)) or [arb(0)]
        while len(self.power) > 1 and self.power[-1] == 0:
            self.power.pop()
        self.exp_coefficient = ball(exp_coefficient)
        self.lam = ball(lam)
        if not self.lam >= 0:
            raise ValueError('lambda must be certified nonnegative')
        self.degree = len(self.power) - 1
        n = self.degree
        self.bernstein = [sum((self.power[k] * comb(j, k) / comb(n, k) for k in range(j + 1)), arb(0)) for j in range(n + 1)]
        self.stats = dict(intervals=0, nodes=0, monotone=0, concave=0, chord=0, subdivisions=0, maximum_depth=0, seconds=0.0)

    def value(self, z):
        """Outward value enclosure at a point or over an Arb input ball."""
        z = ball(z)
        value = self.power[-1]
        for coefficient in self.power[-2::-1]:
            value = value * z + coefficient
        return value + self.exp_coefficient * (-self.lam * z).exp()

    def lower_bound(self, left, right, tolerance='1e-9', max_depth=40):
        """Return a rigorous lower bound on the interval minimum.

        Endpoint balls are enlarged outward, so uncertain endpoints cannot
        omit any part of the requested interval.  Coefficients can be balls;
        the returned lower bound then holds for every represented function.
        """
        start = perf_counter()
        self.stats['intervals'] += 1
        left, right = (ball(left).lower(), ball(right).upper())
        tol = ball(tolerance)
        if not tol > 0:
            raise ValueError('tolerance must be certified positive')
        if not right >= left:
            raise ValueError('interval endpoints must satisfy left <= right')
        if left == right:
            value = self.value(left)
            if not value.upper() - value.lower() <= tol:
                raise ArithmeticError('input uncertainty exceeds the requested tolerance')
            result = value.lower()
        else:
            coefficients = _restrict(self.bernstein, left, right)
            result = self._visit(coefficients, left, right, tol, 0, max_depth)
        self.stats['seconds'] += perf_counter() - start
        return result

    def lower_bounds(self, edges, tolerance='1e-9', max_depth=40):
        """Return bounds on successive intervals of an increasing edge list."""
        edges = list(edges)
        return [self.lower_bound(left, right, tolerance, max_depth) for left, right in zip(edges, edges[1:])]

    def _visit(self, coefficients, left, right, tolerance, depth, max_depth):
        stats = self.stats
        stats['nodes'] += 1
        stats['maximum_depth'] = max(stats['maximum_depth'], depth)
        n = self.degree
        width = right - left
        exponential_left = (-self.lam * left).exp()
        exponential_right = (-self.lam * right).exp()
        exponential_range = _hull_bounds(exponential_right.lower(), exponential_left.upper())
        value_left = coefficients[0] + self.exp_coefficient * exponential_left
        value_right = coefficients[-1] + self.exp_coefficient * exponential_right
        endpoint_lower = min(value_left.lower(), value_right.lower())
        endpoint_upper = min(value_left.upper(), value_right.upper())
        if n == 0:
            if endpoint_upper - endpoint_lower <= tolerance:
                stats['monotone'] += 1
                return endpoint_lower
        else:
            derivative = [n * (coefficients[j + 1] - coefficients[j]) / width for j in range(n)]
            first = _coefficient_range(derivative) - self.lam * self.exp_coefficient * exponential_range
            if first >= 0:
                if value_left.upper() - value_left.lower() <= tolerance:
                    stats['monotone'] += 1
                    return value_left.lower()
            elif first <= 0:
                if value_right.upper() - value_right.lower() <= tolerance:
                    stats['monotone'] += 1
                    return value_right.lower()
        second = [n * (n - 1) * (coefficients[j + 2] - 2 * coefficients[j + 1] + coefficients[j]) / width ** 2 for j in range(max(n - 1, 0))]
        second_range = _coefficient_range(second) + self.lam ** 2 * self.exp_coefficient * exponential_range
        curvature = max(arb(0), second_range.upper())
        if curvature == 0:
            lower = endpoint_lower
            kind = 'concave'
        else:
            lower = (endpoint_lower - curvature * width ** 2 / 8).lower()
            kind = 'chord'
        if endpoint_upper - lower <= tolerance:
            stats[kind] += 1
            return lower
        if depth >= max_depth:
            raise ArithmeticError('minimum bound did not reach tolerance; increase precision or max_depth')
        stats['subdivisions'] += 1
        midpoint = (left + right) / 2
        left_coefficients, right_coefficients = _split(coefficients, arb(1) / 2)
        return min(self._visit(left_coefficients, left, midpoint, tolerance, depth + 1, max_depth), self._visit(right_coefficients, midpoint, right, tolerance, depth + 1, max_depth))


# Exact interval-tail LPs in cumulative interval-mass coordinates.

def interval_dyadic(a, upper=False, bits=60):
    a = a.upper() if upper else a.lower()
    m, e = a.man_exp()
    q = Q(int(m)) * Q(2) ** int(e)
    n = q.numerator * 2 ** bits
    return Q(-(-n // q.denominator) if upper else n // q.denominator, 2 ** bits)

def interval_add(*p):
    return [sum((a[k] if k < len(a) else 0 for a in p)) for k in range(max(map(len, p)))]

def interval_scale(p, c):
    return [c * a for a in p]

def interval_horner(p, z):
    y = p[-1]
    for a in p[-2::-1]:
        y = y * z + a
    return y

def interval_polynomial_curvature(p):
    n = len(p) - 1
    if n < 2:
        return arb(0)
    b = [sum((p[k] * comb(i, k) / comb(n, k) for k in range(i + 1))) for i in range(n + 1)]
    return n * (n - 1) * max(((b[i + 2] - 2 * b[i + 1] + b[i]).abs_upper() for i in range(n - 1)))

def accounting_families():
    f, g, h = load_prices(BASE / 'candidate.json')
    fp = [to_power(a) for a in f]
    gp = [to_power(a) for a in g]
    hp = to_power(h)
    ans = []
    for name, c0, c1 in small_costs(fp, gp, hp):
        name = name.replace('load_', 'small:').replace('_', ':')
        if name == 'continuation:start':
            name = 'small:continuation_start'
        if name == 'continuation:slope':
            name = 'small:continuation_slope'
        ans.append((name, (list(map(ball, c0)), arb(0)), None if name == 'small:continuation_slope' else (list(map(ball, c1)), arb(0))))
    mu = (-ball(LAM)).exp()
    lam = ball(LAM)
    for k, s in enumerate(KNOTS):
        F = list(map(ball, interval_add(fp[k], gp[k])))
        G = list(map(ball, fp[k]))
        F1 = sum(F)
        G1 = sum(G)
        for b in [Q(1, 2), Q(3, 4), Q(1)]:
            bb = ball(b)
            ss = ball(s)
            c0 = interval_add([-lam * j * a for j, a in enumerate(F)], interval_scale(F, -(lam + mu + bb)), interval_scale(G, (1 - bb) * mu), [lam * F1 + (ss + bb) * mu])
            e0 = lam * (ss + bb - F1 + (1 - bb) * G1)
            c1 = interval_add([-lam * j * a for j, a in enumerate(G)], interval_scale(G, -(lam + mu + bb)), [lam * G1 + mu])
            e1 = lam * (1 - G1)
            bid = 'mid' if b == Q(3, 4) else str(float(b))
            ans.append((f'large:0.0:{k}:{bid}', (c0, e0), (c1, e1)))
    H = list(map(ball, hp))
    H1 = sum(H)
    c = interval_add([-lam * j * a for j, a in enumerate(H)], interval_scale(H, -(lam + mu + 1)), [lam * H1 + mu])
    ans.append(('large:0.0:continuation', (c, lam * (1 - H1)), None))
    return ans

class IntervalGrid:

    def __init__(self, N, edges=None, score=True):
        self.N = N
        self.edges = [Q(i, N) for i in range(N + 1)] if edges is None else list(map(Q, edges))
        self.score = score
        if len(self.edges) != N + 1 or self.edges[0] != 0 or self.edges[-1] != 1 or any((a >= b for a, b in zip(self.edges, self.edges[1:]))):
            raise ValueError('The interval mesh must cover [0,1] with strictly increasing endpoints')
        hu = list(H)
        hu[0] = Q(103, 100) - sum(((-1) ** (k + 1) * LAM ** k / (k * factorial(k)) for k in range(1, 81))) + LAM ** 27 / (27 * factorial(27))
        hp = list(map(ball, hu))
        self.Hupper = hp
        self.z = list(map(ball, self.edges))
        self.ex = [(-ball(LAM) * z).exp() for z in self.z]
        self.H = [interval_dyadic(interval_horner(hp, z), True) for z in self.z[1:]]
        self.cache = {}

    def cone(self, scalar):
        if scalar in self.cache:
            return self.cache[scalar]
        N = self.N
        blocks = 1 if scalar else 2
        rr = []
        cc = []
        vv = []
        row = 0
        for b in range(blocks):
            for i in range(N - 1):
                rr.extend([row, row])
                cc.extend([b * N + i, b * N + i + 1])
                vv.extend([1.0, -1.0])
                row += 1
        for kind in range(blocks):
            for i in range(1, N):
                ratio = float(self.edges[i] / self.edges[i + 1])
                rr.extend([row, row])
                cc.extend([i, i - 1])
                vv.extend([ratio, -1.0])
                if kind:
                    rr.extend([row, row])
                    cc.extend([N + i, N + i - 1])
                    vv.extend([2 * ratio, -2.0])
                row += 1
        hc = [self.H[i + 1] - self.H[i] for i in range(N - 1)] + [-self.H[-1]]
        if self.score:
            rr.extend([row] * N)
            cc.extend(range(N))
            vv.extend(map(float, hc))
            row += 1
        A = csr_matrix((vv, (rr, cc)), shape=(row, blocks * N))
        eq = np.zeros((1, blocks * N))
        eq[0, N - 1] = 1
        if not scalar:
            eq[0, 2 * N - 1] = 1
        self.cache[scalar] = (A, eq, hc)
        return self.cache[scalar]

    def lower_cost(self, ce, tight=False):
        p, e = ce
        B = interval_dyadic(interval_polynomial_curvature(p) + e.abs_upper() * ball(LAM) ** 2, True)
        if tight:
            bounder = PolynomialExponential(p, e)
            vals = bounder.lower_bounds(self.edges, tolerance='1e-10')
            return ([interval_dyadic(a) for a in vals], B, Q(0))
        vals = [interval_dyadic(interval_horner(p, z) + e * ee) for z, ee in zip(self.z, self.ex)]
        corrections = [B * (b - a) ** 2 / 8 for a, b in zip(self.edges, self.edges[1:])]
        return ([min(a, b) - e for a, b, e in zip(vals, vals[1:], corrections)], B, max(corrections))

    def certify(self, c0, c1):
        scalar = c1 is None
        N = self.N
        blocks = 1 if scalar else 2
        costs = [c0] if scalar else [c0, c1]
        cq = []
        for c in costs:
            cq.extend([c[i] - c[i + 1] for i in range(N - 1)] + [c[-1]])
        A, eq, hc = self.cone(scalar)
        t0 = time.perf_counter()
        sol = linprog(np.array(cq, float), A_ub=A, b_ub=np.zeros(A.shape[0]), A_eq=eq, b_eq=[1], bounds=(0, None), method='highs', options={'dual_feasibility_tolerance': 1e-09, 'primal_feasibility_tolerance': 1e-09})
        elapsed = time.perf_counter() - t0
        if not sol.success:
            raise RuntimeError(sol.message)
        u = [max(Q(0), Q(round(-a * 2 ** 44), 2 ** 44)) for a in sol.ineqlin.marginals]
        t = Q(round(sol.eqlin.marginals[0] * 2 ** 44), 2 ** 44)
        residual = list(cq)
        row = 0
        for b in range(blocks):
            residual[(b + 1) * N - 1] -= t
            for i in range(N - 1):
                residual[b * N + i] += u[row]
                residual[b * N + i + 1] -= u[row]
                row += 1
        for kind in range(blocks):
            for i in range(1, N):
                ratio = self.edges[i] / self.edges[i + 1]
                residual[i] += ratio * u[row]
                residual[i - 1] -= u[row]
                if kind:
                    residual[N + i] += 2 * ratio * u[row]
                    residual[N + i - 1] -= 2 * u[row]
                row += 1
        if self.score:
            for i, a in enumerate(hc):
                residual[i] += u[row] * a
        raw = []
        for b in range(blocks):
            acc = Q(0)
            for a in residual[b * N:(b + 1) * N][::-1]:
                acc += a
                raw.append(acc)
        correction = min(raw)
        bound = t + correction
        return {'bound': str(bound), 'float_bound': float(bound), 'numerical_bound': sol.fun, 'dual_t': str(t), 'residual_correction': str(correction), 'lp_seconds': elapsed, 'variables': A.shape[1], 'rows': A.shape[0], 'nonzeros': A.nnz, 'dual_nonzero': [[i, str(a)] for i, a in enumerate(u) if a]}


# Coarse kernel fits shared by all 34 accounting families at an endpoint.

def local_arb(p, l, u, M=12):
    l, u = (ball(l), ball(u))
    d = len(p) - 1
    aa = [(u - l) ** k * sum((p[j] * comb(j, k) * l ** (j - k) for j in range(k, d + 1))) for k in range(d + 1)]
    return [sum((aa[k] * comb(i, k) / comb(M, k) for k in range(min(i, d) + 1))) for i in range(M + 1)]

def gap_power(p, x):
    r = (-ball(LAM) * x).exp()
    v = [r ** (j + 1) * ball(c) for j, c in enumerate(p)]
    v[0] += sum((ball(c) * (1 - r ** (j + 1)) / (j + 1) for j, c in enumerate(p)))
    return v

class Endpoint:

    def __init__(self, path, N=12):
        self.baseN = N
        self.M = 12
        self.f, self.g, self.h = load_prices(path)
        self.polys = [to_power([Q(1)] * 13)] + [to_power(v) for pair in zip(self.f, self.g) for v in pair] + [to_power(self.h)]
        self.bp = [[Q(1)] * 13] + [v for pair in zip(self.f, self.g) for v in pair] + [self.h]
        self.prepare(Q(0))

    def transform(self, values):
        return np.einsum('ij,kj->ki', self.inverse, values.reshape(self.N, 13)).ravel()

    def values(self, xq):
        x = ball(xq)
        N = self.N
        if xq == 0:
            exponential = []
            for i in range(N):
                l, u = map(ball, self.edges[i:i + 2])
                mid = (l + u) / 2
                half = (u - l) / 2
                tay = [(-ball(LAM) * mid).exp() * (-ball(LAM)) ** k / factorial(k) for k in range(13)]
                error = (-ball(LAM) * l).exp() * ball(LAM) ** 13 * half ** 13 / factorial(13)
                exponential.extend((arb(v, error.abs_upper()) for v in taylor_to_bernstein(tay, l, u)))
            mu = (-ball(LAM)).exp()
            L = ball(LAM)
            M = []
            G = []
            for p, P in zip(self.polys, self.P):
                deriv = [-L * k * ball(p[k]) for k in range(len(p))]
                dv = [v for i in range(N) for v in local_arb(deriv, *self.edges[i:i + 2])]
                p1 = ball(sum(p))
                M.append([d - (L + mu) * v + L * (1 - e) * p1 for d, v, e in zip(dv, P, exponential)])
                G.append([d - L * v + L * p1 for d, v in zip(dv, P)])
            W = [mu + L * v for v in exponential]
            return (M, G, W, [])
        G = [[v for i in range(N) for v in local_arb(gap_power(p, x), *self.edges[i:i + 2])] for p in self.polys]
        if xq == 1:
            return ([[arb(0)] * self.m for _ in self.polys], G, [arb(1)] * self.m, [])
        mat, _, _ = numerical_kernel(self.z, float(xq), nquad=64)
        score = score_segments(x)
        M = []
        errors = []
        for p, bp in zip(self.polys, self.bp):
            v = self.transform(mat @ np.array(bp, float))
            centers = [Q(round(float(a) * 2 ** 44), 2 ** 44) for a in v]
            approximations = [(ball(self.edges[i]), ball(self.edges[i + 1]), [ball(t) for t in centers[13 * i:13 * (i + 1)]]) for i in range(N)]
            err = approximation_error(x, [ball(q) for q in p], approximations, score=score, pnorm=ball(max((abs(v) for v in bp))))
            M.append([arb(ball(a), e.abs_upper()) for i, e in enumerate(err) for a in centers[13 * i:13 * (i + 1)]])
            errors.append([float(v) for v in err])
        return (M, G, [1 - v for v in M[0]], errors)


# Bernstein subdivision gives constant lower costs on arbitrary state cells.

def _bernstein_split(p, t):
    p = list(p)
    left = [p[0]]
    right = [p[-1]]
    while len(p) > 1:
        p = [(1 - t) * a + t * b for a, b in zip(p, p[1:])]
        left.append(p[0])
        right.append(p[-1])
    return (left, right[::-1])

def _bernstein_restrict(p, a, b):
    if a == 0 and b == 1:
        return p
    if b != 1:
        p = _bernstein_split(p, ball(b))[0]
    if a:
        p = _bernstein_split(p, ball(a / b))[1]
    return p

def elevate(p, d):
    n = len(p) - 1
    return [sum((ball(Q(comb(n, j) * comb(d - n, k - j), comb(d, k))) * p[j] for j in range(max(0, k - (d - n)), min(n, k) + 1))) for k in range(d + 1)]

def interval_constants(pieces, coarse, edges):
    ans = []
    ci = 0
    for a, b in zip(edges, edges[1:]):
        while coarse[ci + 1] <= a:
            ci += 1
        j = ci
        bounds = []
        while j < len(pieces) and coarse[j] < b:
            l, u = coarse[j:j + 2]
            aa = max(l, a)
            bb = min(u, b)
            if aa < bb:
                pp = _bernstein_restrict(pieces[j], (aa - l) / (u - l), (bb - l) / (u - l))
                bounds.append(min((interval_dyadic(q) for q in pp)))
            j += 1
        ans.append(min(bounds))
    return ans

def analytic_score():
    L = Q(9, 2)
    h = list(H)
    h[0] = Q(103, 100) - sum(((-1) ** (k + 1) * L ** k / (k * factorial(k)) for k in range(1, 81))) + L ** 27 / (27 * factorial(27))
    return h


# Analytic raw-gap mass-curvature bounds, rounded upward as exact rationals.

def curvature_ball(q):
    q = Q(q)
    return arb(q.numerator) / q.denominator

def upper_fraction(a):
    m, e = a.upper().man_exp()
    return Q(int(m)) * Q(2) ** int(e)

def age_derivative(p):
    return [-LAM * k * v for k, v in enumerate(p)]

def gap_generator(p):
    """L_G p in z coordinates: -lambda*z*p'(z)+lambda*(p(1)-p(z))."""
    out = [-LAM * (k + 1) * v for k, v in enumerate(p)]
    out[0] += LAM * sum(p)
    return out

def polynomial_norm(p, subdivisions=16):
    """Exact sup-norm upper bound by rational local Bernstein coefficients."""
    degree = len(p) - 1
    return max((abs(v) for i in range(subdivisions) for v in local_bern(p, Q(i, subdivisions), Q(i + 1, subdivisions), degree)))

def polynomial_coefficients(p, subdivisions=16):
    return [v for i in range(subdivisions) for v in local_bern(p, Q(i, subdivisions), Q(i + 1, subdivisions), len(p) - 1)]

@lru_cache(maxsize=None)
def price_norms(bernstein):
    p = to_power(list(bernstein))
    dp = age_derivative(p)
    ddp = age_derivative(dp)
    gp = gap_generator(p)
    ggp = gap_generator(gp)
    gap0 = [sum((v / Q(k + 1) for k, v in enumerate(p))), Q(0)]
    gap0.extend((Q(k, k + 1) * p[k] for k in range(1, len(p))))
    gap1 = age_derivative(gap0)
    gap2 = age_derivative(gap1)
    pc = polynomial_coefficients(p)
    gc = polynomial_coefficients(gp)
    ggc = polynomial_coefficients(ggp)
    return {'F0': max(map(abs, pc)), 'F1': polynomial_norm(dp), 'F2': polynomial_norm(ddp), 'f0': abs(sum(p)), 'd0': abs(sum(dp)), 'g1': polynomial_norm(gap1), 'g2': polynomial_norm(gap2), 'LG1': max(map(abs, gc)), 'LG2': max(map(abs, ggc)), 'minimum': min(pc), 'maximum': max(pc), 'value_at_zero_age': sum(p), 'LG1coeff': gc, 'LG2coeff': ggc}

def kernel_bounds(bernstein, l, u, scale=Q(1), subtract=Q(0)):
    """Return exact rational upper B1,B2 for M_x p, x in [l,u]."""
    l, u = (Q(l), Q(u))
    assert 0 <= l <= u <= 1
    n = dict(price_norms(tuple(map(Q, bernstein))))
    assert scale >= 0
    if scale != 1 or subtract != 0:
        n['F0'] = max(abs(scale * n['minimum'] - subtract), abs(scale * n['maximum'] - subtract))
        n['f0'] = abs(scale * n['value_at_zero_age'] - subtract)
        for key in ['F1', 'F2', 'd0', 'g1', 'g2']:
            n[key] *= scale
    L = curvature_ball(LAM)
    lb = curvature_ball(l)
    ub = curvature_ball(u)
    rl = (-L * lb).exp()
    ru = (-L * ub).exp()
    mu = (-L).exp()
    U = rl - mu * lb
    R = 1 - (-L * (1 - lb)).exp()
    a1 = max((1 - l) * LAM ** 2 / 2, LAM + (1 - l) * LAM ** 2 / 8)
    a2 = a1 ** 2 + LAM ** 2 + 2 * (1 - l) * LAM ** 3 / 3
    d = 1 - u

    def damped_power(k):
        t = LAM if d == 0 or d * LAM <= k else Q(k) / d
        return curvature_ball(t) ** k * (-curvature_ball(d * t)).exp()
    cp = 1 + (1 - l) * LAM
    cpp = 2 * LAM + 2 * (1 - l) * LAM ** 2
    a1 = min(a1, upper_fraction(curvature_ball(cp) * damped_power(1)))
    a2 = min(a2, upper_fraction(curvature_ball(cp ** 2) * damped_power(2) + curvature_ball(cpp) * damped_power(1)))
    A1, A2 = (curvature_ball(a1), curvature_ball(a2))
    Av = L / (arb(1).exp() - 1)
    F0, F1, F2, f0, d0, g1, g2 = [curvature_ball(n[k]) for k in ['F0', 'F1', 'F2', 'f0', 'd0', 'g1', 'g2']]
    B1 = (L * rl + mu) * F0 + U * F1 + L * rl * R * f0 + (1 - ru) * (A1 * F0 + R * g1)
    jump_damping = (-curvature_ball(1 - u) * L * (-L * ub).exp()).exp()
    B2 = L * L * rl * F0 + 2 * (L * rl + mu) * F1 + U * F2 + L * rl * ((2 * A1 + Av + L * R) * f0 + R * d0) + (1 - ru) * (A2 * F0 + 2 * A1 * g1 + R * g2) + (1 - lb) * L ** 3 * F0 * jump_damping
    return (upper_fraction(B1), upper_fraction(B2))

def family_bounds(f, g, h, l, u):
    """Keys 'j:1/2','j:3/4','j:1','continuation'; values exact Fractions."""
    l, u = (Q(l), Q(u))
    B11, B21 = kernel_bounds([Q(1)], l, u)
    out = {}
    for j, s in enumerate(KNOTS):
        f1, f2 = kernel_bounds(f[j], l, u)
        g1, g2 = kernel_bounds(g[j], l, u)
        norms = price_norms(tuple(map(Q, f[j])))
        LG1, LG2 = (norms['LG1'], norms['LG2'])
        endpoints = []
        for b in [Q(1, 2), Q(1)]:
            Bnu1 = B21 + (1 - b * l) * f2 + 2 * b * f1
            Bnu0 = (s + b) * B21 + (1 - b * l) * (b * f2 + (1 - b) * LG2 + g2) + 2 * b * (b * f1 + (1 - b) * LG1 + g1)
            hh = [b * v + w for v, w in zip(f[j], g[j])]
            h1, _ = kernel_bounds(hh, l, u)
            tight1 = max((kernel_bounds(f[j], l, u, a, Q(1))[1] for a in [1 - b * l, 1 - b * u])) + 2 * b * f1
            tight0 = max((kernel_bounds(hh, l, u, a, s + b)[1] for a in [1 - b * l, 1 - b * u])) + 2 * b * h1
            gap_bound = max((abs(a * v - 2 * b * w) for a in [1 - b * l, 1 - b * u] for v, w in zip(norms['LG2coeff'], norms['LG1coeff'])))
            tight0 += (1 - b) * gap_bound
            Bnu0 = min(Bnu0, tight0)
            Bnu1 = min(Bnu1, tight1)
            out[f'{j}:{b}'] = max(Bnu0, Bnu1)
            endpoints.append((Bnu0, Bnu1))
        correction = (2 * (LG1 + f1) + u * (LG2 + f2)) / 8
        mid0 = (endpoints[0][0] + endpoints[1][0]) / 2 + correction
        mid1 = (endpoints[0][1] + endpoints[1][1]) / 2
        out[f'{j}:3/4'] = max(mid0, mid1)
    h1, h2 = kernel_bounds(h, l, u)
    old = B21 + (1 - l) * h2 + 2 * h1
    tight = max((kernel_bounds(h, l, u, a, Q(1))[1] for a in [1 - l, 1 - u])) + 2 * h1
    out['continuation'] = min(old, tight)
    return out


# Initial meshes and score multipliers are suggestions, never certificate data.
# Each mesh is a run-length sequence (count, rational width), starting at zero.
# The 53 meshes and 362 proposals reproduce the previous seed files exactly.
# Every accounting cost and dual residual is freshly recomputed by the verifier.
_SEED_MESH_RUNS = [

    ((20, '1/128'), (2, '1/64'), (18, '1/128'), (3, '1/64'), (13, '1/32'), (12, '1/64'), (4, '1/128')),

    ((10, '1/64'), (1, '1/32'), (10, '1/64'), (15, '1/32'), (12, '1/64')),

    ((32, '1/1024'), (4, '1/512'), (3, '1/256'), (20, '1/512'), (1, '1/256'), (4, '1/128'), (46, '1/512'), (5, '1/256'), (2, '1/128'), (15, '1/256'), (2, '1/512'), (16, '1/256'), (2, '1/128'), (1, '1/64'), (7, '1/32'), (44, '1/256'), (4, '1/128'), (3, '1/64'), (9, '1/128'), (5, '1/256'), (18, '1/512')),

    ((32, '1/32'),),

    ((1, '1/512'), (17, '1/1024'), (108, '1/2048'), (55, '1/1024'), (10, '1/512'), (6, '1/256'), (106, '1/512'), (4, '1/256'), (1, '1/64'), (6, '1/32'), (8, '1/256'), (4, '1/128'), (2, '1/64'), (12, '1/128'), (2, '1/64'), (2, '1/32'), (13, '1/128'), (5, '1/256'), (2, '1/512')),

    ((22, '1/64'), (15, '1/32'), (10, '1/64'), (1, '1/32')),

    ((2, '1/128'), (36, '1/256'), (4, '1/128'), (1, '1/64'), (22, '1/128'), (2, '1/64'), (7, '1/32'), (10, '1/128'), (9, '1/64'), (1, '1/32'), (1, '1/64'), (10, '1/128'), (1, '1/64'), (2, '1/128')),

    ((2, '1/512'), (14, '1/1024'), (152, '1/2048'), (54, '1/1024'), (10, '1/512'), (5, '1/256'), (114, '1/512'), (4, '1/256'), (1, '1/64'), (5, '1/32'), (1, '1/128'), (2, '1/256'), (90, '1/512'), (7, '1/256'), (1, '1/128'), (7, '1/256'), (76, '1/512'), (1, '1/256')),

    ((36, '1/1024'), (6, '1/512'), (2, '1/256'), (20, '1/512'), (4, '1/256'), (2, '1/128'), (4, '1/256'), (46, '1/512'), (7, '1/256'), (2, '1/128'), (11, '1/256'), (30, '1/512'), (8, '1/32'), (26, '1/256'), (5, '1/128'), (1, '1/64'), (4, '1/128'), (20, '1/256'), (6, '1/128'), (2, '1/64'), (1, '1/32')),

    ((192, '1/2048'), (38, '1/1024'), (7, '1/512'), (3, '1/256'), (14, '1/512'), (90, '1/1024'), (45, '1/512'), (1, '1/64'), (9, '1/32'), (1, '1/128'), (26, '1/256'), (120, '1/512')),

    ((172, '1/2048'), (66, '1/1024'), (12, '1/512'), (5, '1/256'), (25, '1/512'), (58, '1/1024'), (36, '1/512'), (1, '1/128'), (9, '1/32'), (1, '1/64'), (7, '1/256'), (154, '1/512')),

    ((58, '1/8192'), (255, '1/4096'), (72, '1/2048'), (15, '1/1024'), (3, '1/512'), (1, '1/256'), (3, '1/512'), (206, '1/1024'), (10, '1/512'), (1, '1/256'), (1, '1/64'), (9, '1/32'), (1, '1/128'), (9, '1/256'), (133, '1/512'), (42, '1/1024')),

    ((34, '1/2048'), (43, '1/1024'), (16, '1/512'), (35, '1/256'), (3, '1/128'), (7, '1/256'), (26, '1/512'), (2, '1/128'), (9, '1/32'), (1, '1/64'), (1, '1/128'), (2, '1/256'), (167, '1/512'), (18, '1/1024')),

    ((82, '1/8192'), (297, '1/4096'), (87, '1/2048'), (20, '1/1024'), (4, '1/512'), (1, '1/256'), (5, '1/512'), (238, '1/1024'), (42, '1/512'), (5, '1/256'), (2, '1/128'), (1, '1/64'), (2, '1/32'), (4, '1/64'), (2, '1/32'), (2, '1/64'), (4, '1/128'), (3, '1/64'), (4, '1/128'), (1, '1/64'), (1, '1/32'), (1, '1/64'), (7, '1/128'), (2, '1/256'), (2, '1/128')),

    ((1000, '1/8192'), (930, '1/4096'), (309, '1/2048'), (32, '1/1024'), (212, '1/2048'), (196, '1/1024'), (67, '1/512'), (3, '1/256'), (1, '1/128'), (1, '1/256'), (8, '1/512'), (4, '1/1024')),

    ((38, '1/128'), (5, '1/64'), (4, '1/128'), (12, '1/64'), (13, '1/32')),

    ((58, '1/512'), (11, '1/256'), (2, '1/128'), (1, '1/64'), (4, '1/128'), (36, '1/256'), (14, '1/128'), (3, '1/32'), (12, '1/128'), (2, '1/64'), (1, '1/32'), (1, '1/64'), (10, '1/128'), (2, '1/64'), (5, '1/32')),

    ((128, '1/2048'), (170, '1/1024'), (135, '1/512'), (52, '1/256'), (3, '1/128'), (2, '1/64'), (16, '1/128'), (1, '1/64'), (3, '1/128'), (9, '1/256'), (25, '1/512'), (2, '1/1024')),

    ((20, '1/64'), (1, '1/32'), (4, '1/64'), (14, '1/32'), (4, '1/64'), (1, '1/32'), (4, '1/64')),

    ((24, '1/64'), (12, '1/32'), (4, '1/64'), (4, '1/32'), (4, '1/64')),

    ((56, '1/512'), (28, '1/256'), (4, '1/128'), (3, '1/64'), (10, '1/128'), (4, '1/64'), (32, '1/128'), (4, '1/64'), (1, '1/32'), (11, '1/64'), (3, '1/128'), (6, '1/256')),

    ((22, '1/64'), (11, '1/32'), (2, '1/64'), (4, '1/32'), (4, '1/64'), (1, '1/32'), (4, '1/64')),

    ((10, '1/64'), (1, '1/32'), (12, '1/64'), (10, '1/32'), (6, '1/64'), (5, '1/32'), (4, '1/64')),

    ((38, '1/128'), (13, '1/64'), (11, '1/32'), (10, '1/64')),

    ((10, '1/64'), (1, '1/32'), (10, '1/64'), (10, '1/32'), (4, '1/64'), (5, '1/32'), (8, '1/64')),

    ((14, '1/64'), (2, '1/32'), (6, '1/64'), (4, '1/32'), (10, '1/64'), (10, '1/32'), (2, '1/64')),

    ((28, '1/128'), (6, '1/64'), (2, '1/128'), (5, '1/64'), (5, '1/32'), (12, '1/64'), (2, '1/32'), (8, '1/64'), (1, '1/32'), (1, '1/64'), (2, '1/128')),

    ((16, '1/64'), (2, '1/32'), (16, '1/64'), (14, '1/32')),

    ((18, '1/64'), (2, '1/32'), (6, '1/64'), (3, '1/32'), (2, '1/64'), (11, '1/32'), (6, '1/64')),

    ((16, '1/64'), (2, '1/32'), (10, '1/64'), (5, '1/32'), (2, '1/64'), (9, '1/32'), (4, '1/64')),

    ((16, '1/64'), (1, '1/32'), (12, '1/64'), (12, '1/32'), (2, '1/64'), (2, '1/32'), (2, '1/64'), (1, '1/32')),

    ((16, '1/64'), (1, '1/32'), (12, '1/64'), (15, '1/32'), (4, '1/64')),

    ((140, '1/2048'), (50, '1/1024'), (12, '1/512'), (2, '1/256'), (1, '1/128'), (4, '1/256'), (98, '1/512'), (27, '1/256'), (4, '1/128'), (2, '1/64'), (6, '1/32'), (10, '1/64'), (1, '1/32'), (5, '1/64'), (2, '1/128')),

    ((446, '1/4096'), (441, '1/2048'), (164, '1/1024'), (8, '1/512'), (8, '1/256'), (64, '1/512'), (28, '1/256'), (26, '1/512'), (35, '1/256'), (4, '1/128'), (4, '1/256')),

    ((130, '1/1024'), (101, '1/512'), (81, '1/256'), (10, '1/128'), (1, '1/32'), (1, '1/64'), (10, '1/128'), (4, '1/64'), (4, '1/128'), (10, '1/256'), (12, '1/512')),

    ((22, '1/64'), (1, '1/32'), (2, '1/64'), (14, '1/32'), (4, '1/64'), (1, '1/32'), (4, '1/64')),

    ((54, '1/512'), (29, '1/256'), (4, '1/128'), (3, '1/64'), (10, '1/128'), (4, '1/64'), (34, '1/128'), (3, '1/64'), (1, '1/32'), (11, '1/64'), (3, '1/128'), (6, '1/256')),

    ((22, '1/64'), (2, '1/32'), (6, '1/64'), (13, '1/32'), (4, '1/64'), (1, '1/32')),

    ((18, '1/64'), (2, '1/32'), (6, '1/64'), (3, '1/32'), (4, '1/64'), (11, '1/32'), (4, '1/64')),

    ((16, '1/64'), (2, '1/32'), (10, '1/64'), (6, '1/32'), (2, '1/64'), (8, '1/32'), (4, '1/64')),

    ((36, '1/256'), (2, '1/128'), (1, '1/64'), (4, '1/128'), (24, '1/256'), (18, '1/128'), (4, '1/64'), (7, '1/32'), (4, '1/64'), (6, '1/32'), (2, '1/64')),

    ((78, '1/512'), (53, '1/256'), (12, '1/128'), (3, '1/64'), (16, '1/32')),

    ((40, '1/128'), (20, '1/64'), (12, '1/32')),

    ((20, '1/128'), (4, '1/64'), (18, '1/128'), (7, '1/64'), (3, '1/32'), (8, '1/64'), (2, '1/32'), (4, '1/64'), (6, '1/32')),

    ((44, '1/256'), (32, '1/128'), (17, '1/64'), (5, '1/32'), (4, '1/64'), (1, '1/32'), (3, '1/64'), (2, '1/128')),

    ((28, '1/64'), (14, '1/32'), (2, '1/64'), (2, '1/32'), (2, '1/64')),

    ((26, '1/64'), (12, '1/32'), (2, '1/64'), (4, '1/32'), (4, '1/64')),

    ((30, '1/128'), (3, '1/64'), (1, '1/32'), (4, '1/64'), (2, '1/32'), (6, '1/64'), (8, '1/128'), (8, '1/64'), (8, '1/32'), (2, '1/64')),

    ((24, '1/64'), (10, '1/32'), (4, '1/64'), (4, '1/32'), (2, '1/64'), (2, '1/32'), (2, '1/64')),

    ((24, '1/64'), (1, '1/32'), (6, '1/64'), (14, '1/32'), (2, '1/64'), (1, '1/32')),

    ((30, '1/128'), (5, '1/64'), (2, '1/128'), (5, '1/64'), (6, '1/32'), (12, '1/64'), (2, '1/32'), (10, '1/64')),

    ((28, '1/64'), (16, '1/32'), (4, '1/64')),

    ((32, '1/64'), (16, '1/32')),

]

# (request mass or "small", accounting family, mesh index, rational multiplier)
_SEED_ROWS = [

    ('small', 'small:0:0', 0, '0'),

    ('small', 'small:0:1', 1, '44370905161/1099511627776'),

    ('small', 'small:1:1', 2, '0'),

    ('small', 'small:1:2', 3, '48761977637/1099511627776'),

    ('small', 'small:2:2', 4, '0'),

    ('small', 'small:2:3', 5, '58420865125/1099511627776'),

    ('small', 'small:3:3', 6, '0'),

    ('small', 'small:3:4', 3, '66106971865/1099511627776'),

    ('small', 'small:4:4', 7, '0'),

    ('small', 'small:4:5', 3, '75738330037/1099511627776'),

    ('small', 'small:5:5', 8, '150910638449/549755813888'),

    ('small', 'small:5:6', 3, '85942114703/274877906944'),

    ('small', 'small:6:6', 9, '120294721299/274877906944'),

    ('small', 'small:6:7', 3, '266135575069/549755813888'),

    ('small', 'small:7:7', 10, '53775210897/137438953472'),

    ('small', 'small:7:8', 3, '254980116561/549755813888'),

    ('small', 'small:8:8', 11, '261049721757/549755813888'),

    ('small', 'small:8:9', 3, '154809074789/274877906944'),

    ('small', 'small:9:9', 3, '36575972727/68719476736'),

    ('small', 'small:9:10', 3, '681055048439/1099511627776'),

    ('small', 'small:continuation_start', 12, '735180952231/1099511627776'),

    ('small', 'small:continuation_slope', 3, '705163996545/1099511627776'),

    ('0', '0:1/2', 13, '2185595799/68719476736'),

    ('0', '0:1', 14, '0'),

    ('0', '0:3/4', 15, '0'),

    ('0', '1:1/2', 16, '76226318679/1099511627776'),

    ('0', '1:1', 17, '0'),

    ('0', '1:3/4', 18, '0'),

    ('0', '2:1/2', 19, '64354257385/549755813888'),

    ('0', '2:1', 20, '0'),

    ('0', '2:3/4', 21, '44008173697/1099511627776'),

    ('0', '3:1/2', 22, '182536634707/1099511627776'),

    ('0', '3:1', 23, '0'),

    ('0', '3:3/4', 3, '54878688087/1099511627776'),

    ('0', '4:1/2', 24, '260104046581/1099511627776'),

    ('0', '4:1', 25, '0'),

    ('0', '4:3/4', 3, '84172349499/1099511627776'),

    ('0', '5:1/2', 3, '181028905885/549755813888'),

    ('0', '5:1', 26, '0'),

    ('0', '5:3/4', 3, '65075439255/549755813888'),

    ('0', '6:1/2', 3, '199511570237/549755813888'),

    ('0', '6:1', 27, '0'),

    ('0', '6:3/4', 3, '1203108699/8589934592'),

    ('0', '7:1/2', 3, '387143320559/1099511627776'),

    ('0', '7:1', 28, '2840461633/274877906944'),

    ('0', '7:3/4', 3, '146053304221/1099511627776'),

    ('0', '8:1/2', 3, '209168109517/549755813888'),

    ('0', '8:1', 29, '0'),

    ('0', '8:3/4', 3, '5786109949/34359738368'),

    ('0', '9:1/2', 3, '424677494673/1099511627776'),

    ('0', '9:1', 30, '0'),

    ('0', '9:3/4', 3, '3075119773/17179869184'),

    ('0', '10:1/2', 3, '514986585499/1099511627776'),

    ('0', '10:1', 31, '5489304741/549755813888'),

    ('0', '10:3/4', 3, '252425103137/1099511627776'),

    ('0', 'continuation', 3, '20118373/274877906944'),

    ('1/1000', '0:1/2', 32, '17318363693/549755813888'),

    ('1/1000', '0:1', 33, '0'),

    ('1/1000', '0:3/4', 15, '0'),

    ('1/1000', '1:1/2', 16, '75469805745/1099511627776'),

    ('1/1000', '1:1', 34, '0'),

    ('1/1000', '1:3/4', 35, '0'),

    ('1/1000', '2:1/2', 19, '127771965783/1099511627776'),

    ('1/1000', '2:1', 36, '0'),

    ('1/1000', '2:3/4', 21, '21833219173/549755813888'),

    ('1/1000', '3:1/2', 22, '90491511351/549755813888'),

    ('1/1000', '3:1', 37, '0'),

    ('1/1000', '3:3/4', 3, '54018797997/1099511627776'),

    ('1/1000', '4:1/2', 24, '258810069321/1099511627776'),

    ('1/1000', '4:1', 25, '0'),

    ('1/1000', '4:3/4', 3, '41524628989/549755813888'),

    ('1/1000', '5:1/2', 3, '22498930109/68719476736'),

    ('1/1000', '5:1', 26, '0'),

    ('1/1000', '5:3/4', 3, '128953785677/1099511627776'),

    ('1/1000', '6:1/2', 3, '198617562601/549755813888'),

    ('1/1000', '6:1', 27, '0'),

    ('1/1000', '6:3/4', 3, '76031781695/549755813888'),

    ('1/1000', '7:1/2', 3, '3011463755/8589934592'),

    ('1/1000', '7:1', 38, '2729114983/274877906944'),

    ('1/1000', '7:3/4', 3, '144697909979/1099511627776'),

    ('1/1000', '8:1/2', 3, '416241297897/1099511627776'),

    ('1/1000', '8:1', 39, '0'),

    ('1/1000', '8:3/4', 3, '22955689181/137438953472'),

    ('1/1000', '9:1/2', 3, '420557145277/1099511627776'),

    ('1/1000', '9:1', 30, '0'),

    ('1/1000', '9:3/4', 3, '194690286397/1099511627776'),

    ('1/1000', '10:1/2', 3, '128166557679/274877906944'),

    ('1/1000', '10:1', 31, '8425529029/1099511627776'),

    ('1/1000', '10:3/4', 3, '62518490427/274877906944'),

    ('1/1000', 'continuation', 3, '0'),

    ('1/100', '0:1/2', 40, '8178254027/274877906944'),

    ('1/100', '0:1', 41, '0'),

    ('1/100', '0:3/4', 42, '0'),

    ('1/100', '1:1/2', 43, '70528312575/1099511627776'),

    ('1/100', '1:1', 44, '0'),

    ('1/100', '1:3/4', 45, '0'),

    ('1/100', '2:1/2', 46, '59585679081/549755813888'),

    ('1/100', '2:1', 47, '0'),

    ('1/100', '2:3/4', 48, '8823010941/274877906944'),

    ('1/100', '3:1/2', 3, '176431951455/1099511627776'),

    ('1/100', '3:1', 49, '0'),

    ('1/100', '3:3/4', 3, '50771445945/1099511627776'),

    ('1/100', '4:1/2', 3, '62984514687/274877906944'),

    ('1/100', '4:1', 25, '0'),

    ('1/100', '4:3/4', 3, '38015478753/549755813888'),

    ('1/100', '5:1/2', 3, '170249032443/549755813888'),

    ('1/100', '5:1', 50, '0'),

    ('1/100', '5:3/4', 3, '57811081777/549755813888'),

    ('1/100', '6:1/2', 3, '376760448531/1099511627776'),

    ('1/100', '6:1', 27, '0'),

    ('1/100', '6:3/4', 3, '67083777925/549755813888'),

    ('1/100', '7:1/2', 3, '370242092803/1099511627776'),

    ('1/100', '7:1', 3, '4084879911/549755813888'),

    ('1/100', '7:3/4', 3, '125502195207/1099511627776'),

    ('1/100', '8:1/2', 3, '99374912641/274877906944'),

    ('1/100', '8:1', 3, '0'),

    ('1/100', '8:3/4', 3, '167760286285/1099511627776'),

    ('1/100', '9:1/2', 3, '99823581103/274877906944'),

    ('1/100', '9:1', 3, '0'),

    ('1/100', '9:3/4', 3, '175169986493/1099511627776'),

    ('1/100', '10:1/2', 3, '29952098031/68719476736'),

    ('1/100', '10:1', 3, '0'),

    ('1/100', '10:3/4', 3, '223804075847/1099511627776'),

    ('1/100', 'continuation', 3, '0'),

    ('1/20', '0:1/2', 51, '12724863583/549755813888'),

    ('1/20', '0:1', 52, '0'),

    ('1/20', '0:3/4', 3, '0'),

    ('1/20', '1:1/2', 3, '46870191671/1099511627776'),

    ('1/20', '1:1', 52, '0'),

    ('1/20', '1:3/4', 3, '0'),

    ('1/20', '2:1/2', 3, '91260293777/1099511627776'),

    ('1/20', '2:1', 3, '0'),

    ('1/20', '2:3/4', 3, '0'),

    ('1/20', '3:1/2', 3, '133226419959/1099511627776'),

    ('1/20', '3:1', 3, '0'),

    ('1/20', '3:3/4', 3, '13257889873/549755813888'),

    ('1/20', '4:1/2', 3, '199715235791/1099511627776'),

    ('1/20', '4:1', 3, '0'),

    ('1/20', '4:3/4', 3, '27532218999/1099511627776'),

    ('1/20', '5:1/2', 3, '274713570231/1099511627776'),

    ('1/20', '5:1', 3, '0'),

    ('1/20', '5:3/4', 3, '63944104967/1099511627776'),

    ('1/20', '6:1/2', 3, '304024364515/1099511627776'),

    ('1/20', '6:1', 3, '0'),

    ('1/20', '6:3/4', 3, '2504911101/34359738368'),

    ('1/20', '7:1/2', 3, '297357287349/1099511627776'),

    ('1/20', '7:1', 3, '0'),

    ('1/20', '7:3/4', 3, '37489856347/549755813888'),

    ('1/20', '8:1/2', 3, '316854611271/1099511627776'),

    ('1/20', '8:1', 3, '0'),

    ('1/20', '8:3/4', 3, '45064157303/549755813888'),

    ('1/20', '9:1/2', 3, '318398189191/1099511627776'),

    ('1/20', '9:1', 3, '0'),

    ('1/20', '9:3/4', 3, '5762528907/68719476736'),

    ('1/20', '10:1/2', 3, '194261807157/549755813888'),

    ('1/20', '10:1', 3, '0'),

    ('1/20', '10:3/4', 3, '133854187883/1099511627776'),

    ('1/20', 'continuation', 3, '0'),

    ('1/10', '0:1/2', 3, '16765400217/1099511627776'),

    ('1/10', '0:1', 3, '0'),

    ('1/10', '0:3/4', 3, '0'),

    ('1/10', '1:1/2', 3, '15552682639/1099511627776'),

    ('1/10', '1:1', 3, '0'),

    ('1/10', '1:3/4', 3, '0'),

    ('1/10', '2:1/2', 3, '50154387719/1099511627776'),

    ('1/10', '2:1', 3, '0'),

    ('1/10', '2:3/4', 3, '0'),

    ('1/10', '3:1/2', 3, '89210212935/1099511627776'),

    ('1/10', '3:1', 3, '0'),

    ('1/10', '3:3/4', 3, '0'),

    ('1/10', '4:1/2', 3, '18241977641/137438953472'),

    ('1/10', '4:1', 3, '0'),

    ('1/10', '4:3/4', 3, '0'),

    ('1/10', '5:1/2', 3, '102048384887/549755813888'),

    ('1/10', '5:1', 3, '0'),

    ('1/10', '5:3/4', 3, '5119284735/549755813888'),

    ('1/10', '6:1/2', 3, '28250302469/137438953472'),

    ('1/10', '6:1', 3, '0'),

    ('1/10', '6:3/4', 3, '4595801583/549755813888'),

    ('1/10', '7:1/2', 3, '217681487407/1099511627776'),

    ('1/10', '7:1', 3, '0'),

    ('1/10', '7:3/4', 3, '6640140815/549755813888'),

    ('1/10', '8:1/2', 3, '116491061569/549755813888'),

    ('1/10', '8:1', 3, '0'),

    ('1/10', '8:3/4', 3, '15871922981/1099511627776'),

    ('1/10', '9:1/2', 3, '235524557695/1099511627776'),

    ('1/10', '9:1', 3, '0'),

    ('1/10', '9:3/4', 3, '6243350741/274877906944'),

    ('1/10', '10:1/2', 3, '292058806737/1099511627776'),

    ('1/10', '10:1', 3, '0'),

    ('1/10', '10:3/4', 3, '22190010545/549755813888'),

    ('1/10', 'continuation', 3, '0'),

    ('1/5', '0:1/2', 3, '0'),

    ('1/5', '0:1', 3, '0'),

    ('1/5', '0:3/4', 3, '0'),

    ('1/5', '1:1/2', 3, '0'),

    ('1/5', '1:1', 3, '0'),

    ('1/5', '1:3/4', 3, '0'),

    ('1/5', '2:1/2', 3, '158457109/137438953472'),

    ('1/5', '2:1', 3, '0'),

    ('1/5', '2:3/4', 3, '0'),

    ('1/5', '3:1/2', 3, '6811131771/274877906944'),

    ('1/5', '3:1', 3, '0'),

    ('1/5', '3:3/4', 3, '0'),

    ('1/5', '4:1/2', 3, '34277273773/549755813888'),

    ('1/5', '4:1', 3, '0'),

    ('1/5', '4:3/4', 3, '0'),

    ('1/5', '5:1/2', 3, '13835277297/137438953472'),

    ('1/5', '5:1', 3, '0'),

    ('1/5', '5:3/4', 3, '0'),

    ('1/5', '6:1/2', 3, '126513610353/1099511627776'),

    ('1/5', '6:1', 3, '0'),

    ('1/5', '6:3/4', 3, '0'),

    ('1/5', '7:1/2', 3, '118782581763/1099511627776'),

    ('1/5', '7:1', 3, '0'),

    ('1/5', '7:3/4', 3, '0'),

    ('1/5', '8:1/2', 3, '62235840169/549755813888'),

    ('1/5', '8:1', 3, '0'),

    ('1/5', '8:3/4', 3, '0'),

    ('1/5', '9:1/2', 3, '122045988441/1099511627776'),

    ('1/5', '9:1', 3, '0'),

    ('1/5', '9:3/4', 3, '0'),

    ('1/5', '10:1/2', 3, '76427945867/549755813888'),

    ('1/5', '10:1', 3, '0'),

    ('1/5', '10:3/4', 3, '0'),

    ('1/5', 'continuation', 3, '0'),

    ('2/5', '0:1/2', 3, '0'),

    ('2/5', '0:1', 3, '0'),

    ('2/5', '0:3/4', 3, '0'),

    ('2/5', '1:1/2', 3, '0'),

    ('2/5', '1:1', 3, '0'),

    ('2/5', '1:3/4', 3, '0'),

    ('2/5', '2:1/2', 3, '0'),

    ('2/5', '2:1', 3, '0'),

    ('2/5', '2:3/4', 3, '0'),

    ('2/5', '3:1/2', 3, '2823021111/549755813888'),

    ('2/5', '3:1', 3, '0'),

    ('2/5', '3:3/4', 3, '0'),

    ('2/5', '4:1/2', 3, '28575136577/1099511627776'),

    ('2/5', '4:1', 3, '0'),

    ('2/5', '4:3/4', 3, '0'),

    ('2/5', '5:1/2', 3, '7156318937/137438953472'),

    ('2/5', '5:1', 3, '0'),

    ('2/5', '5:3/4', 3, '0'),

    ('2/5', '6:1/2', 3, '66955415099/1099511627776'),

    ('2/5', '6:1', 3, '0'),

    ('2/5', '6:3/4', 3, '0'),

    ('2/5', '7:1/2', 3, '14790174921/274877906944'),

    ('2/5', '7:1', 3, '0'),

    ('2/5', '7:3/4', 3, '0'),

    ('2/5', '8:1/2', 3, '64223125837/1099511627776'),

    ('2/5', '8:1', 3, '0'),

    ('2/5', '8:3/4', 3, '0'),

    ('2/5', '9:1/2', 3, '4283350677/68719476736'),

    ('2/5', '9:1', 3, '0'),

    ('2/5', '9:3/4', 3, '0'),

    ('2/5', '10:1/2', 3, '100846277507/1099511627776'),

    ('2/5', '10:1', 3, '0'),

    ('2/5', '10:3/4', 3, '0'),

    ('2/5', 'continuation', 3, '0'),

    ('3/5', '0:1/2', 3, '1115133917/137438953472'),

    ('3/5', '0:1', 3, '0'),

    ('3/5', '0:3/4', 3, '0'),

    ('3/5', '1:1/2', 3, '2303835435/137438953472'),

    ('3/5', '1:1', 3, '0'),

    ('3/5', '1:3/4', 3, '0'),

    ('3/5', '2:1/2', 3, '30371634721/1099511627776'),

    ('3/5', '2:1', 3, '0'),

    ('3/5', '2:3/4', 3, '0'),

    ('3/5', '3:1/2', 3, '12062979643/274877906944'),

    ('3/5', '3:1', 3, '0'),

    ('3/5', '3:3/4', 3, '0'),

    ('3/5', '4:1/2', 3, '68549852497/1099511627776'),

    ('3/5', '4:1', 3, '0'),

    ('3/5', '4:3/4', 3, '0'),

    ('3/5', '5:1/2', 3, '11770217575/137438953472'),

    ('3/5', '5:1', 3, '0'),

    ('3/5', '5:3/4', 3, '0'),

    ('3/5', '6:1/2', 3, '104344655563/1099511627776'),

    ('3/5', '6:1', 3, '0'),

    ('3/5', '6:3/4', 3, '0'),

    ('3/5', '7:1/2', 3, '107449777285/1099511627776'),

    ('3/5', '7:1', 3, '0'),

    ('3/5', '7:3/4', 3, '0'),

    ('3/5', '8:1/2', 3, '15280232377/137438953472'),

    ('3/5', '8:1', 3, '0'),

    ('3/5', '8:3/4', 3, '4096054623/1099511627776'),

    ('3/5', '9:1/2', 3, '142895808391/1099511627776'),

    ('3/5', '9:1', 3, '0'),

    ('3/5', '9:3/4', 3, '5280045165/274877906944'),

    ('3/5', '10:1/2', 3, '176905229417/1099511627776'),

    ('3/5', '10:1', 3, '0'),

    ('3/5', '10:3/4', 3, '25606374357/549755813888'),

    ('3/5', 'continuation', 3, '28231717163/1099511627776'),

    ('4/5', '0:1/2', 3, '56150813733/1099511627776'),

    ('4/5', '0:1', 3, '0'),

    ('4/5', '0:3/4', 3, '13167405037/1099511627776'),

    ('4/5', '1:1/2', 3, '17250531269/274877906944'),

    ('4/5', '1:1', 3, '11141584247/1099511627776'),

    ('4/5', '1:3/4', 3, '24468554059/1099511627776'),

    ('4/5', '2:1/2', 3, '83633031993/1099511627776'),

    ('4/5', '2:1', 3, '0'),

    ('4/5', '2:3/4', 3, '38258670171/1099511627776'),

    ('4/5', '3:1/2', 3, '102904399129/1099511627776'),

    ('4/5', '3:1', 3, '0'),

    ('4/5', '3:3/4', 3, '55323302367/1099511627776'),

    ('4/5', '4:1/2', 3, '61997730249/549755813888'),

    ('4/5', '4:1', 3, '0'),

    ('4/5', '4:3/4', 3, '75845801383/1099511627776'),

    ('4/5', '5:1/2', 3, '149387727383/1099511627776'),

    ('4/5', '5:1', 3, '0'),

    ('4/5', '5:3/4', 3, '102521461777/1099511627776'),

    ('4/5', '6:1/2', 3, '40414865459/274877906944'),

    ('4/5', '6:1', 3, '0'),

    ('4/5', '6:3/4', 3, '55792301369/549755813888'),

    ('4/5', '7:1/2', 3, '85048171001/549755813888'),

    ('4/5', '7:1', 3, '0'),

    ('4/5', '7:3/4', 3, '120720309047/1099511627776'),

    ('4/5', '8:1/2', 3, '96899509561/549755813888'),

    ('4/5', '8:1', 3, '0'),

    ('4/5', '8:3/4', 3, '139958349267/1099511627776'),

    ('4/5', '9:1/2', 3, '55363821319/274877906944'),

    ('4/5', '9:1', 3, '0'),

    ('4/5', '9:3/4', 3, '168218822531/1099511627776'),

    ('4/5', '10:1/2', 3, '253927464255/1099511627776'),

    ('4/5', '10:1', 3, '0'),

    ('4/5', '10:3/4', 3, '200166408073/1099511627776'),

    ('4/5', 'continuation', 3, '141700981545/1099511627776'),

    ('1', '0:1/2', 3, '98433155913/1099511627776'),

    ('1', '0:1', 3, '98433155913/1099511627776'),

    ('1', '0:3/4', 3, '98433155913/1099511627776'),

    ('1', '1:1/2', 3, '56519352947/549755813888'),

    ('1', '1:1', 3, '110061111203/1099511627776'),

    ('1', '1:3/4', 3, '56519352947/549755813888'),

    ('1', '2:1/2', 3, '130040526357/1099511627776'),

    ('1', '2:1', 3, '10469948079/274877906944'),

    ('1', '2:3/4', 3, '130040526361/1099511627776'),

    ('1', '3:1/2', 3, '149426326115/1099511627776'),

    ('1', '3:1', 3, '27210770011/549755813888'),

    ('1', '3:3/4', 3, '37356581555/274877906944'),

    ('1', '4:1/2', 3, '171927663253/1099511627776'),

    ('1', '4:1', 3, '0'),

    ('1', '4:3/4', 3, '21491213639/137438953472'),

    ('1', '5:1/2', 3, '198174423863/1099511627776'),

    ('1', '5:1', 3, '6635323091/1099511627776'),

    ('1', '5:3/4', 3, '99087211937/549755813888'),

    ('1', '6:1/2', 3, '213314594061/1099511627776'),

    ('1', '6:1', 3, '171358285/137438953472'),

    ('1', '6:3/4', 3, '106657297121/549755813888'),

    ('1', '7:1/2', 3, '228511421849/1099511627776'),

    ('1', '7:1', 3, '0'),

    ('1', '7:3/4', 3, '114255715743/549755813888'),

    ('1', '8:1/2', 3, '254702907763/1099511627776'),

    ('1', '8:1', 3, '0'),

    ('1', '8:3/4', 3, '127351454769/549755813888'),

    ('1', '9:1/2', 3, '283596499621/1099511627776'),

    ('1', '9:1', 3, '0'),

    ('1', '9:3/4', 3, '197229071511/1099511627776'),

    ('1', '10:1/2', 3, '160655812303/549755813888'),

    ('1', '10:1', 3, '0'),

    ('1', '10:3/4', 3, '173387748187/1099511627776'),

    ('1', 'continuation', 3, '235054665515/1099511627776'),

]


# Initial adaptive mass grid; no saved numerical bounds are used.
_INITIAL_MASSES = ('0', '1/262144', '1/131072', '1/65536', '1/32768', '3/65536', '1/16384', '3/32768', '1/8192', '5/32768', '3/16384', '1/4096', '5/16384', '3/8192', '7/16384', '1/2048', '5/8192', '3/4096', '7/8192', '1/1024', '9/8192', '5/4096', '3/2048', '7/4096', '1/512', '9/4096', '5/2048', '11/4096', '3/1024', '13/4096', '7/2048', '1/256', '9/2048', '5/1024', '11/2048', '3/512', '13/2048', '7/1024', '1/128', '9/1024', '5/512', '11/1024', '3/256', '13/1024', '7/512', '15/1024', '1/64', '9/512', '5/256', '11/512', '3/128', '13/512', '7/256', '15/512', '1/32', '9/256', '5/128', '11/256', '3/64', '13/256', '7/128', '15/256', '1/16', '9/128', '5/64', '11/128', '3/32', '13/128', '7/64', '15/128', '1/8', '9/64', '5/32', '11/64', '3/16', '13/64', '7/32', '15/64', '1/4', '17/64', '9/32', '5/16', '11/32', '3/8', '13/32', '7/16', '15/32', '1/2', '17/32', '9/16', '19/32', '5/8', '21/32', '11/16', '3/4', '13/16', '7/8', '15/16', '1')



def seeds():
    """Decode exact mesh/multiplier suggestions without reading external files."""
    meshes = []
    for runs in _SEED_MESH_RUNS:
        edges = [Q(0)]
        for count, width in runs:
            width = Q(width)
            if count <= 0 or width <= 0:
                raise ValueError('Invalid embedded mesh run')
            for _ in range(count):
                edges.append(edges[-1] + width)
        if edges[-1] != 1:
            raise ValueError('Embedded mesh does not cover [0,1]')
        meshes.append(edges)
    return {(mass if mass == 'small' else Q(mass), family):
            (meshes[index], Q(alpha))
            for mass, family, index, alpha in _SEED_ROWS}


def proposal(seed):
    """Return a mutable edge list and the nonnegative exact score multiplier."""
    edges, alpha = seed
    if alpha < 0:
        raise ValueError('Negative embedded score multiplier')
    return list(edges), alpha


def source_hashes():
    """Identify this complete accounting implementation for a fresh run."""
    return {'adwords/verify.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


# Complete small/generator checks, endpoint certification, and adaptive mass cover.
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def family_name(name):
    if name.endswith('continuation'):
        return 'continuation'
    a = name.split(':')
    b = '3/4' if a[-1] == 'mid' else str(Q(a[-1]))
    return f'{a[-2]}:{b}'

def refine(edges):
    return sorted(edges + [(a + b) / 2 for a, b in zip(edges, edges[1:])])

class IntervalEndpoint(Endpoint):
    """Reuse only the whole-age kernel enclosure from the original evaluator."""

    def prepare(self, x):
        edges = [Q(i, self.baseN) for i in range(self.baseN + 1)]
        if 0 < x < 1:
            r = Q(round(float((-ball(LAM) * ball(x)).exp()) * 2 ** 30), 2 ** 30)
            if 0 < r < 1 and min((abs(r - a) for a in edges)) > Q(1, 2 ** 32):
                edges = sorted(edges + [r])
        self.edges = edges
        self.N = len(edges) - 1
        self.m = self.N * 13
        self.P = [[ball(q) for q in local_edges(p, edges, 12)] for p in self.polys]
        t = (1 - np.cos(np.pi * np.arange(13) / 12)) / 2
        self.z = np.concatenate([float(l) + t * float(u - l) for l, u in zip(edges, edges[1:])])
        self.inverse = np.linalg.inv(bern(t))

def zero_and_small(candidate, output):
    if Path(candidate).resolve() != (BASE / 'candidate.json').resolve():
        raise ValueError('This verifier currently checks the packaged candidate.json only')
    f, g, h = load_prices(candidate)
    shapes = shape(f, g, h)
    if min(shapes.values()) < 0:
        raise RuntimeError('Negative shape condition')
    initial = sum(add(f[0], g[0])) / 13
    if initial != Q(53, 100):
        raise RuntimeError('Initialization is not exactly53/100')
    records = {}
    small = {}
    seed = seeds()
    started = time.monotonic()
    for name, c0, c1 in accounting_families():
        key = ('small', name) if name.startswith('small:') else (Q(0), family_name(name))
        edges, alpha = proposal(seed[key])
        for attempt in range(5):
            grid = IntervalGrid(len(edges) - 1, edges=edges, score=False)
            modified = (interval_add(c0[0], interval_scale(grid.Hupper, -ball(alpha))), c0[1])
            d0, _, _ = grid.lower_cost(modified, tight=True)
            d1 = None if c1 is None else grid.lower_cost(c1, tight=True)[0]
            cert = grid.certify(d0, d1)
            if Q(cert['bound']) > 0:
                break
            edges = refine(edges)
        else:
            raise RuntimeError(f'Nonpositive exact small/generator bound: {name}')
        cert.update(name=name, N=len(edges) - 1, alpha=str(alpha), edges=list(map(str, edges)), cost0=list(map(str, d0)), cost1=None if d1 is None else list(map(str, d1)))
        if name.startswith('small:'):
            small[name] = cert
        else:
            records[family_name(name)] = cert
    if len(small) != 22 or set(records) != FAMILIES:
        raise RuntimeError('Missing small/generator family')
    polynomial = {'completed': True, 'scope': 'shape_initialization_and_22_small_interval_tests', 'candidate_sha256': digest(candidate), 'initial_potential': str(initial), 'shape_conditions': {k: str(x) for k, x in shapes.items()}, 'small_conditions': small, 'minimum_small_bound': str(min((Q(r['bound']) for r in small.values())))}
    zero = {'completed': True, 'scope': 'zero_mass_generator_interval_tests', 'candidate_sha256': digest(candidate), 'x': '0', 'results': records, 'minimum_bound': str(min((Q(r['bound']) for r in records.values()))), 'seconds_including_small': time.monotonic() - started}
    write_json(output / 'polynomial.json', polynomial)
    write_json(output / 'zero.json', zero)
    print('PASS small+generator', len(small) + len(records), 'maxN', max((r['N'] for r in list(small.values()) + list(records.values()))), flush=True)
    return (polynomial, zero)

def endpoint(candidate, xq):
    xq = Q(xq)
    ctx.prec = 384
    started = time.monotonic()
    all_seeds = seeds()
    available = sorted({a for a, b in all_seeds if a != 'small'})
    source = min(available, key=lambda a: abs(a - xq))
    ep = IntervalEndpoint(candidate, 12)
    ep.prepare(xq)
    M, G, W, errors = ep.values(xq)
    Hpower = analytic_score()
    Hpieces = [[ball(q) for q in local_bern(Hpower, l, u, 26)] for l, u in zip(ep.edges, ep.edges[1:])]
    results = {}
    for family in sorted(FAMILIES):
        x = ball(xq)
        if family == 'continuation':
            c0 = [(w - p + (1 - x) * m) / x for w, p, m in zip(W, ep.P[-1], M[-1])]
            c1 = None
        else:
            ks, bs = family.split(':')
            k = int(ks)
            bq = Q(bs)
            b = ball(bq)
            s = ball(KNOTS[k])
            fi, gi = (1 + 2 * k, 2 + 2 * k)
            f, g = (ep.P[fi], ep.P[gi])
            c1 = [(w - a + (1 - b * x) * mf) / x for w, a, mf in zip(W, f, M[fi])]
            c0 = [((s + b) * w - a - gg + (1 - b * x) * (b * mf + (1 - b) * gf + mg)) / x for w, a, gg, mf, gf, mg in zip(W, f, g, M[fi], G[fi], M[gi])]
            if bq == Q(3, 4):
                c0 = [a - (gf - mf) / 16 for a, gf, mf in zip(c0, G[fi], M[fi])]
        edges, alpha = proposal(all_seeds[source, family])
        p0 = [c0[i:i + 13] for i in range(0, len(c0), 13)]
        p1 = None if c1 is None else [c1[i:i + 13] for i in range(0, len(c1), 13)]
        if alpha:
            p0 = [[a - ball(alpha) * hh for a, hh in zip(elevate(p, 26), hp)] for p, hp in zip(p0, Hpieces)]
        for attempt in range(6):
            d0 = interval_constants(p0, ep.edges, edges)
            d1 = None if p1 is None else interval_constants(p1, ep.edges, edges)
            grid = IntervalGrid(len(edges) - 1, edges=edges, score=False)
            cert = grid.certify(d0, d1)
            if Q(cert['bound']) > 0:
                break
            edges = refine(edges)
            if len(edges) > 24000:
                break
        if Q(cert['bound']) <= 0:
            raise RuntimeError(f"Nonpositive interval bound at x={xq}, family={family}, source={source}: {cert['bound']}")
        cert.update(N=len(edges) - 1, alpha=str(alpha), edges=list(map(str, edges)), cost0=list(map(str, d0)), cost1=None if d1 is None else list(map(str, d1)))
        results[family] = cert
    return {'completed': True, 'scope': 'positive_mass_endpoint_all_ages', 'candidate_sha256': digest(candidate), 'x': str(xq), 'source_mesh_mass': str(source), 'kernel_coarse_cells': ep.N, 'maximum_kernel_error': max((max(e) for e in errors), default=0), 'results': results, 'seconds': time.monotonic() - started}

def quadratic_minimum(left_value, right_value, curvature, l, u):
    q = curvature * (u - l) ** 2 / 2
    if q < 0:
        raise RuntimeError('Negative curvature upper bound')
    t = min(Q(1), max(Q(0), (q + left_value - right_value) / (2 * q))) if q else Q(0)
    return left_value + (right_value - left_value - q) * t + q * t * t

def worker(candidate, x, path):
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--candidate', str(candidate), '--endpoint-worker', str(x), '--endpoint-output', str(path)], check=True)
    return json.loads(Path(path).read_text())

def write_endpoint_certificate(stream, record):
    """Append one complete endpoint record to a compressed JSON Lines stream."""
    stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')

def read_endpoint_certificates(path):
    """Read one endpoint at a time, without loading the full archive into RAM."""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f'Endpoint archive record {number} is not an object')
            yield record

def mass_cover(candidate, output, zero, workers=2):
    if workers < 1:
        raise ValueError('workers must be positive')
    f, g, h = load_prices(candidate)
    started = time.monotonic()
    cache = {Q(0): {name: Q(value['bound']) for name, value in zero['results'].items()}}
    accepted = []
    endpoint_archive = output / 'endpoints.jsonl.gz'
    temporary_archive = output / 'endpoints.jsonl.gz.writing'
    write_json(output / 'mass-cover.json', {'completed': False, 'status': 'running'})
    points = list(map(Q, _INITIAL_MASSES))
    with TemporaryDirectory(prefix='.endpoint-run-', dir=output) as temp, ThreadPoolExecutor(max_workers=workers) as pool, gzip.open(temporary_archive, 'wt', encoding='utf-8') as archive:

        def compute(new):
            new = sorted(set(new) - set(cache))
            futures = {}
            for x in new:
                if x < Q(1, 10000):
                    K = family_bounds(f, g, h, Q(0), x)
                    lower = {n: cache[Q(0)][n] - K[n] * x / 2 for n in FAMILIES}
                    if min(lower.values()) > 0:
                        cache[x] = lower
                        write_endpoint_certificate(archive, {'completed': True, 'scope': 'endpoint_from_generator_curvature', 'x': str(x), 'bounds': {n: str(vv) for n, vv in lower.items()}, 'curvature': {n: str(vv) for n, vv in K.items()}})
                        print('Taylor endpoint', x, float(min(lower.values())), flush=True)
                        continue
                path = Path(temp) / (str(x).replace('/', '_') + '.json')
                futures[pool.submit(worker, candidate, x, path)] = x
            for future in as_completed(futures):
                x = futures[future]
                r = future.result()
                if not r.get('completed') or Q(r['x']) != x or set(r['results']) != FAMILIES or (r['candidate_sha256'] != digest(candidate)):
                    raise RuntimeError('Incorrect endpoint result')
                bounds = {n: Q(a['bound']) for n, a in r['results'].items()}
                if min(bounds.values()) <= 0:
                    raise RuntimeError('Nonpositive endpoint')
                cache[x] = bounds
                write_endpoint_certificate(archive, r)
                print('endpoint', x, 'minimum', float(min(bounds.values())), 'maxN', max((a['N'] for a in r['results'].values())), flush=True)
        compute(points[1:])
        active = list(zip(points, points[1:]))
        roundno = 0
        while active:
            failures = []
            for l, u in active:
                K = family_bounds(f, g, h, l, u)
                if l == 0:
                    normalized = {n: cache[Q(0)][n] - K[n] * u / 2 for n in FAMILIES}
                    minima = {n: Q(0) for n in FAMILIES}
                    method = 'generator derivative and raw-gap curvature'
                else:
                    minima = {n: quadratic_minimum(l * cache[l][n], u * cache[u][n], K[n], l, u) for n in FAMILIES}
                    normalized = {n: minima[n] / u for n in FAMILIES}
                    method = 'raw-gap endpoint chord and curvature'
                if min(normalized.values()) < 0:
                    failures.append((l, u))
                    continue
                accepted.append({'l': str(l), 'u': str(u), 'method': method, 'curvature': {n: str(a) for n, a in K.items()}, 'quadratic_minimum': {n: str(a) for n, a in minima.items()}, 'normalized_bounds': {n: str(a) for n, a in normalized.items()}, 'normalized_lower_bound': str(min(normalized.values()))})
            print('cover round', roundno, 'accepted', len(accepted), 'splitting', len(failures), flush=True)
            if not failures:
                break
            if roundno >= 40 or len(cache) + len(failures) > 10000:
                raise RuntimeError('Mass subdivision limit exceeded')
            compute([(l + u) / 2 for l, u in failures])
            active = [a for l, u in failures for a in [(l, (l + u) / 2), ((l + u) / 2, u)]]
            roundno += 1
    accepted.sort(key=lambda a: Q(a['l']))
    if not accepted or Q(accepted[0]['l']) != 0 or Q(accepted[-1]['u']) != 1 or any((Q(a['l']) >= Q(a['u']) for a in accepted)) or any((Q(a['u']) != Q(b['l']) for a, b in zip(accepted, accepted[1:]))):
        raise RuntimeError('Mass cover has a gap or overlap')
    temporary_archive.replace(endpoint_archive)
    result = {'completed': True, 'scope': 'all_request_masses_interval_states', 'candidate_sha256': digest(candidate), 'families': 34, 'endpoint_count': len(cache), 'interval_count': len(accepted), 'conditions_checked': 34 * len(accepted), 'minimum_normalized_bound': str(min((Q(a['normalized_lower_bound']) for a in accepted))), 'seconds': time.monotonic() - started, 'intervals': accepted, 'endpoints': [str(x) for x in sorted(cache)]}
    write_json(output / 'mass-cover.json', result)
    write_json(output / 'endpoint-summary.json', [{'x': str(x), 'bounds': {n: str(a) for n, a in cache[x].items()}} for x in sorted(cache)])
    return result

def verify_prices(candidate, output, workers=2):
    candidate = Path(candidate).resolve()
    output = Path(output).resolve()
    if candidate != (BASE / 'candidate.json').resolve():
        raise ValueError('This verifier checks the packaged candidate.json only')
    if output.exists() and any(output.iterdir()):
        raise ValueError('price verification requires a new, empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    report = {'completed': False, 'scope': 'all_interval_price_conditions', 'schema_version': 3, 'endpoint_storage': 'endpoints.jsonl.gz', 'status': 'running', 'candidate_sha256': digest(candidate), 'source_sha256': source_hashes()}
    write_json(output / 'report.json', report)
    try:
        polynomial, zero = zero_and_small(candidate, output)
        mass = mass_cover(candidate, output, zero, workers)
        report.update(completed=True, status='passed', initial_potential=polynomial['initial_potential'], small_conditions=22, generator_conditions=34, endpoint_count=mass['endpoint_count'], interval_count=mass['interval_count'], minimum_mass_normalized_bound=mass['minimum_normalized_bound'])
        write_json(output / 'report.json', report)
        return report
    except Exception as error:
        (output / 'endpoints.jsonl.gz.writing').unlink(missing_ok=True)
        report.update(status='failed', error=str(error))
        write_json(output / 'report.json', report)
        raise


# Independent exact audit of saved costs, dual witnesses, and complete covers.
# Archived hashes identify the original enclosure generators. Their source files
# have been consolidated; recognizing these fingerprints is not re-execution.
import importlib.util

ARCHIVED_PRICE_SOURCE_SHA256 = {'interval/verify_prices.py': 'a150a36da3fd7758d371bf69648ae9c6dfae790a9834d8188fbc85e2f0a04238', 'simplification/interval_rigorous.py': '9e0cd85d8e30d10117802216ea40d32b3e3103a2ee7dcfadd6b78a08d96ddc2e', 'simplification/interval_bounds.py': 'a63edd149dbbdd5c03644fe035ac0df7b3dfee253e111c35b3b662fbd22c4cec', 'simplification/interval_endpoint_rigorous.py': '4ce5a6a304459856af4ce01217df821ccf9522a30456fe68476dad60b67a8482', 'price_endpoint.py': '3aaa49fb011a290eb79d2f668c5be2fbd4516d7e6c6b5ada951cd486a6ae9c7f', 'rigorous_kernel.py': '9d3ff89d833ba225e0f49d2531742195248a89edca6211575823b72d64c25910', 'curvature.py': '6b9722cc9e121027ea780e0c1566cda518aa24cdd589d70b69298a9a37ed0011', 'verify_polynomial.py': '8c4c52f65f9f11556d8727601808f996ca779c5e694ee80cf842304af0492dd4'}

def check_price_generation_sources(report):
    """Distinguish current generation from the paper's archived generation."""
    recorded = report.get('source_sha256', {})
    current = {'adwords/verify.py': digest(Path(__file__))}
    if recorded == current:
        return {'mode': 'current_generation_sources', 'generation_sources_rechecked': True}
    if recorded == ARCHIVED_PRICE_SOURCE_SHA256:
        return {'mode': 'recognized_archived_generation_sources',
                'generation_sources_rechecked': False,
                'note': 'Original source fingerprints retained; this audit does not rerun numerical enclosures.'}
    raise ValueError('Unrecognized price-generation source hashes; regenerate the certificate with this verifier.')

def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as source:return json.load(source)
    return json.loads(path.read_text())

def require(condition,message):
    if not condition:raise ValueError(message)

def witness(record):
    """Reconstruct the exact normalized cumulative-tail matrix implicitly."""
    edges=list(map(Q,record['edges']));N=record['N']
    require(len(edges)==N+1 and edges[0]==0 and edges[-1]==1 and all(a<b for a,b in zip(edges,edges[1:])), 'Invalid state partition')
    require(Q(record['alpha'])>=0,'Negative score multiplier')
    scalar=record['cost1'] is None;blocks=1 if scalar else 2
    costs=[list(map(Q,record['cost0']))]
    if not scalar:costs.append(list(map(Q,record['cost1'])))
    require(all(len(c)==N for c in costs),'Incorrect interval cost count')
    rows=2*blocks*(N-1)
    require(record['rows']==rows and record['variables']==blocks*N,'Incorrect tail matrix dimensions')
    u={}
    for index,weight in record['dual_nonzero']:
        require(isinstance(index,int) and 0<=index<rows and index not in u,'Invalid dual row')
        weight=Q(weight);require(weight>=0,'Negative tail multiplier');u[index]=weight
    t=Q(record['dual_t']);residual=[]
    for c in costs:residual.extend([c[i]-c[i+1] for i in range(N-1)]+[c[-1]])
    row=0
    for b in range(blocks):
        residual[(b+1)*N-1]-=t
        for i in range(N-1):
            weight=u.get(row,Q(0));row+=1
            residual[b*N+i]+=weight;residual[b*N+i+1]-=weight
    for kind in range(blocks):
        for i in range(1,N):
            weight=u.get(row,Q(0));row+=1;ratio=edges[i]/edges[i+1]
            residual[i]+=ratio*weight;residual[i-1]-=weight
            if kind:residual[N+i]+=2*ratio*weight;residual[N+i-1]-=2*weight
    require(row==rows,'Tail matrix reconstruction mismatch')
    minimum=None
    for b in range(blocks):
        acc=Q(0)
        for r in reversed(residual[b*N:(b+1)*N]):
            acc+=r;minimum=acc if minimum is None else min(minimum,acc)
    bound=t+minimum
    require(minimum==Q(record['residual_correction']) and bound==Q(record['bound']),'Exact dual residual does not match report')
    require(bound>0,'Nonpositive endpoint witness')
    return bound

def minimum_quadratic(left_value,right_value,B,l,u):
    q=B*(u-l)**2/2
    require(q>=0,'Negative curvature bound')
    t=min(Q(1),max(Q(0),(q+left_value-right_value)/(2*q))) if q else Q(0)
    return left_value+(right_value-left_value-q)*t+q*t*t

def audit_prices(output):
    ctx.prec = 384
    output=Path(output);candidate=BASE/'candidate.json';candidate_hash=digest(candidate)
    report=read(output/'report.json')
    require(report.get('completed') is True and report.get('status')=='passed','Price run is incomplete')
    require(report['candidate_sha256']==candidate_hash,'Candidate changed after price run')
    provenance = check_price_generation_sources(report)
    f,g,h=load_prices(candidate);shapes=shape(f,g,h)
    initial=sum(add(f[0],g[0]))/13
    require(initial==Q(53,100) and min(shapes.values())>=0,'Initialization or shape failure')
    polynomial=read(output/'polynomial.json');zero=read(output/'zero.json')
    require(polynomial['completed'] and zero['completed'],'Missing polynomial/generator certificate')
    require(polynomial['candidate_sha256']==candidate_hash==zero['candidate_sha256'],'Component candidate mismatch')
    require({k:Q(a) for k,a in polynomial['shape_conditions'].items()}==shapes,'Recorded shape conditions differ')
    expected_small={f'small:{j}:{e}' for j in range(10) for e in [j,j+1]}|{'small:continuation_start','small:continuation_slope'}
    require(set(polynomial['small_conditions'])==expected_small and set(zero['results'])==FAMILIES,'Missing accounting family')
    small={k:witness(r) for k,r in polynomial['small_conditions'].items()}
    cache={Q(0):{k:witness(r) for k,r in zero['results'].items()}}
    max_intervals=max(r['N'] for r in list(polynomial['small_conditions'].values())+list(zero['results'].values()))
    witness_count=56
    endpoint_summary=read(output/'endpoint-summary.json')
    expected_masses={Q(r['x']) for r in endpoint_summary}
    require(len(expected_masses)==len(endpoint_summary),'Duplicate endpoint summary')
    require(0 in expected_masses and all(0<=x<=1 for x in expected_masses),'Invalid endpoint mass domain')
    for data in read_endpoint_certificates(output/'endpoints.jsonl.gz'):
        x=Q(data['x'])
        require(0<x<=1 and x in expected_masses,'Unexpected endpoint proof')
        require(x not in cache,'Duplicate endpoint proof')
        require(data.get('completed') is True,'Incomplete endpoint proof')
        if data.get('scope')=='positive_mass_endpoint_all_ages':
            require(data['candidate_sha256']==candidate_hash and set(data['results'])==FAMILIES,'Endpoint candidate/family mismatch')
            cache[x]={k:witness(r) for k,r in data['results'].items()};witness_count+=34
            max_intervals=max(max_intervals,max(r['N'] for r in data['results'].values()))
        elif data.get('scope')=='endpoint_from_generator_curvature':
            K=family_bounds(f,g,h,Q(0),x)
            require(set(data['curvature'])==FAMILIES and set(data['bounds'])==FAMILIES,'Missing Taylor endpoint family')
            cache[x]={}
            for k in FAMILIES:
                B=Q(data['curvature'][k]);require(B>=K[k],'Taylor curvature is too small')
                b=cache[Q(0)][k]-B*x/2
                require(b==Q(data['bounds'][k]) and b>0,'Invalid Taylor endpoint bound')
                cache[x][k]=b
        else:
            raise ValueError('Unknown endpoint proof scope')
    require(set(cache)==expected_masses,'Missing endpoint proof')
    for row in endpoint_summary:
        x=Q(row['x'])
        require(set(row['bounds'])==FAMILIES and {k:Q(a) for k,a in row['bounds'].items()}==cache[x],'Endpoint summary differs from witnesses')
    mass=read(output/'mass-cover.json')
    require(mass.get('completed') is True and mass['candidate_sha256']==candidate_hash,'Mass cover incomplete or stale')
    cover=mass['intervals'];require(bool(cover),'Empty mass cover')
    require(Q(cover[0]['l'])==0 and Q(cover[-1]['u'])==1,'Mass cover misses endpoints')
    require(all(Q(a['u'])==Q(b['l']) for a,b in zip(cover,cover[1:])),'Mass cover gap or overlap')
    minimum=None
    for interval in cover:
        l,u=Q(interval['l']),Q(interval['u']);require(0<=l<u<=1 and l in cache and u in cache,'Invalid mass interval')
        K=family_bounds(f,g,h,l,u);bounds={}
        require(set(interval['curvature'])==FAMILIES and set(interval['normalized_bounds'])==FAMILIES,'Missing cover family')
        for k in FAMILIES:
            B=Q(interval['curvature'][k]);require(B>=K[k],'Mass curvature is too small')
            qmin=Q(0) if l==0 else minimum_quadratic(l*cache[l][k],u*cache[u][k],B,l,u)
            bound=cache[Q(0)][k]-B*u/2 if l==0 else qmin/u
            require(qmin==Q(interval['quadratic_minimum'][k]) and bound==Q(interval['normalized_bounds'][k]) and bound>=0,'Invalid mass interpolation bound')
            bounds[k]=bound
        b=min(bounds.values());require(b==Q(interval['normalized_lower_bound']),'Incorrect interval minimum')
        minimum=b if minimum is None else min(minimum,b)
    require(minimum==Q(mass['minimum_normalized_bound']) and len(cover)==mass['interval_count'],'Incorrect mass-cover summary')
    require(set(map(Q,mass['endpoints']))==set(cache) and len(cache)==mass['endpoint_count'],'Incorrect endpoint list')
    require(mass['conditions_checked']==34*len(cover),'Incorrect mass condition count')
    return {'generation_provenance': provenance, 'candidate_sha256':candidate_hash,'initial_potential':str(initial),'shape_conditions':len(shapes),'small_conditions':22,'generator_conditions':34,'endpoint_count':len(cache),'interval_count':len(cover),'large_request_conditions':34*len(cover),'exact_lp_witnesses_checked':witness_count,'maximum_state_intervals':max_intervals,'minimum_small_bound':str(min(small.values())),'minimum_mass_bound':str(minimum)}

def audit_all(output):
    started=time.monotonic();output=Path(output)
    price=audit_prices(output/'prices')
    spec=importlib.util.spec_from_file_location('interval_score_verifier',HERE/'verify_score.py')
    score_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(score_module)
    score=score_module.audit(output/'score')
    return {'completed':True,'passed':True,'scope':'exact_witness_and_cover_audit','kernel_enclosures_recomputed':False,'score_integration_enclosures_recomputed':False,'prices':price,'score':score,'audit_verifier_sha256':digest(Path(__file__)),'seconds':time.monotonic()-started}


def main():
    """Run fresh components in isolated processes, or audit saved results."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        help='new results directory, or existing directory to audit')
    parser.add_argument('--workers', type=int, default=2,
                        help='worker processes per numerical component (default: 2)')
    parser.add_argument('--audit-existing', action='store_true',
                        help='read-only exact audit; do not repeat numerical integrations')
    parser.add_argument('--candidate', type=Path, default=BASE / 'candidate.json',
                        help=argparse.SUPPRESS)
    parser.add_argument('--price-component', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--endpoint-worker', help=argparse.SUPPRESS)
    parser.add_argument('--endpoint-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.candidate.resolve() != (BASE / 'candidate.json').resolve():
        parser.error('this verifier checks the packaged candidate.json only')
    if args.endpoint_worker is not None:
        if args.endpoint_output is None:
            parser.error('--endpoint-output is required for an endpoint worker')
        write_json(args.endpoint_output, endpoint(args.candidate, Q(args.endpoint_worker)))
        return
    if args.output is None:
        parser.error('--output is required')
    if args.price_component:
        verify_prices(args.candidate, args.output, args.workers)
        return

    output = args.output.resolve()
    started = time.monotonic()
    report = {
        'completed': False, 'schema_version': 2,
        'scope': 'full_adwords_interval_certificate', 'status': 'running',
        'execution_mode': ('audit_completed_component_runs' if args.audit_existing
                           else 'fresh_component_runs_and_exact_audit'),
        'enclosures_recomputed_in_this_invocation': not args.audit_existing,
    }
    if not args.audit_existing:
        # Never replace the paper's bundled certificate accidentally.
        if output.exists() and any(output.iterdir()):
            parser.error('a fresh run needs an empty output directory; use --audit-existing for saved results')
        write_json(output / 'report.json', report)
    try:
        if not args.audit_existing:
            # Arb precision is process-global. Each component and endpoint
            # therefore runs in an isolated Python process.
            commands = [
                [sys.executable, str(Path(__file__).resolve()), '--price-component',
                 '--output', str(output / 'prices'), '--workers', str(args.workers)],
                [sys.executable, str(HERE / 'verify_score.py'),
                 '--output', str(output / 'score'), '--workers', str(args.workers)],
            ]
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(subprocess.run, command, check=True) for command in commands]
                for job in jobs:
                    job.result()
        checked = audit_all(output)
        price, score = checked['prices'], checked['score']
        report.update(
            completed=True, status='passed', ratio='53/100',
            candidate_sha256=price['candidate_sha256'],
            initialization_and_shape_conditions=True,
            small_request_conditions=price['small_conditions'],
            generator_conditions=price['generator_conditions'],
            mass_endpoints=price['endpoint_count'], mass_intervals=price['interval_count'],
            large_request_interval_conditions=price['large_request_conditions'],
            maximum_state_intervals=price['maximum_state_intervals'],
            exact_lp_witnesses_checked=price['exact_lp_witnesses_checked'],
            score_rectangles=score['rectangles'],
            minimum_small_normalized_bound=price['minimum_small_bound'],
            minimum_large_normalized_bound=price['minimum_mass_bound'],
            minimum_score_normalized_defect=score['minimum_normalized_defect_exact'],
            independent_exact_audit_passed=True,
            price_generation_provenance=price['generation_provenance'],
            score_generation_provenance=score.get('source_provenance', score.get('generation_provenance')),
            component_reports=['prices/report.json', 'score/report.json'],
            audit_verifier_sha256=digest(Path(__file__)),
            score_verifier_sha256=digest(HERE / 'verify_score.py'),
            seconds=time.monotonic() - started,
        )
        if not args.audit_existing:
            write_json(output / 'audit.json', checked)
            write_json(output / 'report.json', report)
        print('PASS: complete interval-mass AdWords certificate, ratio 53/100', flush=True)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        report.update(status='failed', error=str(error), seconds=time.monotonic() - started)
        if not args.audit_existing:
            write_json(output / 'report.json', report)
        raise


if __name__ == '__main__':
    main()
