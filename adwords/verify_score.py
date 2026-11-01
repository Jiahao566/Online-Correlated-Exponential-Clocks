#!/usr/bin/env python3
"""Certify the paper's analytic signed-score invariant on all ages and masses.

This self-contained entry point includes the analytic kernel, second-order
interval derivatives, adaptive rectangle cover, exact constant transfer, and
read-only saved-certificate audit. No project-local helper files are imported.

Fresh verification requires python-flint >= 0.9.0; the audit uses only the
Python standard library. See ../README.md for the mathematical checks and the
separate provenance of the included historical numerical certificate.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

HERE = Path(__file__).resolve().parent
LAM = Q(9, 2)
C0 = Q(-263341615608130387, 250000000000000000)

# Fingerprints recorded when the paper's included numerical enclosures were
# generated, before their implementation was consolidated into this file.
# Recognizing these fingerprints does not claim those retired sources were
# rehashed or the saved numerical integrations were rerun by a later audit.
ARCHIVED_SOURCE_SHA256 = {
    'interval/verify_score.py': '59581ed8774883cd5c25a77ca3b46024d0e1b001abb7c9a83d17601fb1f355be',
    'moment_kernel.py': '176c591e2bd6545fc25656de7b1486be07926fbd289a8f95bfe956e96069980f',
    'verify_moment.py': 'ff796fbe6cbc12c593146807af3460e62ed817443d11f0db95f82d7d2dc5764a',
    'check_cover.py': '346d96b677c90053aea177bc410ad51b86c6c8a966bcebd835d6d49da13ab040',
}
_ARITHMETIC_READY = False


def require(condition, message):
    """Reject an invalid certificate even when Python runs with -O."""
    if not condition:
        raise ValueError(message)


def source_hashes():
    return {'adwords/verify_score.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def source_provenance(recorded):
    """Distinguish checked current code from recognized historical sources."""
    if recorded == source_hashes():
        return dict(source_provenance='current_consolidated_generation_source',
                    generation_sources_rechecked=True)
    require(recorded == ARCHIVED_SOURCE_SHA256,
            'Unknown or changed score-generation source fingerprints')
    return dict(source_provenance='recognized_archived_generation_sources',
                generation_sources_rechecked=False)


def _initialize_arithmetic():
    """Load Arb only for fresh enclosures, always at the original 80 bits."""
    global _ARITHMETIC_READY, flint, arb, acb, ctx
    global L, C0_BALL, MU, H0, POLYNOMIAL_TRANSFER
    if _ARITHMETIC_READY:
        return
    import flint
    from flint import arb, acb, ctx
    ctx.prec = 80
    L = arb(9)/2
    C0_BALL = arb(-263341615608130387)/250000000000000000
    MU = (-L).exp()
    H0 = C0_BALL + L*acb(-L).hypgeom([1, 1], [2, 2])
    # Preserve the original conservative smooth-gap deduction. Its former
    # interpretation as a polynomial transfer is not used in this proof.
    POLYNOMIAL_TRANSFER = arb(1)/10**9
    _ARITHMETIC_READY = True


# Analytic kernel and pure second derivatives.

def ev(v):return v.hypgeom_1f1(1,2) if v.contains(0) or abs(v).upper()<arb(1)/2 else v.expm1()/v

def ed(v,e):return v.hypgeom_1f1(2,3)/2 if v.contains(0) or abs(v).upper()<arb(1)/2 else (v.exp()-e)/v

def lv(v):
 if not v.real < 1:return acb(arb.nan())
 return v.hypgeom_2f1(1,1,2) if v.contains(0) or abs(v).upper()<arb(1)/2 else -(1-v).log(analytic=True)/v

def ld(v,l):
 if not v.real < 1:return acb(arb.nan())
 return v.hypgeom_2f1(2,2,3)/2 if v.contains(0) or abs(v).upper()<arb(1)/2 else (1/(1-v)-l)/v

class Scalar:
 def __init__(self,v,d=0):self.v=acb(v)
 def __add__(self,b):return Scalar(self.v+(b.v if isinstance(b,Scalar) else b))
 __radd__=__add__
 def __neg__(self):return Scalar(-self.v)
 def __sub__(self,b):return self+(-b)
 def __rsub__(self,b):return -self+b
 def __mul__(self,b):return Scalar(self.v*(b.v if isinstance(b,Scalar) else b))
 __rmul__=__mul__
 def __truediv__(self,b):return Scalar(self.v/(b.v if isinstance(b,Scalar) else b))
 def __rtruediv__(self,b):return Scalar(b/self.v)
 def exp(self):return Scalar(self.v.exp())
 def exprel(self):return Scalar(ev(self.v))
 def logrel(self):return Scalar(lv(self.v))
 def H(self):return Scalar(C0_BALL+L*self.v*(-L*self.v).hypgeom([1,1],[2,2]))

class Jet2:
 def __init__(self,v,d=0,d2=0):self.v=acb(v);self.d=acb(d);self.d2=acb(d2)
 def __add__(self,b):
  if not isinstance(b,Jet2):b=Jet2(b)
  return Jet2(self.v+b.v,self.d+b.d,self.d2+b.d2)
 __radd__=__add__
 def __neg__(self):return Jet2(-self.v,-self.d,-self.d2)
 def __sub__(self,b):return self+-b if isinstance(b,Jet2) else self+Jet2(-b)
 def __rsub__(self,b):return -self+b
 def __mul__(self,b):
  if not isinstance(b,Jet2):b=Jet2(b)
  return Jet2(self.v*b.v,self.v*b.d+self.d*b.v,self.v*b.d2+2*self.d*b.d+self.d2*b.v)
 __rmul__=__mul__
 def inv(self):
  q=1/self.v;return Jet2(q,-self.d*q*q,2*self.d*self.d*q*q*q-self.d2*q*q)
 def __truediv__(self,b):
  if not isinstance(b,Jet2):b=Jet2(b)
  return self*b.inv()
 def __rtruediv__(self,b):return Jet2(b)*self.inv()
 def apply(self,v,df,ddf):return Jet2(v,df*self.d,df*self.d2+ddf*self.d*self.d)
 def exp(self):
  e=self.v.exp();return self.apply(e,e,e)
 def exprel(self):
  v=self.v;e=ev(v);d=ed(v,e);d2=v.hypgeom_1f1(3,4)/3 if v.contains(0) or abs(v).upper()<arb(1)/2 else (v.exp()-2*d)/v
  return self.apply(e,d,d2)
 def logrel(self):
  v=self.v;e=lv(v);d=ld(v,e);d2=2*v.hypgeom_2f1(3,3,4)/3 if v.contains(0) or abs(v).upper()<arb(1)/2 else ((1/(1-v))*(1/(1-v))-2*d)/v
  return self.apply(e,d,d2)
 def log(self):
  q=1/self.v;return self.apply(self.v.log(analytic=True),q,-q*q)
 def real_remainder(self):
  v=self.v.real;lo=v.lower();hi=v.upper()
  al=1-(1+lo)*(-lo).exp();ah=1-(1+hi)*(-hi).exp()
  lower=arb(0) if v.contains(0) else max(arb(0),min(al.lower(),ah.lower()))
  upper=max(al.upper(),ah.upper())
  av=acb(arb((lower+upper)/2,(upper-lower)/2))
  return self.apply(av,self.v*(-self.v).exp(),(1-self.v)*(-self.v).exp())
 def H(self):
  v=self.v;h=C0_BALL+L*v*(-L*v).hypgeom([1,1],[2,2]);return self.apply(h,L*ev(-L*v),-L*L*ed(-L*v,ev(-L*v)))

def gap_expectation(w):return w+H0-1+((-L*w).exp()-MU)/L

def norm(x,u,domain,der=None,tol=1e-10,deep=False,real_N=0):
 """Return defect (der=None), or pure second derivative ('x'/'u').

 real_N>0 uses rigorous real interval range quadrature on that many panels;
 otherwise use Arb's certified complex integration.
 """
 _initialize_arithmetic()
 if real_N and der is None:raise ValueError('real interval quadrature is used only for derivative bounds')
 D=Jet2 if der else Scalar
 x=D(x,int(der=='x'));u=D(u,int(der=='u'));r=(-L*x).exp()
 z=r*u if domain=='old' else (-L*x*u).exp()
 c=r*(1+L*x*(-L*(1-x)).exprel())
 def integr(v,analytic,young):
  if domain=='old':t=D(v);length=D(1)
  elif young:t=(1-u)*v;length=1-u
  else:t=1-u+u*v;length=u
  y=z*(-L*x*t).exp()
  if young:
   k=u+t
   if real_N and x.v.real>arb(1)/20:
    F=r+(L*x*k).real_remainder();ell=-F.log()/x
   else:
    W=L*y*(k+(1-k)*(-L*x*(1-k)).exprel());ell=W*(x*W).logrel()
  else:ell=L*y*(L*x*y).logrel()
  loss=ell*(-(1-x)*ell).exprel()
  val=length*L*((-L*x*t).exp()*loss*gap_expectation((-L*x*(1-t)).exp())-c*(1-(-L*y).exp()))
  return val.d2 if der else val.v
 if real_N:
  v=acb(0)
  for k in range(real_N):
   vv=acb(arb((arb(k)+arb(1)/2)/real_N,arb(1)/(2*real_N)))
   v+=integr(vv,False,False)/real_N
   if domain=='young':v+=integr(vv,False,True)/real_N
 else:
  v=acb.integral(lambda v,a:integr(v,a,False),0,1,abs_tol=tol,rel_tol=tol,eval_limit=(300 if der and not deep else 10000),depth_limit=(8 if der and not deep else 18),deg_limit=(32 if der and not deep else 96))
  if domain=='young':v+=acb.integral(lambda v,a:integr(v,a,True),0,1,abs_tol=tol,rel_tol=tol,eval_limit=(300 if der and not deep else 10000),depth_limit=(8 if der and not deep else 18),deg_limit=(32 if der and not deep else 96))
 last=c*(arb(1)/100+arb(13)/10*x)*z.H()
 return (v-(last.d2 if der else last.v)).real


# Exact cover, constant transfer, and saved-certificate audit.

def check_cover(data):
    rects = data['leaves']
    left, right = data.get('x_interval', [0, 1])
    edges = sorted({u for row in rects for u in row[2:4]})
    index = {value: i for i, value in enumerate(edges)}
    events = collections.defaultdict(list)
    for xl, xu, ul, uu, lower in rects:
        require(left <= xl < xu <= right and 0 <= ul < uu <= 1, 'Invalid score certificate: left <= xl < xu <= right and 0 <= ul < uu <= 1')
        require(lower > 0, 'Invalid score certificate: lower > 0')
        events[xl].append((index[ul], index[uu], 1))
        events[xu].append((index[ul], index[uu], -1))
    xs = sorted(events)
    require(xs[0] == left and xs[-1] == right, 'Invalid score certificate: xs[0] == left and xs[-1] == right')
    require(edges[0] == 0 and edges[-1] == 1, 'Invalid score certificate: edges[0] == 0 and edges[-1] == 1')
    coverage = [0] * (len(edges) - 1)
    for x in xs:
        for lo, hi, delta in events[x]:
            for j in range(lo, hi):
                coverage[j] += delta
        require(all((count == (0 if x == right else 1) for count in coverage)), 'Invalid score certificate: all((count == (0 if x == right else 1) for count in coverage))')
    return len(rects)

def ein_sum(n):
    return sum(((-1) ** (k + 1) * LAM ** k / (k * math.factorial(k)) for k in range(1, n + 1)), Q(0))

def constant_transfer():
    """Exact alternating-series enclosure and normalized-defect allowance.

    The k-th absolute term has ratio lambda*k/(k+1)^2 <= 1 for k>=2.
    Thus S40 <= Ein(lambda) <= S41.  If delta = H-Htilde, the
    regular normalized-gap formula changes by
      delta * [integral lambda*exp(-lambda*x*t)*R_x(y) dt
               - C_x*(1/100+13*x/10)].
    Here 0<=R_x<=lambda, 0<=C_x<=1, and 0<=x<=1.  The resulting
    absolute difference is <= (lambda^2+131/100)*abs(delta), including
    the continuous extensions at x=0 and x=1.
    """
    lower = Q(103, 100) - C0 - ein_sum(41)
    upper = Q(103, 100) - C0 - ein_sum(40)
    require(-Q(5, 10 ** 19) < lower <= upper < 0, 'Invalid score certificate: -Q(5, 10 ** 19) < lower <= upper < 0')
    allowance = (LAM ** 2 + Q(131, 100)) * max(abs(lower), abs(upper))
    require(allowance < Q(11, 10 ** 18), 'Invalid score certificate: allowance < Q(11, 10 ** 18)')
    return (lower, upper, allowance)

def rounded_down(value):
    """Export an exact rational lower bound as a downward-rounded float."""
    result = float(value)
    if Q(result) > value:
        result = math.nextafter(result, -math.inf)
    require(Q(result) <= value, 'Invalid score certificate: Q(result) <= value')
    return result

def audit(output):
    """Check coverage, exact transfer, minima, and generation provenance.

    This checks the structure of a completed numerical certificate; generating
    that certificate still requires the default full interval-arithmetic run.
    """
    output = Path(output)
    report = json.loads((output / 'report.json').read_text())
    require(report['completed'] is True and report['passed'] is True, "Invalid score certificate: report['completed'] is True and report['passed'] is True")
    require(report['recomputed'] is True, "Invalid score certificate: report['recomputed'] is True")
    dl, du, allowance = constant_transfer()
    require(Q(report['delta_lower_exact']) == dl, "Invalid score certificate: Q(report['delta_lower_exact']) == dl")
    require(Q(report['delta_upper_exact']) == du, "Invalid score certificate: Q(report['delta_upper_exact']) == du")
    require(Q(report['transfer_allowance_exact']) == allowance, "Invalid score certificate: Q(report['transfer_allowance_exact']) == allowance")
    bounds, count = ([], 0)
    for domain in ['old', 'young']:
        source = json.loads((output / 'smooth' / f'moment-{domain}.json').read_text())
        result = json.loads((output / f'score-{domain}.json').read_text())
        require(source['completed'] is True and result['completed'] is True, "Invalid score certificate: source['completed'] is True and result['completed'] is True")
        require(source['x_interval'] == result['x_interval'] == [0, 1], "Invalid score certificate: source['x_interval'] == result['x_interval'] == [0, 1]")
        require(source['polynomial_transfer_subtracted'] == '1e-9', "Invalid score certificate: source['polynomial_transfer_subtracted'] == '1e-9'")
        check_cover(source)
        count += check_cover(result)
        require(len(source['leaves']) == len(result['leaves']), "Invalid score certificate: len(source['leaves']) == len(result['leaves'])")
        for original, transferred in zip(source['leaves'], result['leaves']):
            require(original[:4] == transferred[:4], 'Invalid score certificate: original[:4] == transferred[:4]')
            require(transferred[4] == rounded_down(Q(original[4]) - allowance), 'Invalid score certificate: transferred[4] == rounded_down(Q(original[4]) - allowance)')
        minimum = min((row[4] for row in result['leaves']))
        require(minimum == result['min_lower_bound'], "Invalid score certificate: minimum == result['min_lower_bound']")
        require(Q(minimum) > Q(1, 20000), 'Invalid score certificate: Q(minimum) > Q(1, 20000)')
        bounds.append(Q(minimum))
    require(count == report['rectangles'], "Invalid score certificate: count == report['rectangles']")
    require(min(bounds) == Q(report['minimum_normalized_defect_exact']), "Invalid score certificate: min(bounds) == Q(report['minimum_normalized_defect_exact'])")
    require(min(bounds) == Q(report['minimum_normalized_defect']), "Invalid score certificate: min(bounds) == Q(report['minimum_normalized_defect'])")
    provenance = source_provenance(report['source_sha256'])
    return dict(completed=True, passed=True, rectangles=count, minimum_normalized_defect_exact=str(min(bounds)), transfer_allowance_exact=str(allowance), **provenance)


# Fresh adaptive rectangle verification.

def ball(left, right):
    return arb((arb(left) + arb(right)) / 2, (arb(right) - arb(left)) / 2)

def verify_domain(domain, output, part=None, progress=True):
    """Certify one coordinate square (or one of its four mass bands)."""
    _initialize_arithmetic()
    ctx.prec = 80
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stack = [(i/16, (i+1)/16, j/16, (j+1)/16, 0)
             for i in range(16) for j in range(16)
             if part is None or i//4 == part]
    leaves, cache = [], {}
    tested = 0

    def point(x, u):
        key = (x, u)
        if key not in cache:
            cache[key] = norm(arb(x), arb(u), domain, tol=1e-8)
        return cache[key]

    while stack:
        xl, xu, ul, uu, depth = stack.pop()
        tested += 1
        xm, um = (xl+xu)/2, (ul+uu)/2
        hx, hu = (xu-xl)/2, (uu-ul)/2
        xb, ub = ball(xl, xu), ball(ul, uu)
        xx = norm(xb, ub, domain, 'x', real_N=16)
        uu_derivative = norm(xb, ub, domain, 'u', real_N=16)
        if not xx.is_finite() and max(hx, hu) < .008:
            xx = norm(xb, ub, domain, 'x', real_N=128)
        if not uu_derivative.is_finite() and max(hx, hu) < .008:
            uu_derivative = norm(xb, ub, domain, 'u', real_N=128)
        error_x = abs(xx).upper() * arb(hx)**2 / 2
        error_u = abs(uu_derivative).upper() * arb(hu)**2 / 2
        lower = arb.nan()
        if error_x.is_finite() and error_u.is_finite() and error_x+error_u < arb(1)/2:
            corners = [point(x, u).lower() for x in (xl, xu) for u in (ul, uu)]
            lower = min(corners)-error_x-error_u-POLYNOMIAL_TRANSFER
        if lower > 0:
            # Float storage is rounded down; proof acceptance used Arb above.
            saved_lower = math.nextafter(float(lower.lower()), -math.inf)
            leaves.append([xl, xu, ul, uu, saved_lower])
        else:
            if depth >= 28:
                raise RuntimeError(f'Uncertified {domain} rectangle {(xl,xu,ul,uu)}')
            split_x = ((xx.is_finite() and uu_derivative.is_finite() and error_x > error_u)
                       or (not (xx.is_finite() and uu_derivative.is_finite()) and hx >= hu))
            if split_x:
                stack.extend([(xl,xm,ul,uu,depth+1), (xm,xu,ul,uu,depth+1)])
            else:
                stack.extend([(xl,xu,ul,um,depth+1), (xl,xu,um,uu,depth+1)])
        if progress and tested % 500 == 0:
            print(domain, part, ':', tested, 'tested,', len(leaves), 'accepted', flush=True)

    result = dict(domain=domain, x_interval=[0,1] if part is None else [part/4,(part+1)/4],
                  method='bilinear interpolation with interval pure second derivatives',
                  completed=True, tested=tested, leaves=leaves,
                  min_lower_bound=min(row[4] for row in leaves),
                  polynomial_transfer_subtracted='1e-9', precision_bits=ctx.prec,
                  python_flint=flint.__version__, seconds=time.monotonic()-started)
    check_cover(result)
    suffix = '' if part is None else f'-part{part}'
    path = output/f'moment-{domain}{suffix}.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    if progress:
        print('CERTIFIED', domain, part, ':', len(leaves), 'rectangles; lower bound',
              result['min_lower_bound'], flush=True)
    return result

def _run_smooth_worker(domain, part, output):
    subprocess.run([
        sys.executable, str(Path(__file__).resolve()),
        '--output', str(output), '--smooth-worker', '--domain', domain,
        '--part', str(part),
    ], check=True)


def verify_smooth_covers(output, workers):
    """Compute both covers; parallel workers communicate via temporary shards.

    Each subprocess has its own Arb precision and computes one exact quarter
    of a coordinate square. Only the two combined moment files are retained.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        return [verify_domain(domain, output) for domain in ('old', 'young')]
    with TemporaryDirectory(prefix='.score-parts-', dir=output) as temporary:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(_run_smooth_worker, domain, part, temporary)
                    for domain in ('old', 'young') for part in range(4)]
            for job in jobs:
                job.result()
        results = []
        for domain in ('old', 'young'):
            parts = [json.loads((Path(temporary)/f'moment-{domain}-part{i}.json').read_text())
                     for i in range(4)]
            result = dict(parts[0])
            result.update(
                x_interval=[0, 1],
                leaves=sum((p['leaves'] for p in parts), []),
                tested=sum(p['tested'] for p in parts),
                seconds=sum(p['seconds'] for p in parts),
                min_lower_bound=min(p['min_lower_bound'] for p in parts),
            )
            check_cover(result)
            (output/f'moment-{domain}.json').write_text(json.dumps(result, indent=2)+'\n')
            print('CERTIFIED', domain, ':', len(result['leaves']), 'rectangles; lower bound',
                  result['min_lower_bound'], flush=True)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--smooth-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--domain', choices=['old', 'young'], help=argparse.SUPPRESS)
    parser.add_argument('--part', type=int, choices=range(4), help=argparse.SUPPRESS)
    parser.add_argument('--audit-only', action='store_true', help='audit existing covers and provenance without numerical integrations')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.smooth_worker:
        if args.domain is None or args.part is None or args.audit_only:
            parser.error('internal worker requires --domain and --part')
        verify_domain(args.domain, args.output, args.part)
        return
    if args.domain is not None or args.part is not None:
        parser.error('--domain and --part are internal worker options')
    if args.audit_only:
        print(json.dumps(audit(args.output), indent=2), flush=True)
        return
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'report.json'
    report_path.write_text(json.dumps(dict(completed=False, status='running')) + '\n')
    dl, du, allowance = constant_transfer()
    smooth = args.output / 'smooth'
    verify_smooth_covers(smooth, args.workers)
    results = []
    for domain in ['old', 'young']:
        source = json.loads((smooth / f'moment-{domain}.json').read_text())
        require(source['completed'] is True, "Invalid score certificate: source['completed'] is True")
        require(source['x_interval'] == [0, 1], "Invalid score certificate: source['x_interval'] == [0, 1]")
        require(source['polynomial_transfer_subtracted'] == '1e-9', "Invalid score certificate: source['polynomial_transfer_subtracted'] == '1e-9'")
        check_cover(source)
        leaves = [[*row[:4], rounded_down(Q(row[4]) - allowance)] for row in source['leaves']]
        result = dict(domain=domain, x_interval=[0, 1], completed=True, score='103/100 - Ein(9/2) + Ein(9*z/2)', leaves=leaves, min_lower_bound=min((row[4] for row in leaves)), source=f'smooth/moment-{domain}.json', extra_constant_transfer_subtracted=str(allowance), conservative_smooth_gap_deduction_retained='1e-9')
        check_cover(result)
        require(Q(result['min_lower_bound']) > Q(1, 20000), "Invalid score certificate: Q(result['min_lower_bound']) > Q(1, 20000)")
        (args.output / f'score-{domain}.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
    report = dict(completed=True, passed=True, recomputed=True, score='103/100 - Ein(9/2) + Ein(9*z/2)', alpha='(exp(-9*x/2)-x*exp(-9/2))*(1+x/100+13*x*x/10)', statement='M_x H(z) >= alpha(x) H(z) for every x,z in [0,1]', normalization='(M_x H(z)-alpha(x) H(z))/(x*(1-x))', domains=['old', 'young'], charts={'old': 'z=exp(-9*x/2)*u', 'young': 'z=exp(-9*x*u/2)'}, rectangles=sum((len(r['leaves']) for r in results)), minimum_normalized_defect=min((r['min_lower_bound'] for r in results)), minimum_normalized_defect_exact=str(min((Q(r['min_lower_bound']) for r in results))), minimum_by_domain={r['domain']: dict(rectangles=len(r['leaves']), bound=r['min_lower_bound'], bound_exact=str(Q(r['min_lower_bound']))) for r in results}, delta_lower_exact=str(dl), delta_upper_exact=str(du), delta_enclosure_decimal=[float(dl), float(du)], transfer_allowance_exact=str(allowance), transfer_allowance_decimal=float(allowance), analytic_endpoint_argument='M_0=Id, alpha(0)=1; M_1=0, alpha(1)=0', initial_score_argument='pi H = 3/100 + (1-exp(-9/2))/(9/2) > 0', alpha_nonnegative_argument='exp(-9*x/2) >= exp(-9/2) >= x*exp(-9/2)', files=[f'score-{d}.json' for d in ['old', 'young']], source_sha256=source_hashes(), implementation='self_contained_analytic_score_v1', seconds=time.monotonic() - started)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    audit(args.output)
    print(json.dumps({k: report[k] for k in ['completed', 'passed', 'rectangles', 'minimum_normalized_defect', 'transfer_allowance_decimal', 'seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
