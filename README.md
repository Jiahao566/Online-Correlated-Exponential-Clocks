# Online-Correlated-Exponential-Clocks

Verification code and certificates for the paper's potential inequalities:

- [Matching and Display Ads](#matching-and-display-ads-potential-verification):
  scripts and results in [`display-ads/`](display-ads/).
- [AdWords](#adwords-potential-verification): scripts, candidate prices, and
  certificates in [`adwords/`](adwords/).
- [Load Balancing](#load-balancing-potential-verification): scripts and results
  in [`load-balancing/`](load-balancing/).

## Matching and Display Ads potential verification

This project certifies the potential accounting inequality for every request
mass `0 <= x <= 1` and every measure reachable from the stationary age
distribution by matching and gap updates, with `lambda = 9/2`.

The small- and large-request verifiers share one potential implementation.
Running both together constructs the potential tables once and reuses them.
All numerical bounds use 160-bit Arb interval arithmetic through
`python-flint`.

### Run

From the repository root, enter `display-ads/`:

```bash
cd display-ads
python -m pip install python-flint==0.9.0
python verify.py
```

The remaining commands and file paths in this section are relative to
`display-ads/`. To preserve the supplied results, write a fresh run to another
directory:

```bash
python verify.py --output-dir rerun-results
```

Either part can also be run separately:

```bash
python verify.py --only small --output-dir small-results
python verify.py --only large --output-dir large-results
```

The equivalent standalone entry points are:

```bash
python verify_small.py --output-dir small-results
python verify_large.py --output-dir large-results
```

Use ordinary Python, without `-O` or `PYTHONOPTIMIZE`: verification uses
assertions. Each invocation builds its tables from scratch and recomputes
every interval in the selected range. Saved tables and results are never
trusted as inputs. Only the default combined run verifies the entire
`[0, 1]` range in one invocation.

### Files and results

| File | Purpose |
|---|---|
| `verify.py` | Unified entry point, shared precomputation, timing, and result metadata. |
| `potential.py` | Rigorous finite-sum bounds for the potential, reset expectation, and their coordinate derivatives. |
| `verify_small.py` | Small-request verification on `[0, 1/10]`. |
| `verify_large.py` | Large-request verification on `[1/10, 1]`. |
| `results/small.json` | Complete small-request interval bounds and timing. |
| `results/large.json` | Complete large-request interval bounds and timing. |
| `results/summary.json` | Combined-run status, scope, timings, environment, and source hashes. |

The JSON reports associate results with a run ID and source hashes. Inspect
`complete` and the selected request range when interpreting a report. A
successful small-only or large-only run certifies only that part of the
domain. The supplied reports were generated from the code in this project.
Read `summary.json` first and use only the part files listed there with the
same `run_id`. Reusing an output directory for a single-part run can leave
an older file for the other part; that file does not belong to the new run.

### Mathematical certificate

The desired inequality is `nu R_x >= 0`, where

```text
R_x = 1 - Gamma - M_x(1) + (1-x) M_x(Gamma).
```

The reachable-measure lower tail bound reduces this to finite stationary
averages of increasing minorants. In the coordinate `z = exp(-lambda*a)`,
the stationary distribution is uniform on `[0, 1]`. For each complete age
cell, the code computes a rigorous lower bound. It then takes running
prefix minima in increasing `z` and averages them. This is a lower bound
on the manuscript's `B_N` expression, and hence gives a lower bound on the
integral under every reachable measure, after multiplication by `nu(1)`.

Individual pointwise residuals or derivative values need not be nonnegative.
The certified quantity is the minorant average used by the measure bound.

#### Large requests

For every interval `[left, right]`, request monotonicity gives

```text
nu R_x >= nu C_left,right,
C_left,right = 1 - Gamma - M_left(1) + (1-right) M_right(Gamma).
```

The explicit kernel integrals are bounded by finite endpoint sums. The
resulting lower bound covers every age in each cell. Nonnegative minorant
averages for all 144 request intervals prove the claim on `[1/10, 1]`.
These intervals are `[16/(j+1), 16/j]` for `j = 16, ..., 159`.

#### Small requests

In the first-arrival integral, replace the win probability by its smooth
lower bound. The no-arrival contribution remains unchanged. This produces
a residual lower bound `Rlower_x` with

```text
Rlower_0 = 0,     d Rlower_x/dx at x=0 = 0.
```

Writing `T_x = d^2 Rlower_x/dx^2`, the exact Taylor integral identity gives

```text
nu R_x >= nu Rlower_x = integral_0^x (x-t) nu T_t dt.
```

The code certifies nonnegative minorant averages for `T_x` throughout each
complete request interval. Consequently `nu T_x >= 0` on `[0, 1/10]`, which
proves the desired residual inequality. This uses an exact integral identity,
not a second-order polynomial approximation to the residual. The reported
small-request lower bounds apply to the minorant averages of `T_x`, whereas
the large-request lower bounds apply to those of `C_left,right`.

### Shared potential tables and error control

Both verifiers use the same exact potential. Write

```text
G(z) = Gamma(-log(z)/lambda),
H(w) = w G(w) + integral_w^1 G(v) dv.
```

1. The integral formulas for `G`, `G'`, and `G''` are bounded by finite
   weighted endpoint sums. The power weight is integrated analytically on
   each cell; the exponential factor is bounded by its endpoint values.
   These bounds also provide a rigorous enclosure of `Gamma(0)`.
2. Monotonicity of `G`, `G'`, and `G''` turns their grid bounds into bounds
   on complete argument intervals. No interpolation or sampled extrema are
   used. The derivatives belong to the exact potential, not the table's
   step-function approximation.
3. `H` is bounded by finite endpoint sums, with derivative bounds obtained
   from `H' = w G'` and `H'' = G' + w G''`. All primes here differentiate
   `z` or `w`, not the original age.
4. The small-request `u` integral is enclosed on each entire `u` cell.
   Multiplying that interval enclosure by the positive cell length and
   summing gives a rigorous integral enclosure, even if the integrand
   changes sign. No Taylor quadrature is used.
5. Arb controls rounding and propagated input uncertainty throughout.
   The finite-sum upper and lower bounds also account for the integration
   error; the method does not require an infinite-series truncation tail.

Large requests query function values only. Small requests additionally use
first and second coordinate derivatives. The combined run reuses the same
extended table for both. Quantities depending only on a small-request
interval and a `u` cell are cached across age cells; all remaining age
dependence is still enclosed on every complete age cell.

### Default coverage

| Parameter | Shared / small requests | Large requests |
|---|---:|---:|
| Poisson rate | `9/2` | `9/2` |
| Arb precision | 160 bits | 160 bits |
| Potential integration cells | 8192 | Shared table |
| Potential lookup cells | 8192 | Shared table |
| Request intervals | 128 | 144 |
| Age-coordinate cells per interval | 1024 | 512 |
| Kernel integration cells | 256 in `u` | 256 per kernel integral |

The small-request intervals are `[j/1280, (j+1)/1280]` for
`j = 0, ..., 127`. They include `x=0`. Both age-coordinate endpoints
`z=0` and `z=1` are covered; `z=0` represents the infinite-age limit.

### Timing and reproducibility

`results/summary.json` records the measured shared precomputation time,
the verification time for each selected part, and the total run time.
Shared precomputation is counted once in a combined run. Separate
invocations each rebuild the tables, so their total times should not be
added to estimate the shared-table run.

The included complete combined run passed all 128 small-request and 144
large-request intervals. Every interval bound agrees exactly with the
corresponding previous standalone finite-sum certificate.

| Measured stage | Seconds |
|---|---:|
| Shared potential-table construction | 42.2280 |
| Small-request verification | 108.2424 |
| Large-request verification | 131.1727 |
| Total, including reporting overhead | 281.6472 |

The minimum certified minorant averages were `0.127708250491939315...`
for small requests and `0.0006132087436977444425...` for large requests.
These are bounds on different quantities (`T_x` and `C_left,right`), as
explained above. The run used Python 3.12.14 and python-flint 0.9.0 on
Linux x86_64.

Runtime depends on the machine and concurrent load. Use the recorded
Python version, `python-flint` version, precision, and source hashes when
comparing runs. The supplied timings are measurements, not estimates.

A negative computed lower bound would mean that the chosen enclosure has
not certified the inequality. It would not, by itself, be a counterexample
to the mathematical statement.

## AdWords potential verification

The AdWords verifier certifies the paper's potential with exact initial value
`53/100`, for every request mass `0 <= x <= 1` and every reachable state.
The supplied candidate contains 299 rational coefficients for the degree-12
prices `F_s`, `G_s`, and continuation slope `h`, with `lambda = 9/2`.
The state LP uses interval masses and the analytic signed score
`H(z) = 103/100 - Ein(9/2) + Ein(9*z/2)`.

The two Python entry points are self-contained. Their only candidate input
is `adwords/candidate.json`; mesh suggestions, score multipliers, and the
initial request-mass grid are embedded in `verify.py`. A fresh run recomputes
all bounds and witnesses and does not read the supplied results as inputs.

### Run

From the repository root:

```bash
python3 -m venv /tmp/adwords-verification
/tmp/adwords-verification/bin/python -m pip install -r adwords/requirements.txt
/tmp/adwords-verification/bin/python adwords/verify.py \
    --output /tmp/adwords-rerun-results --workers 2
```

Use a new, empty output directory. The full driver rejects a nonempty one,
so the supplied `adwords/results/` can be retained for comparison. The pinned
dependencies were tested with Python 3.14.8: NumPy 2.5.4, SciPy 1.18.1, and
python-flint 0.9.0. Matplotlib is not required. Use ordinary Python without
`-O` or `PYTHONOPTIMIZE`.

The command recomputes the accounting and signed-score certificates, then
independently audits the exact witnesses, mass-curvature bounds, and complete
covers. Arb state is isolated in subprocesses. `--workers` controls workers
per numerical component; it changes runtime, not the accepted inequalities.
The complete verification can take several minutes and exits nonzero on failure.

Success requires all of these fields in the new `report.json`:

```text
completed: true
scope: full_adwords_interval_certificate
status: passed
execution_mode: fresh_component_runs_and_exact_audit
enclosures_recomputed_in_this_invocation: true
```

To audit the supplied certificate without repeating numerical integrations:

```bash
/tmp/adwords-verification/bin/python adwords/verify.py \
    --output adwords/results --audit-existing
```

This is read-only and prints its report. It checks initialization and shape
conditions, reconstructs all rational LP residuals, recomputes the curvature
bounds, and checks the mass cover and signed-score transfer and rectangle
covers. It does not regenerate kernel or score integration enclosures. Its
report records `execution_mode: audit_completed_component_runs` and
`enclosures_recomputed_in_this_invocation: false`.

The signed-score component can also be reproduced separately:

```bash
/tmp/adwords-verification/bin/python adwords/verify_score.py \
    --output /tmp/adwords-score-results --workers 2
```

Use a new directory for this component as well. A score-only run proves the
signed-score invariant; it does not certify the full competitive ratio.
Its saved covers can be audited with:

```bash
python adwords/verify_score.py --output adwords/results/score --audit-only
```

The score-only audit needs just the Python standard library.

### Files and results

Paths in this table are relative to `adwords/`.

| File | Purpose |
|---|---|
| `verify.py` | Full entry point; price parsing, kernel enclosures, interval LPs, mass-curvature bounds, and independent exact audit. |
| `verify_score.py` | Signed-score kernel, interval derivatives, adaptive rectangle verification, and exact transfer/coverage audit. |
| `candidate.json` | Exact rational price coefficients, load knots, and normalization. |
| `requirements.txt` | Pinned numerical dependencies. |
| `results/report.json`, `results/audit.json` | Combined historical certificate and its audit. |
| `results/prices/report.json` | Price-component completion and generation provenance. |
| `results/prices/polynomial.json` | Initialization, shape conditions, and 22 small-request witnesses. |
| `results/prices/zero.json` | The 34 generator-limit witnesses. |
| `results/prices/endpoint-summary.json` | Bounds at every request-mass endpoint. |
| `results/prices/endpoints.jsonl.gz` | All nonzero endpoint certificates: interval costs and rational LP witnesses, including the tiny endpoint derived from the generator and curvature. |
| `results/prices/mass-cover.json` | Exact gap-free mass cover and bounds for all 34 families. |
| `results/score/report.json` | Analytic signed-score summary and generation provenance. |
| `results/score/score-old.json`, `results/score/score-young.json` | Complete analytic-score rectangle covers. |
| `results/score/smooth/moment-old.json`, `results/score/smooth/moment-young.json` | Smooth-score bounds needed to check the exact constant transfer. |

Here “old” and “young” refer to the two age-coordinate domains, not to
different versions of the certificate.

`endpoints.jsonl.gz` is gzip-compressed JSON Lines: one complete JSON object
per nonzero request-mass endpoint. The supplied archive contains 167 records
(166 LP certificates and one generator-curvature certificate); `x` identifies
the mass and `scope` identifies the proof type. The zero-mass certificate
remains in `zero.json`. Fresh runs write this same format, and the audit reads
one record at a time and rejects missing, duplicate, or unexpected endpoints.

### Mathematical certificate

1. Exact fractions check the initialization `53/100` and every coefficient
   shape and load condition. Rigorous interval minima certify all 22
   small-request conditions and all 34 zero-mass generator conditions.
2. For positive request masses, outward Arb arithmetic encloses the complete
   matching kernel. Polynomial expansions with analytic remainder bounds,
   exact branch splitting, and Bernstein subdivision yield constant lower
   costs on entire age intervals. The accounting computation uses 384-bit
   precision. Floating interpolation only proposes approximations.
3. The state LP imposes tail restrictions on `nu0` and `nu0 + 2*nu1`.
   A nonnegative multiplier subtracts an upper bound for the signed score
   from the cost. SciPy proposes LP dual multipliers; exact rational residual
   corrections certify their bounds. Cumulative-coordinate residuals are
   converted back to raw interval masses by suffix sums. Floating objective
   values and solver status alone never constitute a certificate.
4. Analytic curvature bounds for the original accounting cost cover every
   mass between endpoints. Exact quadratic minima certify each positive
   interval; the generator derivative and curvature certify the first
   interval starting at zero. Rational intervals must meet exactly and cover
   `[0, 1]` for all 34 large-request families.
5. The signed-score invariant uses two complete coordinate squares:
   `z = exp(-9*x/2)*u` and `z = exp(-9*x*u/2)`. At 80-bit precision, Arb
   encloses corner integrals and pure second derivatives. Interpolation-error
   bounds certify whole rectangles, which are subdivided until positive.
   Exact dyadic coverage checks rule out gaps and overlaps.
6. The score evaluator's rational constant is transferred to the exact
   analytic `H` using an alternating-series enclosure. The additional defect
   allowance is less than `8.897e-18` and is deducted from every rectangle.
   Saved lower bounds are rounded downward, and nonfinite enclosures are
   never accepted.

These accounting and invariant certificates establish the paper's potential
lemma and its `53/100` competitive guarantee. Kernel and score integrations
are recomputed only by a fresh run; the independent saved-certificate audit
checks the exact witnesses and coverage built from those enclosures.

### Reported results and reproducibility

The supplied certificate has the following counts and bounds:

| Check | Certified coverage | Lower bound |
|---|---|---:|
| Initialization | Exact rational coefficients | `53/100` |
| Small-request accounting | 22 conditions | `> 1.6489e-5` |
| Large-request accounting | 167 intervals, 34 families | `> 4.6426e-7` |
| Signed-score normalized defect | 1,810 rectangles | `> 5.0379e-5` |

There are 168 mass endpoints, 5,700 exact LP witnesses, and at most 2,763
state intervals per LP. The first mass interval is `[0, 2^-21]`; the score
cover comprises 629 old-age and 1,181 young-age rectangles. All 34
generator-limit conditions pass.

The supplied certificate values are unchanged from the paper's certificate;
the individual endpoint records have been combined into one archive.
Their component reports retain the original `source_sha256` fingerprints,
including historical filenames from before the implementation was
consolidated. The audit recognizes these specific historical fingerprints
and explicitly reports that the retired generation source files were not
rechecked. It records the current audit implementation separately; it does
not relabel historical enclosures as newly generated. Fresh outputs record
the hashes of `adwords/verify.py` and `adwords/verify_score.py`.

Inspect both the completion fields and execution mode: the supplied combined
report was assembled by auditing completed component runs, not by rerunning
all integrations in that same invocation. The code rejects unrecognized
generation fingerprints. Recognizing a historical fingerprint does not
repeat its numerical integrations.

A fresh run of the consolidated implementation with the dependency versions
above reproduced all reported counts and passed the complete audit. Its
minimum large-request bound was `4.629976197430433e-7`; the small-request and
score bounds matched the saved values exactly. Rerunning the original code
in the same environment produced identical certificates at the limiting
endpoints, so this small difference is not caused by consolidation. Both
runs certify the same `53/100` ratio. Numerical library versions can change
valid proposals, witnesses, partitions, and positive margins. Use
`--audit-existing` to check the precise supplied paper certificate and a
fresh run to regenerate all enclosures in your environment.

## Load Balancing verification

The scripts in [`load-balancing/`](load-balancing/) verify the numerical bounds used in the covariance and amortized analyses for minimizing squared loads on unrelated machines.

### Run

From the repository root:

```bash
cd load-balancing
python -m pip install python-flint sympy

python verify_kernel.py
python verify_reserve.py
python verify_kappa.py
python verify_credit_reserve.py
python verify_local.py
python check_uncapped_example.py
```

Use Python without `-O`, since the scripts use assertions. A successful run finishes without an assertion or verification error.

### Verification

| Script | What it verifies | Method |
|---|---|---|
| [`verify_kernel.py`](load-balancing/verify_kernel.py) | The finite-grid inequalities for the covariance kernel. | Interval arithmetic and lower Riemann sums. |
| [`verify_reserve.py`](load-balancing/verify_reserve.py) | The integral bound for the reserve and potential. | Interval arithmetic. |
| [`verify_kappa.py`](load-balancing/verify_kappa.py) | The integral bound used in the constraints on `kappa`. | Interval arithmetic. |
| [`verify_credit_reserve.py`](load-balancing/verify_credit_reserve.py) | The positive-part integral bound in the credit–reserve constraint. | Interval bounds on integration cells, plus a bound on the tail. |
| [`verify_local.py`](load-balancing/verify_local.py) | The three polynomial cases of the local amortized inequality. | Exact arithmetic and subdivision. |
| [`check_uncapped_example.py`](load-balancing/check_uncapped_example.py) | The example where the uncapped reserve fails the amortized comparison. | Interval arithmetic. |

The main results are:

- The kernel verifier checks all **22,500,000** inequalities in the paper's finite grid, together with **1,500** additional checks at `s = 3/2`.
- The reserve coefficient is less than `0.00539`; the integral of `delta` is between `0.219` and `0.22`; and the positive-part integral is less than `0.004`.
- The local polynomial verifier terminates with **8,056 boxes**: `7,953`, `22`, and `81` in the three cases.
- In the uncapped-reserve example, the normalized slack on the final job is approximately `-0.0222`, showing that the amortized comparison fails.

These scripts verify the numerical bounds. The analytic reductions and the extension of the kernel bound to the infinite tail are given in the paper.

### Figure

To generate the covariance-kernel comparison figure, first run `verify_kernel.py`, which writes `kernel_bounds.csv`. Then run:

```bash
python -m pip install numpy matplotlib
python plot_kernel.py
```

The script [`plot_kernel.py`](load-balancing/plot_kernel.py) writes `selected-kernel-comparison.pdf` using the certified lower bounds in `kernel_bounds.csv`.
