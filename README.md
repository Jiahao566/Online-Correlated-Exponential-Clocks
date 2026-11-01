# Online-Correlated-Exponential-Clocks
## Matching and Display Ads potential verification

This project certifies the potential accounting inequality for every request
mass `0 <= x <= 1` and every measure reachable from the stationary age
distribution by matching and gap updates, with `lambda = 9/2`.

The small- and large-request verifiers share one potential implementation.
Running both together constructs the potential tables once and reuses them.
All numerical bounds use 160-bit Arb interval arithmetic through
`python-flint`.

### Run

From this directory:

```bash
python -m pip install -r requirements.txt
python verify.py
```

To preserve the supplied results, write a fresh run to another directory:

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
| `requirements.txt` | Pinned Python dependency. |
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
