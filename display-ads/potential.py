"""Finite-sum enclosures of the matching potential and its derivatives.

The only dependency is python-flint.  The caller sets ``flint.ctx.prec``.
All integration is bounded by weighted endpoint sums; no infinite series,
floating-point grid indices, or sampled extrema are used.

In z = exp(-lambda*a), write G(z) = Gamma(a) and

    J_r(z) = integral_0^1 u^(q/lambda - 1 + r) exp(-lambda*z*u) du.

Then G = (mu + lambda*Gamma(0))/q + (1-Gamma(0))*J_0,
G' = -lambda*(1-Gamma(0))*J_1, and
G'' = lambda^2*(1-Gamma(0))*J_2.  Primes in this module always
differentiate the coordinate z (or w for H), not the original age.
"""

from flint import arb


def _lower(value):
    assert value.is_finite(), value
    return value.lower()


def _upper(value):
    assert value.is_finite(), value
    return value.upper()


def _hull(lo, hi):
    assert lo <= hi, (lo, hi)
    return lo.union(hi)


class PotentialTable:
    """Enclose G, G', G'' and H, H', H'' on complete input intervals.

    ``gamma(z)`` and ``reset(w)`` return tuples of three Arb balls.
    An input ball is intersected with [0,1]; a disjoint or nonfinite
    input is rejected.  H(w) = w*G(w) + integral_w^1 G(v) dv.
    """

    def __init__(self, integration_cells, table_cells, lam, mu):
        if integration_cells < 1 or table_cells < 1:
            raise ValueError("Both grid sizes must be positive integers")
        self.size = table_cells
        self.integration_cells = integration_cells
        self.lam, self.mu = arb(lam), arb(mu)
        assert self.lam > 0 and self.mu > 0
        self.q = 1 + self.lam + self.mu
        power = self.q / self.lam

        # Integrate the three power factors exactly on each u cell.
        # Higher powers reuse u^power, avoiding two extra exponentiations.
        weights = []
        previous = (arb(0), arb(0), arb(0))
        for k in range(1, integration_cells + 1):
            u = arb(k) / integration_cells
            first = u ** power
            current = (first, first*u, first*u*u)
            weight = tuple((current[r]-previous[r])/(power+r)
                           for r in range(3))
            assert all(w > 0 for w in weight)
            weights.append(weight)
            previous = current
        reverse_weights = list(reversed(weights))

        def j_bounds(z):
            ratio = (-self.lam*z/integration_cells).exp()
            # Each Horner value is a left-endpoint sum; multiplication
            # by ratio gives the corresponding right-endpoint sum.
            value0, value1, value2 = arb(0), arb(0), arb(0)
            for w0, w1, w2 in reverse_weights:
                value0 = value0*ratio + w0
                value1 = value1*ratio + w1
                value2 = value2*ratio + w2
            return tuple((_lower(ratio*v), _upper(v))
                         for v in (value0, value1, value2))

        at_one = j_bounds(arb(1))

        def initial_potential(j):
            return (self.mu+self.q*j)/(1+self.mu+self.q*j)

        self.g0_lower = _lower(initial_potential(at_one[0][0]))
        self.g0_upper = _upper(initial_potential(at_one[0][1]))
        assert 0 < self.g0_lower <= self.g0_upper < 1

        base_lo = (self.mu+self.lam*self.g0_lower)/self.q
        base_hi = (self.mu+self.lam*self.g0_upper)/self.q
        remaining_lo = 1-self.g0_upper
        remaining_hi = 1-self.g0_lower
        self.g_lower, self.g_upper = [], []
        self.gd_lower, self.gd_upper = [], []
        self.gdd_lower, self.gdd_upper = [], []

        for k in range(table_cells + 1):
            if k == 0:
                # At z=0 the exponential is one, so these integrals
                # have exact formulas rather than quadrature gaps.
                js = tuple((_lower(1/(power+r)), _upper(1/(power+r)))
                           for r in range(3))
                limit = (self.mu+self.lam)/self.q
                glo, ghi = _lower(limit), _upper(limit)
            else:
                js = at_one if k == table_cells else j_bounds(arb(k)/table_cells)
                if k == table_cells:
                    glo, ghi = self.g0_lower, self.g0_upper
                else:
                    # For the exact J_0, lambda/q-J_0 >= 0.  Therefore
                    # G is nondecreasing in Gamma(0) and in J_0, and
                    # paired endpoint substitution avoids dependency loss.
                    glo = _lower(base_lo+(1-self.g0_lower)*js[0][0])
                    ghi = _upper(base_hi+(1-self.g0_upper)*js[0][1])
            self.g_lower.append(glo)
            self.g_upper.append(ghi)
            self.gd_lower.append(_lower(-self.lam*remaining_hi*js[1][1]))
            self.gd_upper.append(_upper(-self.lam*remaining_lo*js[1][0]))
            self.gdd_lower.append(_lower(self.lam**2*remaining_lo*js[2][0]))
            self.gdd_upper.append(_upper(self.lam**2*remaining_hi*js[2][1]))

        # H uses monotone endpoint sums for its remaining integral.
        self.h_lower = [arb(0)]*(table_cells+1)
        self.h_upper = [arb(0)]*(table_cells+1)
        self.h_lower[-1], self.h_upper[-1] = self.g0_lower, self.g0_upper
        suffix_lo, suffix_hi = arb(0), arb(0)
        for k in range(table_cells-1, -1, -1):
            suffix_lo += self.g_lower[k+1]/table_cells
            suffix_hi += self.g_upper[k]/table_cells
            z = arb(k)/table_cells
            self.h_lower[k] = _lower(z*self.g_lower[k]+suffix_lo)
            self.h_upper[k] = _upper(z*self.g_upper[k]+suffix_hi)

        assert all(0 <= lo <= hi <= 1 for lo, hi in
                   zip(self.g_lower, self.g_upper))
        assert all(lo <= hi < 0 for lo, hi in
                   zip(self.gd_lower, self.gd_upper))
        assert all(0 < lo <= hi for lo, hi in
                   zip(self.gdd_lower, self.gdd_upper))
        assert all(0 <= lo <= hi <= 1 for lo, hi in
                   zip(self.h_lower, self.h_upper))

    def _indices(self, value):
        value = arb(value)
        if not value.is_finite():
            raise ValueError("The input must be finite")
        lo = max(arb(0), value.lower())
        hi = min(arb(1), value.upper())
        if lo > hi:
            raise ValueError("The input interval is disjoint from [0,1]")
        i = int((lo*self.size).lower().floor().unique_fmpz())
        j = int((hi*self.size).upper().ceil().unique_fmpz())
        i = max(0, min(self.size, i))
        j = max(0, min(self.size, j))
        assert i <= j
        return i, j, _hull(lo, hi)

    def _gamma_at_indices(self, i, j):
        # G and G'' are nonincreasing; G' is nondecreasing.
        return (_hull(self.g_lower[j], self.g_upper[i]),
                _hull(self.gd_lower[i], self.gd_upper[j]),
                _hull(self.gdd_lower[j], self.gdd_upper[i]))

    def gamma(self, z):
        """Return enclosures of (G(z), G'(z), G''(z))."""
        i, j, _ = self._indices(z)
        return self._gamma_at_indices(i, j)

    def reset(self, w):
        """Return enclosures of (H(w), H'(w), H''(w)).

        H is nonincreasing.  Its derivatives need not be monotone,
        so they are enclosed using H'=w*G' and H''=G'+w*G''.
        """
        i, j, clipped = self._indices(w)
        _, gd, gdd = self._gamma_at_indices(i, j)
        return (_hull(self.h_lower[j], self.h_upper[i]),
                clipped*gd, gd+clipped*gdd)

    def _value_bounds(self, value, lowers, uppers):
        """Return endpoint bounds without an additional Arb hull conversion.

        This is the value-only interface used by the large-request code.
        Preserve the original outward-rounded index calculation.
        """
        value = arb(value)
        assert value.is_finite() and not (value < 0) and not (value > 1)
        scaled = value*self.size
        i = int(scaled.lower().floor().unique_fmpz())
        j = int(scaled.upper().ceil().unique_fmpz())
        i = max(0, min(self.size, i))
        j = max(0, min(self.size, j))
        assert i <= j
        return lowers[j], uppers[i]

    def gamma_bounds(self, z):
        """Return lower and upper bounds on G(z), without derivatives."""
        return self._value_bounds(z, self.g_lower, self.g_upper)

    def reset_bounds(self, w):
        """Return lower and upper bounds on H(w), without derivatives."""
        return self._value_bounds(w, self.h_lower, self.h_upper)
