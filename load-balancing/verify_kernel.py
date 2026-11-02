import csv
import multiprocessing as mp
from flint import arb, ctx

DEN = 10000
NX, NU, NS = 1500, 30000, 15000


def initialize():
    global L, h, u, E, target, earlier
    ctx.prec = 100
    L, h = arb(9) / 2, arb(1) / DEN
    u = [arb(n) / DEN for n in range(NU + 1)]
    E = [(-L * s).exp() for s in u]
    target = [
        arb(43) / 50 * (-s - arb(25) / 3 * s*s).exp()
        for s in u[:NS + 1]
    ]
    xb = arb(3) / 20
    earlier = (
        1 - (1 - L*xb*(-L*xb).exp())**(1/xb)
        - L*xb*(-L).exp()
    )


def verify_row(m):
    xl, xh = arb(m) / DEN, arb(m + 1) / DEN
    exponent = arb(DEN) / (m + 1)
    prefactor = (1 - xh) * L**3
    total = arb(0)
    samples = {}

    for n in range(NU - 1, -1, -1):
        # total = sum_{k=n+1}^{29999} g_mk.
        # n < NS: original rectangle checks.
        # n = NS: additional endpoint check.
        if n <= NS:
            lower = earlier * h * total
            if not lower > target[n]:
                raise ArithmeticError(
                    f"Failed or inconclusive: m={m}, n={n}"
                )
            if n % 100 == 0:
                samples[n] = int(
                    (lower.lower() * 10**20)
                    .lower().floor().unique_fmpz()
                )

        # F_{x_{m+1}}(u_n).
        if n >= m + 1:
            fh = 1 - L*xh*E[n]
        else:
            fh = 1 + E[m+1] - (1 + L*u[n])*E[n]

        # F_{x_m}(u_{n+1}); includes F_0 = 1.
        if n + 1 >= m:
            fl = 1 - L*xl*E[n+1]
        else:
            fl = 1 + E[m] - (1 + L*u[n+1])*E[n+1]

        small = arb(min(n, m + 1)) / (m + 1)
        total += (
            prefactor * E[n+1]**2 * small
            * (exponent * fh.log()).exp() / fl**2
        )

    return samples


def main():
    curve = {}
    with mp.Pool(4, initializer=initialize) as pool:
        for count, samples in enumerate(
            pool.imap_unordered(verify_row, range(NX)), 1
        ):
            for n, value in samples.items():
                curve[n] = min(curve.get(n, value), value)
            if count % 100 == 0:
                print(f"Certified {count}/{NX} rows", flush=True)

    # Certified downward-rounded values for the figure.
    with open("kernel_bounds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["s", "certified_lower_bound"])
        for n in sorted(curve):
            value = curve[n]
            writer.writerow([
                f"{n // DEN}.{n % DEN:04d}",
                f"{value // 10**20}.{value % 10**20:020d}",
            ])

    print("PASS: 22,500,000 original rectangle checks.")
    print("PASS: 1,500 additional endpoint checks.")


if __name__ == "__main__":
    main()
