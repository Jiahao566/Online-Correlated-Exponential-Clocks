from math import comb, lcm
from sympy import Poly, Rational as Q, symbols

T = symbols("t1:6")
ZERO = (0,) * 5

x = Q(3, 20) * T[1]
kappa = Q(11, 50) - (
    Q(43, 100)*x + Q(1, 1000)
) * T[3]
c = kappa * T[2]
r = Q(7, 100)*c - Q(1, 250) + Q(739, 50000)*T[4]

cases = [
    (1 - x/2)*T[0],
    1 - x*T[0]/2,
    1 + Q(3, 4)*T[0],
]


def bisect(coefficients, axis):
    degree = max(m[axis] for m in coefficients)
    left, right = {}, {}

    for m, value in coefficients.items():
        # Positive scaling clears the dyadic denominators.
        value *= 2**(degree - m[axis])
        left[m] = value

        for j in range(m[axis] + 1):
            n = m[:axis] + (j,) + m[axis+1:]
            right[n] = (
                right.get(n, 0)
                + value * comb(m[axis], j)
            )

    return left, {m: b for m, b in right.items() if b}


def verify(expression):
    terms = Poly(expression, *T).terms()
    denominator = lcm(*(int(b.q) for _, b in terms))
    stack = [
        ({m: int(b*denominator) for m, b in terms}, 0)
    ]
    leaves = 0

    while stack:
        coefficients, depth = stack.pop()
        bound = coefficients.get(ZERO, 0) + sum(
            b for m, b in coefficients.items()
            if m != ZERO and b > 0
        )

        if bound < 0:
            leaves += 1
            continue

        if depth >= 100:
            raise RuntimeError("Unresolved box; verification failed.")

        left, right = bisect(coefficients, depth % 5)
        stack.extend([
            (right, depth + 1),
            (left, depth + 1),
        ])

    return leaves


counts = []
for case, a in enumerate(cases, 1):
    cap = a if case < 3 else 1
    phi = (
        1 + (a + x/2)**2
        if case == 1 else 2*(a + x/2)
    )
    z = Q(1, 2) - Q(3, 10)*cap*kappa + Q(3, 10)*a*c

    polynomial = (
        1 + 4*a - 2*a*c
        - Q(4573, 1000) * (
            phi - 2*a*(1-r)*z - x*z*z
        )
    )

    count = verify(polynomial)
    counts.append(count)
    print(f"Case {case}: {count} terminal boxes")

assert counts == [7953, 22, 81]
print("PASS:", sum(counts), "terminal boxes")
