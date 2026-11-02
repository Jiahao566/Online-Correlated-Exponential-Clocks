from flint import arb, ctx

ctx.prec = 128
A, C = arb(25) / 3, arb(43) / 50


def delta(s):
    return C * (-s - A*s*s).exp()


def tail(s):
    # Integral_s^infinity delta(u) du.
    root = A.sqrt()
    return (
        C * arb.pi().sqrt() / (2 * root)
        * (1 / (4*A)).exp()
        * (root*s + 1 / (2*root)).erfc()
    )


N, T = 10000, arb(3)
h = T / N
upper = arb(0)

for k in range(N):
    l, u = k*h, (k+1)*h

    # On [l,u], delta(s) <= delta(l), tail(s) >= tail(u).
    bound = (
        arb(7) / 100 * delta(l)
        - arb(3) / 10 * tail(u)
    )
    upper += h * max(arb(0), arb(bound.upper()))

# Beyond T, the positive part is at most 0.07*delta(s).
upper += arb(7) / 100 * tail(T)

assert upper < arb(4) / 1000
print("Certified upper bound:", upper)
