from flint import arb, ctx

ctx.prec = 160
A, C = arb(25) / 3, arb(43) / 50
root = A.sqrt()


def integral(b):
    # Integral_0^infinity C exp(-A*s^2 - b*s) ds.
    return (
        C * arb.pi().sqrt() / (2 * root)
        * (b*b / (4*A)).exp()
        * (b / (2*root)).erfc()
    )


value = arb(3) / 10 * (
    integral(arb(1)) - integral(arb(3) / 2)
)
assert value < arb(539) / 100000
print("Certified reserve coefficient:", value)
