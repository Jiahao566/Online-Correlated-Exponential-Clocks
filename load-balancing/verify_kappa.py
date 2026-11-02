from flint import arb, ctx

ctx.prec = 160
A, C = arb(25) / 3, arb(43) / 50
root = A.sqrt()

value = (
    C * arb.pi().sqrt() / (2 * root)
    * (1 / (4*A)).exp()
    * (1 / (2*root)).erfc()
)

assert value > arb(219) / 1000
assert value < arb(22) / 100
print("Certified integral of delta:", value)
