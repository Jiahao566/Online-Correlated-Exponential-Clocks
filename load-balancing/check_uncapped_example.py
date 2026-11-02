from flint import arb, ctx

ctx.prec = 160
C, A = arb(43) / 50, arb(25) / 3
GAMMA = arb(4573) / 1000
x = arb(1) / 50


def delta(s):
    return C * (-s - A*s*s).exp()


def tail(s):
    root = A.sqrt()
    return (
        C * arb.pi().sqrt() / (2*root)
        * (1 / (4*A)).exp()
        * (root*s + 1 / (2*root)).erfc()
    )


def integrated_tail(s):
    return delta(s)/(2*A) - (s + 1/(2*A))*tail(s)


def phi(alpha, p):
    return p*p + alpha*alpha if alpha <= p else 2*p*alpha


alpha = arb(1)  # Preloaded load is 2.
history = []

for j, factor in enumerate([arb(3)/2]*30 + [arb(5)/8]):
    p = factor*alpha
    alpha_new = alpha + p*x/2
    raw_cost = (p*p + 4*p*alpha)*x
    price = x*phi(alpha_new, p)

    independent_slack = (
        GAMMA*(price - (alpha_new**2 - alpha**2))
        - raw_cost
    )
    assert independent_slack < 0

    reserve, saving = arb(0), arb(0)
    for k, pk in enumerate(history):
        age = (j-k-1)*x
        before = (
            integrated_tail(age)
            - integrated_tail(age+x)
        )
        after = (
            integrated_tail(age+x)
            - integrated_tail(age+2*x)
        )
        reserve += arb(3)/10 * pk*before
        saving += 2*pk*(before-after)

    deposit = (
        arb(3)/10 * p
        * (integrated_tail(arb(0)) - integrated_tail(x))
    )
    reserve_new = reserve - arb(3)/20*saving + deposit

    old_aux = alpha-reserve
    new_aux = alpha_new-reserve_new
    slack = (
        GAMMA*(price - (new_aux**2 - old_aux**2))
        - (raw_cost - p*saving)
    )

    if j == 30:
        assert slack < 0
        print("Final normalized slack:", slack/(p*p*x))

    history.append(p)
    alpha = alpha_new
