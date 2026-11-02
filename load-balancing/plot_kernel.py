import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.loadtxt(
    "kernel_bounds.csv", delimiter=",", skiprows=1
)
s, lower = data.T
delta = 0.86*np.exp(-s - (25/3)*s*s)

fig, axes = plt.subplots(
    1, 2, figsize=(10, 3.5), constrained_layout=True
)
axes[0].plot(
    s, lower, ".", markersize=3,
    label="Certified lower bounds",
)
axes[0].plot(s, delta, "--", label=r"$\delta(s)$")
axes[0].set_ylabel("Covariance kernel")
axes[0].legend(frameon=False)

axes[1].plot(s, lower-delta, ".", markersize=3)
axes[1].axhline(0, color="black", linewidth=0.6)
axes[1].set_ylabel(r"Lower bound $-\delta(s)$")

for ax in axes:
    ax.set_xlabel(r"Separation $s$")
    ax.grid(alpha=0.2)

fig.savefig("selected-kernel-comparison.pdf")
