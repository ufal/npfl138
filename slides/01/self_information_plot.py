#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

xs = np.geomspace(1e-6, 1.0, 100)
ys = -np.log(xs)

plt.rcParams.update({"text.usetex": True})

plt.figure(figsize=(3, 4))
plt.plot(xs, ys)
point, = plt.plot(xs[-1], ys[-1], marker="o", markersize=4, color="C0", zorder=3)
point.set_clip_on(False)
plt.xlim([0.0, 1.0])
plt.xlabel("Outcome $x$ probability $P(x)$")
plt.ylim([0, 6])
plt.ylabel("Self-information $I(x)$ [nats]")
plt.grid(linestyle=":", color="gray")
plt.savefig("self_information_plot.svg", transparent=True, bbox_inches="tight")
plt.show()
