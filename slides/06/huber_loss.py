#!/usr/bin/env python3
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["mathtext.fontset"] = "cm"

xs = np.linspace(-2.5, 2.5, 51)
l2 = xs * xs / 2
huber = np.where(np.abs(xs) <= 1, xs * xs / 2, np.abs(xs) - 0.5)
d_huber = np.where(np.abs(xs) <= 1, np.abs(xs), 1)

plt.figure(figsize=(5, 3.5))
plt.plot(xs, l2, label="L2 loss $\\frac{1}{2} x^2$")
plt.plot(xs, huber, label="Huber loss")
plt.plot(xs, d_huber, label="Huber loss derivative")
plt.gca().set_aspect(1)
plt.grid(True)
plt.legend(loc="upper center")
plt.savefig("huber_loss.svg", bbox_inches="tight", transparent=True)
