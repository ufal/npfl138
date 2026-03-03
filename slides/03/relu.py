#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

xs = np.linspace(-4, 4, 101)
ys = np.maximum(0, xs)
ds = np.where(ys > 0, 1, 0)

plt.figure(figsize=(10,3))
plt.plot(xs, ys, label="ReLU")
plt.plot(np.where(xs == 0, np.nan, xs), np.where(xs == 0, np.nan, ds), label="Derivative of ReLU")
plt.xlabel("x")
plt.xlim(-3.5, 3.5)
plt.yticks(np.arange(0, 4.0001, step=1))
plt.gca().set_aspect(1)
plt.grid(True)
plt.title("Plot of the ReLU Function")
plt.legend(loc="upper left")
plt.savefig("relu.svg", transparent=True, bbox_inches="tight")
