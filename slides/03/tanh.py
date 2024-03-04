#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

xs = np.linspace(-4, 4, 101)
ys = np.tanh(xs)
ds = 1 - ys * ys

plt.figure(figsize=(10,3))
plt.plot(xs, ys, label="Tanh")
plt.plot(xs, ds, label="Derivative of tanh")
plt.xlabel("x")
plt.xlim(-3.5, 3.5)
plt.yticks(np.arange(-1.0, 1.0001, step=0.5))
plt.gca().set_aspect(1)
plt.grid(True)
plt.title("Plot of the Tanh Function")
plt.legend(loc="upper left")
plt.savefig("tanh.svg", transparent=True, bbox_inches="tight")
