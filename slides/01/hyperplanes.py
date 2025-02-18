#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

fig = plt.figure(figsize=(15, 4.3))

# 1D, hyperplane x1 = 0.5
x = np.random.rand(100)
y = np.where(x > 0.5, 1, 0)

ax = fig.add_subplot(131)
ax.scatter(x, np.zeros_like(x), c=np.where(y == 1, "blue", "red"))
ax.scatter([0.5], [0.0], c="green", label="Hyperplane (x1 = 0.5)")
ax.set_title("1D Hyperplane")
ax.set_xlabel("x1")
ax.set_yticks([])
ax.legend()

# 2D, hyperplane x1 + x2 = 1
x = np.random.rand(100, 2)
y = np.where(x[:, 0] + x[:, 1] > 1, 1, 0)

ax = fig.add_subplot(132)
ax.scatter(x[:, 0], x[:, 1], c=np.where(y == 1, "blue", "red"))
x1_line = np.linspace(0, 1, 2)
x2_line = 1 - x1_line
ax.plot(x1_line, x2_line, color="green", linestyle="--", label="Hyperplane (x1 + x2 = 1)")

ax.set_title("2D Hyperplane")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()

# 3D, hyperplane x1 + x2 + x3 = 1.5
x = np.random.rand(100, 3)
y = np.where(x[:, 0] + x[:, 1] + x[:, 2] > 1.5, 1, 0)

ax = fig.add_subplot(133, projection="3d")
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=np.where(y == 1, "blue", "red"))
x1_plane = np.linspace(0, 1, 2)
x2_plane = np.linspace(0, 1, 2)
X1, X2 = np.meshgrid(x1_plane, x2_plane)
X3 = 1.5 - X1 - X2
ax.plot_surface(X1, X2, X3, alpha=0.5, color="green", label="Hyperplane (x1 + x2 + x3 = 1.5)")

ax.set_title("3D Hyperplane")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.legend()

plt.savefig("hyperplanes.svg", transparent=True)
