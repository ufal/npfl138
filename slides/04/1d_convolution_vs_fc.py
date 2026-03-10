#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


def draw_network(ax, layer_type):
    # Number of nodes.
    n = 6

    # Coordinates for inputs (left) and outputs (right).
    # We use n-1-i to draw the first nodes at the top.
    inputs_x = np.zeros(n)
    inputs_y = np.array([n - 1 - i for i in range(n)])
    outputs_x = np.ones(n)
    outputs_y = np.array([n - 1 - i for i in range(n)])

    # Draw nodes.
    ax.scatter(inputs_x, inputs_y, s=600, c="#D0E4F5", edgecolors="#3B72A8", zorder=3)
    ax.scatter(outputs_x, outputs_y, s=600, c="#D5F5E3", edgecolors="#2D884D", zorder=3)

    # Label nodes.
    for i in range(n):
        ax.text(0, n - 1 - i, f"x{i+1}", ha="center", va="center", fontsize=12, zorder=4)
        ax.text(1, n - 1 - i, f"y{i+1}", ha="center", va="center", fontsize=12, zorder=4)

    if layer_type == "fc":
        ax.set_title("Fully Connected Layer\n(36 Unique Weights)", fontsize=14, pad=15)
        # Generate 36 distinct colors for the fully connected edges.
        colors = plt.cm.rainbow(np.linspace(0, 1, n * n))
        idx = 0
        for i in range(n):  # input
            for j in range(n):  # output
                ax.plot([0, 1], [inputs_y[i], outputs_y[j]], c=colors[idx], alpha=0.5, lw=1.5, zorder=1)
                idx += 1

    elif layer_type == "conv":
        ax.set_title("1D Convolutional Layer (Kernel Size 3)\n(3 Shared Weights)", fontsize=14, pad=15)
        # 3 distinct colors representing the shared weights in the kernel.
        kernel_colors = ["#FF5733", "#335BFF", "#FFBD33"]  # red, blue, yellow

        for j in range(n):  # output index
            # Kernel connects output j to inputs j-1, j, j+1 (assuming "same" padding).
            for k, offset in enumerate([-1, 0, 1]):
                i = j + offset  # input index
                if 0 <= i < n:  # only draw edges for valid input nodes
                    ax.plot([0, 1], [inputs_y[i], outputs_y[j]], c=kernel_colors[k], alpha=0.8, lw=2.5, zorder=2)

    # Clean up axes.
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")


# Set up the figure side-by-side.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Draw the two networks.
draw_network(ax1, "fc")
draw_network(ax2, "conv")

plt.tight_layout()
plt.savefig("1d_convolution_vs_fc.svg", bbox_inches="tight", transparent=True)
plt.show()
