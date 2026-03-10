#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows

xs = np.linspace(-20, 20, 300)
ys = np.sin(xs)
window = np.pad(scipy.signal.windows.hann(200), (50, 50))

fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
fig.set_size_inches(15, 5)
axs[0].plot(xs, 6 * ys)
axs[0].xaxis.set_ticklabels([])
axs[0].yaxis.set_ticklabels([])
axs[0].set_aspect(1)
axs[0].grid(True)
axs[0].set_title("Time Signal x")

axs[1].plot(xs, 6 * window)
axs[1].xaxis.set_ticklabels([])
axs[1].yaxis.set_ticklabels([])
axs[1].set_aspect(1)
axs[1].grid(True)
axs[1].set_title("Window Function w")
fig.savefig("windowing_signal.svg", bbox_inches="tight", transparent=True)

fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True)
fig.set_size_inches(20, 5)
for ax, o in zip(axs, [-60, 0, 60]):
    w = np.pad(window, (60, 60))[60 - o:60 - o + len(ys)]
    ax.plot(xs, 6 * ys * w)
    ax.plot(xs, 6 * w)
    ax.plot(xs, -6 * w)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_aspect(1)
    ax.grid(True)
    if o == 0: ax.set_title("Applied Window Function")
fig.savefig("windowing_applied.svg", bbox_inches="tight", transparent=True)
