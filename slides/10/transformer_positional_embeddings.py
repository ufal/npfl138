#!/usr/bin/env python3
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
# import PIL

matplotlib.rcParams["mathtext.fontset"] = "cm"

def positional_embeddings(num, dim):
    data = np.arange(num)[:, np.newaxis] / (10000 ** (np.arange(dim // 2) / (dim // 2)))
    data = np.concatenate([np.sin(data), np.cos(data)], axis=-1)
    if num > 99:
        kmeans = sklearn.cluster.KMeans(n_clusters=64)
        data = kmeans.fit_predict(np.reshape(data, [-1, 1]))
        data = np.reshape(kmeans.cluster_centers_[data], [num, dim])
    plt.figure()
    plt.imshow(data, extent=(-0.5, dim - 0.5, num - 0.5, -0.5), vmin=-1, vmax=1, aspect="auto", interpolation="none")
    plt.colorbar(pad=0.025, ticks=np.linspace(-1, 1, 5))
    plt.title("Positional embeddings, {} tokens, dimension {}".format(num, dim))
    plt.xlim(-0.5, dim - 0.5)
    plt.xticks(np.arange(dim, step=dim//8))
    plt.xlabel("Embedding dimensions")
    plt.ylim(num - 0.5, -0.5)
    plt.yticks(np.arange(num, step=num//8), map("  {:2d}".format, np.arange(num, step=num//8)) if num < 100 else None)
    plt.ylabel("Token positions")
    plt.savefig("transformer_positional_embeddings_{}.svg".format(num), bbox_inches="tight", transparent=True)

positional_embeddings(16,512)
positional_embeddings(64,512)
positional_embeddings(512,512)
