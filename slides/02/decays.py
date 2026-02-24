#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

xs = np.linspace(0, 1000, 200)

for schedule, name in [
        (tf.optimizers.schedules.PolynomialDecay(0.1, decay_steps=1000, end_learning_rate=0.001, power=1), "linear"),
        (tf.optimizers.schedules.ExponentialDecay(0.1, decay_steps=1000, decay_rate=0.001/0.1), "exponential"),
        (tf.optimizers.schedules.CosineDecay(0.1, decay_steps=1000, alpha=0.001/0.1), "cosine"),
]:
    plt.figure(figsize=(8,4))
    plt.plot(xs, [schedule(x) for x in xs])
    plt.xlabel("steps")
    plt.ylabel("learning rate")
    plt.grid(True)
    plt.title("{} decay rate".format(name.title()))
    plt.savefig("decay_{}.svg".format(name), transparent=True, bbox_inches="tight")
