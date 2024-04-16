#!/usr/bin/env python
import argparse
import os

import numpy as np
import torch
import torch.utils.tensorboard

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_embeddings", type=str, help="Embedding file to use.")
    parser.add_argument("--elements", default=20000, type=int, help="Words to export.")
    parser.add_argument("--output_dir", default="embeddings", type=str, help="Output directory.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Generate the embeddings for the projector
    with open(args.input_embeddings, "r") as embedding_file:
        _, dim = map(int, embedding_file.readline().split())

        embeddings = np.zeros([args.elements, dim], np.float32)
        words = []
        for i, line in zip(range(args.elements), embedding_file):
            word, *embedding = line.split()
            words.append(word)
            embeddings[i] = list(map(float, embedding))

    # Save the embeddings
    torch.utils.tensorboard.SummaryWriter(args.output_dir).add_embedding(
        torch.tensor(embeddings),
        metadata=words,
        tag="embeddings",
    )
