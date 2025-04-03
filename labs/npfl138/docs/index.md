# Introduction

This is the documentation of the `npfl138` package used on the
[Deep Learning course NPFL138](https://ufal.mff.cuni.cz/courses/npfl138).

The main class in the `npfl138` package is the [TrainableModule][npfl138.TrainableModule],
which provides:

- high-level training, evaluation, and prediction API, including automatic device management;
- serialization and deserialization of weights (optionally including the optimizer) and configuration;
- automatic logging.

The [TransformedDataset][npfl138.TransformedDataset] class allows applying both
per-example and per-batch transformation functions on a given dataset, and
simplifies the creation of a corresponding dataloader.

Furthermore, the package contains a large collection of datasets.

Finally, several [utilities](utilities.md) are provided.

::: npfl138.__version__
    options:
      heading: "The npfl138 version"
      heading_level: 3
      show_labels: false
      show_symbol_type_toc: false
      show_symbol_type_heading: false
