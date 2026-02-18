# Introduction

This is the documentation of the `npfl138` package used on the
[Deep Learning course NPFL138](https://ufal.mff.cuni.cz/courses/npfl138).

The main class in the `npfl138` package is the [TrainableModule][npfl138.TrainableModule],
which provides:

- high-level [training][npfl138.TrainableModule.fit], [evaluation][npfl138.TrainableModule.evaluate],
  and [prediction][npfl138.TrainableModule.predict] API, including automatic device management;
- [serialization][npfl138.TrainableModule.save_weights] and [deserialization][npfl138.TrainableModule.load_weights]
  of weights (optionally including the optimizer) and module [options][npfl138.TrainableModule.save_options].
- automatic logging via various [loggers](logger.md),
- easy [profiling support][npfl138.TrainableModule.profile].

The [TransformedDataset](transformed_dataset.md) class allows applying both
per-example and per-batch transformation functions on a given dataset, and
simplifies the creation of a corresponding dataloader (in a multi-GPU setting if
required).

Furthermore, the package contains a collection of [losses](loss.md) and [metrics](metric.md); however,
losses from PyTorch and metrics from [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)
can also be used directly.

Finally, several [utilities](utilities.md) are provided, including
a [Vocabulary](vocabulary.md) class for converting between strings and their
indices.
