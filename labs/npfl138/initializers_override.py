# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import math
from typing import Any, Callable

import torch


class KerasParameterInitialization:
    def reset_parameters_linear(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def reset_parameters_bilinear(self) -> None:
        # Keras does not have a Bilinear layer. But we analogously use
        # the Xavier uniform initialization, where
        # - the fan_out for each out_feature is in_feature1 * in_feature2
        # - the fan_in for each in_feature1 is out_feature * in_feature2
        # - the fan_in for each in_feature2 is out_feature * in_feature1
        # - the overall fan_in is computed as a weighted average of the above two as
        #   (2 * out_feature * in_feature1 * in_feature2) / (in_feature1 + in_feature2)
        out, in1, in2 = self.weight.shape()
        fan_in = (2 * out * in1 * in2) / (in1 + in2)
        fan_out = in1 * in2
        bound = math.sqrt(6 / (fan_in + fan_out))
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def reset_parameters_rnn(self) -> None:
        for name, parameter in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(parameter)
            elif "bias" in name:
                torch.nn.init.zeros_(parameter)
                if isinstance(self, (torch.nn.LSTM, torch.nn.LSTMCell)):  # Set LSTM forget gate bias to 1
                    parameter.data[self.hidden_size:self.hidden_size * 2] = 1

    def reset_parameters_embedding(self) -> None:
        torch.nn.init.uniform_(self.weight, -0.05, 0.05)
        self._fill_padding_idx_with_zero()

    overrides: dict[torch.nn.Module, Callable] = {
        torch.nn.Linear: reset_parameters_linear,
        torch.nn.Conv1d: reset_parameters_linear,
        torch.nn.Conv2d: reset_parameters_linear,
        torch.nn.Conv3d: reset_parameters_linear,
        torch.nn.ConvTranspose1d: reset_parameters_linear,
        torch.nn.ConvTranspose2d: reset_parameters_linear,
        torch.nn.ConvTranspose3d: reset_parameters_linear,
        torch.nn.Bilinear: reset_parameters_bilinear,
        torch.nn.RNN: reset_parameters_rnn,
        torch.nn.RNNCell: reset_parameters_rnn,
        torch.nn.LSTM: reset_parameters_rnn,
        torch.nn.LSTMCell: reset_parameters_rnn,
        torch.nn.GRU: reset_parameters_rnn,
        torch.nn.GRUCell: reset_parameters_rnn,
        torch.nn.Embedding: reset_parameters_embedding,
        torch.nn.EmbeddingBag: reset_parameters_embedding,
    }


class KerasBatchNormMomentum:
    @staticmethod
    def override_default_argument_value(func: Callable, name: str, default: Any) -> None:
        default_names = func.__code__.co_varnames[:func.__code__.co_argcount][-len(func.__defaults__):]
        assert name in default_names, f"Argument {name} not found in {func.__name__} arguments"
        func.__defaults__ = tuple(
            default if arg_name == name else arg_value
            for arg_name, arg_value in zip(default_names, func.__defaults__)
        )

    batch_norms = [
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.LazyBatchNorm1d,
        torch.nn.LazyBatchNorm2d,
        torch.nn.LazyBatchNorm3d,
        torch.nn.SyncBatchNorm,
    ]


def global_keras_initializers(
    parameter_initialization: bool = True,
    batchnorm_momentum_override: bool = True,
) -> None:
    """Change default PyTorch initializers to Keras defaults.

    The following initializers are used:

    - `Linear`, `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`, `Bilinear`:
      Xavier uniform for weights, zeros for biases.
    - `Embedding`, `EmbeddingBag`: Uniform [-0.05, 0.05] for weights.
    - `RNN`, `RNNCell`, `LSTM`, `LSTMCell`, `GRU`, `GRUCell`: Xavier uniform for input weights,
      orthogonal for recurrent weights, zeros for biases (with LSTM forget gate bias set to 1).

    Furthermore, for batch normalization layers, the default momentum value is changed from 0.1 to 0.01.

    Parameters:
     parameter_initialization: If True, override the default PyTorch initializers with Keras defaults.
     batchnorm_momentum_override: If True, override the default momentum value of 0.1 in
       batch normalization layers to 0.01.
    """
    if parameter_initialization:
        for class_, reset_parameters_method in KerasParameterInitialization.overrides.items():
            class_.reset_parameters = reset_parameters_method

    if batchnorm_momentum_override:
        for batch_norm_super in KerasBatchNormMomentum.batch_norms:
            for batch_norm in [batch_norm_super] + batch_norm_super.__subclasses__():
                KerasBatchNormMomentum.override_default_argument_value(batch_norm.__init__, "momentum", 0.01)
