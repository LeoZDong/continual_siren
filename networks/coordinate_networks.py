from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn


class SineLayer(nn.Module):
    """Linear layer with sine activation. Follows SIREN's official implementation."""

    # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class BaseCoordinateNet(nn.Module):
    """Base coordinate network that defines a template of forward pass."""

    def forward(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(
        self, coords: Tensor, retain_grad: bool = False, include_pre_nonlin: bool = True
    ) -> OrderedDict[str, Tensor]:
        """Return not only model output, but also intermediate activations. By default,
        both pre- and post-sine activations are recorded! For example,
        "<class 'networks.SineLayer'>_0" is the pre-sine activation, and
        "<class 'networks.SineLayer'>_1" is the post-sine activation.

        Args:
            coords: Model input.
            retain_grad: Whether to retain gradients for intermediate activations.
            include_pre_nonlin: Whether to include pre-nonlinearity activations.
        """
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                if include_pre_nonlin:
                    activations[
                        "_".join((str(layer.__class__), "%d" % activation_count))
                    ] = intermed
                    activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class Siren(BaseCoordinateNet):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30,
        **kwargs,
    ) -> None:
        """Initialize a SIREN network.
        Args:
            in_features: Dimension of input features.
            hidden_features: Dimension of hidden features.
            hidden_layers: Number of hidden layers.
            out_features: Dimension of output features.
            outermost_linear: Whether to use a linear layer for the outermost (output)
                layer. If False, use a sine layer instead.
            first_omega_0: omega_0 hyperparameter of the first layer.
            hidden_omega_0: omega_0 hyperparameter of the hidden layers.
        """
        super().__init__()

        self.net = []
        #### First layer ####
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        #### Hidden layers ####
        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        #### Output layer ####
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)


class ReLUNet(BaseCoordinateNet):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool,
        **kwargs,
    ) -> None:
        """Initialize a ReLU network.
        Args:
            in_features: Dimension of input features.
            hidden_features: Dimension of hidden features.
            hidden_layers: Number of hidden layers.
            out_features: Dimension of output features.
            outermost_linear: Whether to use a linear layer for the outermost (output)
                layer. If False, use a sine layer instead.
        """
        super().__init__()

        self.net = []
        #### First layer ####
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())

        #### Hidden layers ####
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())

        #### Output layer ####
        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_features))
        else:
            self.net.append(nn.Linear(hidden_features, out_features))
            self.net.append(nn.ReLU())

        self.net = nn.Sequential(*self.net)