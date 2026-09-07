"""
GradScope Neural Network module.
Neuron, Layer, and MLP built purely on top of the Value autograd engine.
"""

from __future__ import annotations
import random
import math
from typing import List, Union, Tuple, Optional
from engine.value import Value


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> List[Value]:
        return []


class Neuron(Module):
    """
    A single artificial neuron with weights, bias, and an activation function.
    """

    def __init__(self, nin: int, nonlin: Optional[str] = 'tanh', label_prefix: str = ''):
        # Xavier uniform initialization: bound = sqrt(6 / (nin + 1))
        bound = math.sqrt(6.0 / (nin + 1)) if nin > 0 else 1.0
        self.w = [
            Value(
                random.uniform(-bound, bound),
                _label=f"{label_prefix}_w{i}" if label_prefix else f"w{i}"
            )
            for i in range(nin)
        ]
        self.b = Value(0.0, _label=f"{label_prefix}_b" if label_prefix else "b")
        self.nonlin = nonlin
        self.label_prefix = label_prefix

    def __call__(self, x: List[Union[Value, float, int]]) -> Value:
        # Weighted sum: sum(wi * xi) + b
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + (wi * xi)

        if self.nonlin == 'tanh':
            out = act.tanh()
        elif self.nonlin == 'relu':
            out = act.relu()
        elif self.nonlin == 'sigmoid':
            out = act.sigmoid()
        elif self.nonlin is None or self.nonlin == 'linear':
            out = act
        else:
            raise ValueError(f"Unsupported activation: {self.nonlin}")

        if self.label_prefix:
            out.label = f"{self.label_prefix}_out"
        return out

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        act = self.nonlin or 'linear'
        return f"Neuron({len(self.w)}, nonlin='{act}')"


class Layer(Module):
    """
    A dense layer of independent neurons.
    """

    def __init__(self, nin: int, nout: int, nonlin: Optional[str] = 'tanh', layer_idx: int = 0):
        self.neurons = [
            Neuron(nin, nonlin=nonlin, label_prefix=f"L{layer_idx}N{i}")
            for i in range(nout)
        ]
        self.nin = nin
        self.nout = nout

        # Fixed projection matrix for layer visualization if nout > 2
        # P is of shape (nout, 2). Initialized once deterministically.
        if nout > 2:
            rng = random.Random(42 + layer_idx)
            # Generate random 2D orthogonal projection
            # Pick two random vectors and orthonormalize via Gram-Schmidt
            v1 = [rng.gauss(0, 1) for _ in range(nout)]
            norm1 = math.sqrt(sum(v ** 2 for v in v1)) or 1.0
            v1 = [v / norm1 for v in v1]

            v2 = [rng.gauss(0, 1) for _ in range(nout)]
            dot = sum(a * b for a, b in zip(v1, v2))
            v2 = [b - dot * a for a, b in zip(v1, v2)]
            norm2 = math.sqrt(sum(v ** 2 for v in v2)) or 1.0
            v2 = [v / norm2 for v in v2]

            self.proj_matrix = list(zip(v1, v2))
        else:
            self.proj_matrix = None

    def __call__(self, x: List[Union[Value, float, int]]) -> List[Value]:
        return [neuron(x) for neuron in self.neurons]

    def project_to_2d(self, activations: List[float]) -> Tuple[float, float]:
        """
        Projects this layer's activations into a 2D coordinate pair for visualization.
        - If nout == 1: maps (act[0], 0.0)
        - If nout == 2: maps (act[0], act[1]) exactly (zero distortion)
        - If nout > 2: projects using fixed orthonormal matrix P (zero jitter across steps)
        """
        if self.nout == 1:
            return (activations[0], 0.0)
        elif self.nout == 2:
            return (activations[0], activations[1])
        else:
            # P is list of (p0, p1) tuples for each neuron
            x_coord = sum(a * p[0] for a, p in zip(activations, self.proj_matrix))
            y_coord = sum(a * p[1] for a, p in zip(activations, self.proj_matrix))
            return (x_coord, y_coord)

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def __repr__(self) -> str:
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"


class MLP(Module):
    """
    Multi-Layer Perceptron composed of sequential Layers.
    """

    def __init__(
        self,
        nin: int,
        nouts: List[int],
        nonlin: Optional[str] = 'tanh',
        output_nonlin: Optional[str] = 'tanh'
    ):
        sz = [nin] + nouts
        self.layers: List[Layer] = []
        for i in range(len(nouts)):
            is_last = (i == len(nouts) - 1)
            act = output_nonlin if is_last else nonlin
            self.layers.append(Layer(sz[i], sz[i + 1], nonlin=act, layer_idx=i + 1))
        self.nin = nin
        self.nouts = nouts

    def __call__(self, x: List[Union[Value, float, int]]) -> Value | List[Value]:
        cur = [xi if isinstance(xi, Value) else Value(xi, _label=f"x{i}") for i, xi in enumerate(x)]
        for layer in self.layers:
            cur = layer(cur)
        return cur[0] if len(cur) == 1 else cur

    def forward_with_intermediates(
        self,
        x: List[float]
    ) -> Tuple[Value, List[Tuple[float, float]], List[List[float]]]:
        """
        Runs a forward pass and captures:
        1. Final prediction Value
        2. 2D projected coordinates for each layer [input_space, layer1, layer2, ..., output_space]
        3. Raw activation vectors for each layer
        """
        # Wrap inputs in Values
        cur = [Value(xi, _label=f"x{i}") for i, xi in enumerate(x)]

        # Layer 0 (input space) is directly the 2D input (x[0], x[1])
        coords_2d: List[Tuple[float, float]] = [(float(x[0]), float(x[1]))]
        raw_activations: List[List[float]] = [[float(xi) for xi in x]]

        for layer in self.layers:
            cur = layer(cur)
            raw_vals = [n.data for n in cur]
            raw_activations.append(raw_vals)
            coords_2d.append(layer.project_to_2d(raw_vals))

        pred = cur[0] if len(cur) == 1 else cur
        return pred, coords_2d, raw_activations

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self) -> str:
        return f"MLP([{', '.join(str(layer) for layer in self.layers)}])"
