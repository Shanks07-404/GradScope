"""
GradScope Core Engine: Value class
A minimal, dependency-free scalar autograd engine built purely with Python 3.11+.
"""

from __future__ import annotations
import math
from typing import Set, Tuple, Union


class Value:
    """
    Stores a single scalar value and its gradient computed via reverse-mode autodiff.
    """

    __slots__ = ('data', 'grad', '_backward', '_prev', '_op', '_label')

    def __init__(
        self,
        data: float | int,
        _children: Tuple[Value, ...] = (),
        _op: str = '',
        _label: str = '',
    ):
        self.data = float(data)
        self.grad = 0.0
        # Internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # The operation that produced this node (e.g. '+', '*', 'tanh')
        self._label = _label  # Optional label for graph visualization

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, val: str) -> None:
        self._label = val

    def __repr__(self) -> str:
        label_str = f", label='{self._label}'" if self._label else ""
        op_str = f", op='{self._op}'" if self._op else ""
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}{op_str}{label_str})"

    def __add__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Union[Value, float, int]) -> Value:
        return self.__add__(other)

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Union[Value, float, int]) -> Value:
        return self + (-other)

    def __rsub__(self, other: Union[Value, float, int]) -> Value:
        return Value(other) + (-self)

    def __mul__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[Value, float, int]) -> Value:
        return self.__mul__(other)

    def __pow__(self, other: Union[float, int]) -> Value:
        assert isinstance(other, (int, float)), "Power exponent must be int or float"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: Union[Value, float, int]) -> Value:
        return self * (other ** -1 if isinstance(other, Value) else (other ** -1))

    def __rtruediv__(self, other: Union[Value, float, int]) -> Value:
        return Value(other) * (self ** -1)

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        # Clamp to prevent overflow in extreme values
        clamped_data = max(min(self.data, 20.0), -20.0)
        t = math.tanh(clamped_data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        clamped_data = min(self.data, 80.0)
        val = math.exp(clamped_data)
        out = Value(val, (self,), 'exp')

        def _backward():
            self.grad += val * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        val = math.log(max(self.data, 1e-12))
        out = Value(val, (self,), 'log')

        def _backward():
            self.grad += (1.0 / max(self.data, 1e-12)) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        clamped = max(min(self.data, 20.0), -20.0)
        s = 1.0 / (1.0 + math.exp(-clamped))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Executes reverse-mode automatic differentiation starting from this node.
        Computes gradients for all ancestor nodes in the computational DAG.
        """
        topo: list[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
