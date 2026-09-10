"""
GradScope Server: Training Session
Manages the stateful training loop: forward pass, loss calculation, backward pass,
SGD update, manifold coordinate extraction, and graph serialization.
"""

from __future__ import annotations
from typing import List, Tuple
from engine.value import Value
from engine.nn import MLP
from engine.graph_export import export_graph
from datasets.toy_datasets import make_spirals, make_moons, make_xor
from server.schemas import (
    TrainingConfig,
    TrainingSnapshot,
    GraphData,
    GraphNode,
    GraphEdge,
    ManifoldPoint,
)


class TrainingSession:
    """
    Stateful training manager.
    Runs full-batch gradient descent on toy datasets using the pure autograd engine.
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.step_count = 0
        self.is_running = False
        self.loss_history: List[float] = []

        self.init_data()
        self.init_model()

    def init_data(self) -> None:
        """Regenerates the dataset based on current configuration."""
        name = self.config.dataset.lower()
        n_pts = self.config.n_points
        noise = self.config.noise

        if name == "moons":
            self.X, self.Y = make_moons(n_points=n_pts, noise=noise, seed=42)
        elif name == "xor":
            self.X, self.Y = make_xor(n_points=n_pts, noise=noise, seed=42)
        else:  # default spirals
            self.X, self.Y = make_spirals(n_points=n_pts, noise=noise, seed=42)

        # Targets for tanh output layer are in {-1.0, 1.0}
        self.targets = [-1.0 if y == 0 else 1.0 for y in self.Y]

    def init_model(self) -> None:
        """Constructs a new MLP based on current configuration."""
        # 2 inputs -> hidden layers -> 1 output
        arch = self.config.hidden_dims + [1]
        self.model = MLP(
            nin=2,
            nouts=arch,
            nonlin=self.config.activation,
            output_nonlin='tanh'
        )
        self.step_count = 0
        self.loss_history.clear()

    def update_config(self, new_config: TrainingConfig) -> bool:
        """
        Updates training configuration.
        Continuous parameters (learning_rate, step_delay_ms) apply in-flight.
        Structural parameters (dataset, noise, n_points, hidden_dims, activation) trigger a reset.
        Returns True if a reset was performed.
        """
        structural_changed = (
            new_config.dataset != self.config.dataset or
            abs(new_config.noise - self.config.noise) > 1e-4 or
            new_config.n_points != self.config.n_points or
            new_config.hidden_dims != self.config.hidden_dims or
            new_config.activation != self.config.activation
        )

        self.config = new_config

        if structural_changed:
            self.init_data()
            self.init_model()
            return True
        return False

    def reset(self) -> None:
        """Resets the model weights and training history."""
        self.init_data()
        self.init_model()

    def step(self) -> TrainingSnapshot:
        """
        Executes one full-batch step:
        1. Forward pass over all dataset samples
        2. Mean squared error loss
        3. Backward pass (autograd chain rule)
        4. SGD parameter update
        5. Build snapshot with graph and manifold activations
        """
        self.model.zero_grad()
        total_loss = Value(0.0)
        correct_count = 0
        N = len(self.X)

        # Forward pass on all samples
        for x_pt, y_tgt in zip(self.X, self.targets):
            pred, _, _ = self.model.forward_with_intermediates(x_pt)
            # MSE loss: (y_hat - y)^2
            err = pred - y_tgt
            loss_i = err * err
            total_loss = total_loss + loss_i

            # Classification accuracy: pred > 0 corresponds to class 1, <= 0 to class 0
            is_class_1 = pred.data > 0.0
            tgt_is_class_1 = y_tgt > 0.0
            if is_class_1 == tgt_is_class_1:
                correct_count += 1

        loss = total_loss * (1.0 / N)

        # Backward pass
        loss.backward()

        # SGD parameter update
        lr = self.config.learning_rate
        for p in self.model.parameters():
            p.data -= lr * p.grad

        self.step_count += 1
        loss_val = round(loss.data, 4)
        accuracy_val = round((correct_count / N) * 100.0, 1)
        self.loss_history.append(loss_val)

        # Generate snapshot
        return self.build_snapshot(loss_val, accuracy_val)

    def build_snapshot(self, loss_val: float | None = None, accuracy_val: float | None = None) -> TrainingSnapshot:
        """
        Extracts current state for frontend visualization:
        - Layer-by-layer 2D manifold coordinates for each data point
        - Computational graph for a representative sample
        """
        if loss_val is None or accuracy_val is None:
            # Quick evaluation without backward
            correct_count = 0
            total_loss = 0.0
            N = len(self.X)
            for x_pt, y_tgt in zip(self.X, self.targets):
                pred, _, _ = self.model.forward_with_intermediates(x_pt)
                err = pred.data - y_tgt
                total_loss += err * err
                if (pred.data > 0.0) == (y_tgt > 0.0):
                    correct_count += 1
            loss_val = round(total_loss / N, 4)
            accuracy_val = round((correct_count / N) * 100.0, 1)

        # 1. Collect manifold points across all layers
        manifold_points: List[ManifoldPoint] = []
        for i, (x_pt, y_label) in enumerate(zip(self.X, self.Y)):
            _, coords_2d, _ = self.model.forward_with_intermediates(x_pt)
            manifold_points.append(
                ManifoldPoint(
                    id=i,
                    label=y_label,
                    layers=[[round(c[0], 4), round(c[1], 4)] for c in coords_2d]
                )
            )

        # 2. Representative sample for graph visualization
        # Run forward + backward on sample 0 to populate node gradients for the graph
        sample_x = self.X[0]
        rep_pred, _, _ = self.model.forward_with_intermediates(sample_x)
        if hasattr(rep_pred, 'label') and not rep_pred.label:
            rep_pred.label = "Pred"

        target_val = self.targets[0]
        err = rep_pred - target_val
        err.label = "Error"
        rep_loss = err ** 2
        rep_loss.label = "Loss"

        # Run local backward on this sample's loss to illuminate the gradient flow
        rep_loss.backward()
        graph_dict = export_graph(rep_loss, max_nodes=300)

        graph_nodes = [
            GraphNode(
                id=n["id"],
                label=n["label"],
                value=n["value"],
                grad=n["grad"],
                op=n["op"],
                depth=n["depth"]
            )
            for n in graph_dict["nodes"]
        ]

        graph_edges = [
            GraphEdge(
                source=e["source"],
                target=e["target"],
                grad_flow=e["grad_flow"]
            )
            for e in graph_dict["edges"]
        ]

        # Layer labels
        layer_labels = ["Input Space (2D)"]
        for idx in range(len(self.config.hidden_dims)):
            dim = self.config.hidden_dims[idx]
            layer_labels.append(f"Hidden {idx + 1} ({dim}D)")
        layer_labels.append("Output Space (1D)")

        return TrainingSnapshot(
            type="snapshot",
            step=self.step_count,
            epoch=self.step_count,
            loss=loss_val,
            accuracy=accuracy_val,
            graph=GraphData(nodes=graph_nodes, edges=graph_edges),
            manifold=manifold_points,
            num_layers=len(layer_labels),
            layer_labels=layer_labels,
            is_running=self.is_running
        )
