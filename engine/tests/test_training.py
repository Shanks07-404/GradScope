"""
Integration test: Train a tiny MLP on XOR dataset using the pure autograd engine.
"""

import unittest
from engine.nn import MLP
from datasets.toy_datasets import make_xor


class TestMLPTraining(unittest.TestCase):
    def test_xor_convergence(self):
        # 2D input -> [4, 4] hidden -> 1 output
        # Using tanh activations
        model = MLP(2, [4, 4, 1], nonlin='tanh', output_nonlin='tanh')
        X, Y = make_xor(n_points=60, noise=0.05, seed=42)

        # Convert 0/1 labels to -1 / 1 for tanh output
        y_targets = [-1.0 if y == 0 else 1.0 for y in Y]

        initial_loss = 0.0
        final_loss = 0.0

        # Run 40 full-batch steps of gradient descent
        for step in range(40):
            # Forward pass
            total_loss = 0.0
            correct = 0

            # Zero grad
            model.zero_grad()

            for x_pt, y_tgt in zip(X, y_targets):
                pred, _, _ = model.forward_with_intermediates(x_pt)
                # Max-margin hinge loss: max(0, 1 - y_tgt * pred)
                # or MSE loss: (pred - y_tgt) ** 2
                loss_i = (pred - y_tgt) ** 2
                total_loss = total_loss + loss_i

                if (pred.data > 0 and y_tgt > 0) or (pred.data <= 0 and y_tgt <= 0):
                    correct += 1

            loss = total_loss * (1.0 / len(X))

            if step == 0:
                initial_loss = loss.data

            # Backward pass
            loss.backward()

            # SGD update
            lr = 0.1
            for p in model.parameters():
                p.data -= lr * p.grad

            if step == 39:
                final_loss = loss.data
                accuracy = correct / len(X)

        print(f"\nXOR Training test: initial_loss={initial_loss:.4f} -> final_loss={final_loss:.4f}, acc={accuracy * 100:.1f}%")
        self.assertLess(final_loss, initial_loss)
        self.assertGreater(accuracy, 0.85)


if __name__ == '__main__':
    unittest.main()
