"""
Tests for GradScope engine.value
Verifies reverse-mode automatic differentiation against numerical derivatives.
"""

import unittest
import math
from engine.value import Value


class TestValueAutograd(unittest.TestCase):
    def test_addition(self):
        a = Value(3.0)
        b = Value(-4.0)
        c = a + b
        c.backward()
        self.assertEqual(c.data, -1.0)
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

    def test_multiplication(self):
        a = Value(2.5)
        b = Value(-3.0)
        c = a * b
        c.backward()
        self.assertEqual(c.data, -7.5)
        self.assertEqual(a.grad, -3.0)
        self.assertEqual(b.grad, 2.5)

    def test_variable_reuse(self):
        # f(x) = x * x + x
        # f'(x) = 2x + 1
        x = Value(3.0)
        y = x * x + x
        y.backward()
        self.assertEqual(y.data, 12.0)
        self.assertEqual(x.grad, 7.0)

    def test_subtraction_and_division(self):
        a = Value(8.0)
        b = Value(2.0)
        c = (a - b) / b  # (8 - 2) / 2 = 6 / 2 = 3.0
        c.backward()
        # c = a/b - 1 -> dc/da = 1/b = 0.5, dc/db = -a / (b^2) = -8 / 4 = -2.0
        self.assertEqual(c.data, 3.0)
        self.assertAlmostEqual(a.grad, 0.5, places=5)
        self.assertAlmostEqual(b.grad, -2.0, places=5)

    def test_power(self):
        x = Value(3.0)
        y = x ** 3
        y.backward()
        # dy/dx = 3 * x^2 = 27
        self.assertEqual(y.data, 27.0)
        self.assertEqual(x.grad, 27.0)

    def test_tanh_numerical_gradient(self):
        # f(x) = tanh(x)
        # f'(x) = 1 - tanh(x)^2
        h = 1e-6
        x_val = 0.75

        # Numerical gradient
        f_plus = math.tanh(x_val + h)
        f_minus = math.tanh(x_val - h)
        num_grad = (f_plus - f_minus) / (2 * h)

        # Autograd
        x = Value(x_val)
        y = x.tanh()
        y.backward()

        self.assertAlmostEqual(x.grad, num_grad, places=5)
        self.assertAlmostEqual(x.grad, 1.0 - math.tanh(x_val)**2, places=5)

    def test_relu_numerical_gradient(self):
        x_pos = Value(2.5)
        y_pos = x_pos.relu()
        y_pos.backward()
        self.assertEqual(x_pos.grad, 1.0)

        x_neg = Value(-2.5)
        y_neg = x_neg.relu()
        y_neg.backward()
        self.assertEqual(x_neg.grad, 0.0)

    def test_exp_and_log(self):
        x = Value(1.5)
        y = x.exp()
        y.backward()
        self.assertAlmostEqual(y.data, math.exp(1.5), places=5)
        self.assertAlmostEqual(x.grad, math.exp(1.5), places=5)

        x2 = Value(2.5)
        y2 = x2.log()
        y2.backward()
        self.assertAlmostEqual(y2.data, math.log(2.5), places=5)
        self.assertAlmostEqual(x2.grad, 1.0 / 2.5, places=5)

    def test_complex_dag_against_numerical(self):
        """
        f(x, y) = (x * y + tanh(x)) / (y**2 + 1)
        Test all partial derivatives against central finite differences.
        """
        h = 1e-6

        def f(xv, yv):
            return (xv * yv + math.tanh(xv)) / (yv**2 + 1.0)

        x0, y0 = 1.2, -0.8

        df_dx_num = (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h)
        df_dy_num = (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)

        x = Value(x0)
        y = Value(y0)
        out = (x * y + x.tanh()) / (y**2 + 1.0)
        out.backward()

        self.assertAlmostEqual(x.grad, df_dx_num, places=4)
        self.assertAlmostEqual(y.grad, df_dy_num, places=4)


if __name__ == '__main__':
    unittest.main()
