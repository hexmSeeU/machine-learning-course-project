import numpy as np

from src.metrics import (
    accuracy,
    dual_objective,
    max_primal_violation,
    parameter_l2_difference,
    primal_objective,
    support_vector_mask,
)


def test_accuracy():
    assert accuracy(np.array([1, -1, 1]), np.array([1, 1, 1])) == 2 / 3


def test_primal_objective_uses_c_over_n():
    w = np.array([3.0, 4.0])
    zeta = np.array([1.0, 2.0, 0.0, 1.0])
    assert primal_objective(w, zeta, C=2.0) == 12.5 + 2.0


def test_dual_objective_matches_formula():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, -1.0])
    alpha = np.array([0.5, 0.5])
    assert dual_objective(X, y, alpha) == 0.75


def test_max_primal_violation_zero_when_constraints_hold():
    X = np.array([[2.0, 0.0], [-2.0, 0.0]])
    y = np.array([1.0, -1.0])
    w = np.array([1.0, 0.0])
    zeta = np.array([0.0, 0.0])
    assert max_primal_violation(X, y, w, b=0.0, zeta=zeta) == 0.0


def test_support_vector_mask():
    alpha = np.array([0.0, 1e-7, 0.2, 0.5])
    mask = support_vector_mask(alpha, tol=1e-6)
    np.testing.assert_array_equal(mask, [False, False, True, True])


def test_parameter_l2_difference():
    assert parameter_l2_difference(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == 2.0
