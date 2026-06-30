import numpy as np

from src.svm_qp import (
    fit_hard_margin_dual,
    fit_hard_margin_primal,
    fit_soft_margin_dual,
    fit_soft_margin_primal,
)


def separable_data():
    X = np.array([[-2.0, -2.0], [-2.0, -1.0], [2.0, 1.0], [2.0, 2.0]])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    return X, y


def noisy_data():
    X = np.array([[-2.0, -2.0], [-2.0, -1.0], [0.2, 0.0], [2.0, 1.0], [2.0, 2.0]])
    y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0])
    return X, y


def test_hard_margin_primal_and_dual_predict_training_data():
    X, y = separable_data()

    primal = fit_hard_margin_primal(X, y)
    dual = fit_hard_margin_dual(X, y)

    np.testing.assert_array_equal(primal.predict(X), y)
    np.testing.assert_array_equal(dual.predict(X), y)
    np.testing.assert_allclose(primal.w, dual.w, atol=1e-4)
    assert abs(primal.b - dual.b) < 1e-4


def test_soft_margin_primal_constraints_and_predictions():
    X, y = noisy_data()

    model = fit_soft_margin_primal(X, y, C=1.0)
    margins = y * (X @ model.w - model.b)

    assert model.zeta is not None
    assert np.all(model.zeta >= -1e-6)
    assert np.all(margins >= 1.0 - model.zeta - 1e-5)
    assert np.mean(model.predict(X) == y) >= 0.8


def test_soft_margin_dual_matches_primal_weight_and_predictions():
    X, y = noisy_data()

    primal = fit_soft_margin_primal(X, y, C=2.0)
    dual = fit_soft_margin_dual(X, y, C=2.0)

    np.testing.assert_allclose(primal.w, dual.w, atol=1e-4)
    np.testing.assert_array_equal(primal.predict(X), dual.predict(X))
    assert dual.alpha is not None
    assert np.all(dual.alpha >= -1e-6)
    assert np.all(dual.alpha <= 2.0 / len(y) + 1e-5)
    assert abs(primal.objective_value - dual.objective_value) < 1e-5
