from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def primal_objective(w: np.ndarray, zeta: np.ndarray | None, C: float) -> float:
    zeta_sum = 0.0 if zeta is None else float(np.sum(zeta))
    n = 1 if zeta is None else len(zeta)
    return float(0.5 * np.dot(w, w) + (C / n) * zeta_sum)


def dual_objective(X: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    gram_y = (y[:, None] * y[None, :]) * (X @ X.T)
    return float(np.sum(alpha) - 0.5 * alpha @ gram_y @ alpha)


def duality_gap(primal_value: float, dual_value: float) -> float:
    return float(primal_value - dual_value)


def max_primal_violation(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    zeta: np.ndarray | None = None,
) -> float:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if zeta is None:
        zeta = np.zeros(X.shape[0])
    zeta = np.asarray(zeta, dtype=float)
    margins = y * (X @ w - b)
    constraint_violation = np.maximum(0.0, 1.0 - zeta - margins)
    slack_violation = np.maximum(0.0, -zeta)
    return float(max(np.max(constraint_violation), np.max(slack_violation)))


def support_vector_mask(alpha: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    return np.asarray(alpha) > tol


def parameter_l2_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
