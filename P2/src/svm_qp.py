from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class LinearSVMModel:
    w: np.ndarray
    b: float
    alpha: np.ndarray | None = None
    zeta: np.ndarray | None = None
    objective_value: float | None = None
    solver_status: str | None = None

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.w - self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1.0, -1.0)


def _validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D array with one label per sample")
    labels = set(np.unique(y))
    if labels - {-1.0, 1.0}:
        raise ValueError("y labels must be -1 and +1")
    return X, y


def _solve_problem(problem: cp.Problem) -> None:
    installed = set(cp.installed_solvers())
    for solver in ("CLARABEL", "OSQP", "SCS"):
        if solver not in installed:
            continue
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.SolverError:
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return
    raise RuntimeError(f"QP solve failed with status {problem.status}")


def _recover_b_from_support_vectors(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    alpha: np.ndarray,
    upper: float | None,
) -> float:
    tol = 1e-5
    if upper is None:
        mask = alpha > tol
    else:
        mask = (alpha > tol) & (alpha < upper - tol)
    if not np.any(mask):
        mask = alpha > tol
    if not np.any(mask):
        raise RuntimeError("Cannot recover b because no support vectors were found")
    return float(np.mean(X[mask] @ w - y[mask]))


def fit_hard_margin_primal(X: np.ndarray, y: np.ndarray) -> LinearSVMModel:
    X, y = _validate_xy(X, y)
    n_features = X.shape[1]

    w = cp.Variable(n_features)
    b = cp.Variable()
    constraints = [cp.multiply(y, X @ w - b) >= 1]
    objective = cp.Minimize(0.5 * cp.sum_squares(w))
    problem = cp.Problem(objective, constraints)
    _solve_problem(problem)

    return LinearSVMModel(
        w=np.asarray(w.value, dtype=float).reshape(-1),
        b=float(b.value),
        objective_value=float(problem.value),
        solver_status=problem.status,
    )


def fit_soft_margin_primal(X: np.ndarray, y: np.ndarray, C: float) -> LinearSVMModel:
    X, y = _validate_xy(X, y)
    if C <= 0:
        raise ValueError("C must be positive")
    n_samples, n_features = X.shape
    penalty = C / n_samples

    w = cp.Variable(n_features)
    b = cp.Variable()
    zeta = cp.Variable(n_samples)
    constraints = [cp.multiply(y, X @ w - b) >= 1 - zeta, zeta >= 0]
    objective = cp.Minimize(0.5 * cp.sum_squares(w) + penalty * cp.sum(zeta))
    problem = cp.Problem(objective, constraints)
    _solve_problem(problem)

    return LinearSVMModel(
        w=np.asarray(w.value, dtype=float).reshape(-1),
        b=float(b.value),
        zeta=np.maximum(np.asarray(zeta.value, dtype=float).reshape(-1), 0.0),
        objective_value=float(problem.value),
        solver_status=problem.status,
    )


def fit_hard_margin_dual(X: np.ndarray, y: np.ndarray) -> LinearSVMModel:
    X, y = _validate_xy(X, y)
    n_samples = X.shape[0]
    gram_y = (y[:, None] * y[None, :]) * (X @ X.T)
    gram_y = 0.5 * (gram_y + gram_y.T)

    alpha = cp.Variable(n_samples)
    constraints = [alpha >= 0, y @ alpha == 0]
    objective = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(alpha, cp.psd_wrap(gram_y)))
    problem = cp.Problem(objective, constraints)
    _solve_problem(problem)

    alpha_value = np.maximum(np.asarray(alpha.value, dtype=float).reshape(-1), 0.0)
    w = (alpha_value * y) @ X
    b = _recover_b_from_support_vectors(X, y, w, alpha_value, upper=None)
    return LinearSVMModel(
        w=w,
        b=b,
        alpha=alpha_value,
        objective_value=float(problem.value),
        solver_status=problem.status,
    )


def fit_soft_margin_dual(X: np.ndarray, y: np.ndarray, C: float) -> LinearSVMModel:
    X, y = _validate_xy(X, y)
    if C <= 0:
        raise ValueError("C must be positive")
    n_samples = X.shape[0]
    upper = C / n_samples
    gram_y = (y[:, None] * y[None, :]) * (X @ X.T)
    gram_y = 0.5 * (gram_y + gram_y.T)

    alpha = cp.Variable(n_samples)
    constraints = [alpha >= 0, alpha <= upper, y @ alpha == 0]
    objective = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(alpha, cp.psd_wrap(gram_y)))
    problem = cp.Problem(objective, constraints)
    _solve_problem(problem)

    alpha_value = np.clip(np.asarray(alpha.value, dtype=float).reshape(-1), 0.0, upper)
    w = (alpha_value * y) @ X
    b = _recover_b_from_support_vectors(X, y, w, alpha_value, upper=upper)
    return LinearSVMModel(
        w=w,
        b=b,
        alpha=alpha_value,
        objective_value=float(problem.value),
        solver_status=problem.status,
    )
