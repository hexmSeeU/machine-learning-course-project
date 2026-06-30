from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LibSVMLinearModel:
    w: np.ndarray
    b: float
    predictions: np.ndarray
    backend: str
    support_indices: np.ndarray | None = None
    dual_coefficients: np.ndarray | None = None

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.w - self.b


def _fit_with_libsvm_official(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    c_libsvm: float,
) -> LibSVMLinearModel:
    from libsvm.svm import svm_parameter, svm_problem
    from libsvm.svmutil import svm_predict, svm_train

    problem = svm_problem(y_train.tolist(), X_train.tolist())
    parameter = svm_parameter(f"-s 0 -t 0 -c {c_libsvm} -q")
    model = svm_train(problem, parameter)
    predictions, _, _ = svm_predict([0.0] * len(X_eval), X_eval.tolist(), model, "-q")

    sv_coef = np.asarray(model.get_sv_coef(), dtype=float).reshape(-1)
    sv_sparse = model.get_SV()
    sv = np.zeros((len(sv_sparse), X_train.shape[1]), dtype=float)
    for row_idx, sparse_row in enumerate(sv_sparse):
        for one_based_col, value in sparse_row.items():
            sv[row_idx, int(one_based_col) - 1] = float(value)
    support_indices = np.asarray(model.get_sv_indices(), dtype=int) - 1
    rho = float(model.rho[0])
    w = sv_coef @ sv
    b = rho
    return LibSVMLinearModel(
        w=w,
        b=b,
        predictions=np.asarray(predictions, dtype=float),
        backend="libsvm-official",
        support_indices=support_indices,
        dual_coefficients=sv_coef,
    )


def _fit_with_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    c_libsvm: float,
) -> LibSVMLinearModel:
    from sklearn.svm import SVC

    clf = SVC(C=c_libsvm, kernel="linear", shrinking=False, tol=1e-8)
    clf.fit(X_train, y_train)
    w = clf.coef_.reshape(-1)
    b = -float(clf.intercept_[0])
    predictions = clf.predict(X_eval).astype(float)
    return LibSVMLinearModel(
        w=w,
        b=b,
        predictions=predictions,
        backend="sklearn.svm.SVC-libsvm",
        support_indices=clf.support_,
        dual_coefficients=clf.dual_coef_.reshape(-1),
    )


def fit_linear_libsvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    C_assignment: float,
) -> LibSVMLinearModel:
    """Train a linear libsvm model for comparison only.

    The assignment objective uses (C / N) * sum xi_i, while libsvm uses
    C_libsvm * sum xi_i. Therefore C_libsvm = C_assignment / N.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_eval = np.asarray(X_eval, dtype=float)
    c_libsvm = float(C_assignment) / X_train.shape[0]

    try:
        return _fit_with_libsvm_official(X_train, y_train, X_eval, c_libsvm)
    except Exception as libsvm_error:
        try:
            return _fit_with_sklearn(X_train, y_train, X_eval, c_libsvm)
        except Exception as sklearn_error:
            raise RuntimeError(
                "Neither libsvm-official nor scikit-learn SVC is available for comparison. "
                "Install dependencies from P2/requirements.txt."
            ) from sklearn_error
