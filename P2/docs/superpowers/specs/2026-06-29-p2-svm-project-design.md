# P2 SVM Project Design

## Goal

Build a complete P2 deliverable for the SVM assignment in Chinese. The deliverable includes a Python implementation of linear hard-margin and soft-margin SVMs solved in both primal and dual forms, a comparison against libsvm, reproducible experiments on the provided WDBC dataset, focused tests, and a Chinese report draft within the assignment page limits.

## Scope

The implementation will target binary linear SVMs only. This matches the assignment requirements and makes the comparison against libsvm's linear kernel direct. The custom implementation may use an off-the-shelf quadratic programming solver, but it will not call existing SVM APIs for training. libsvm will be used only in the comparison path.

Out of scope:

- Nonlinear kernels in the custom SVM implementation.
- A hand-written SMO solver.
- Hyperparameter search beyond a small set of fixed `C` values for comparison.
- Medical interpretation of the WDBC features beyond dataset description.

## Project Layout

```text
P2/
  requirements.md
  data_a1.zip
  requirements.txt
  README.md
  src/
    __init__.py
    data.py
    svm_qp.py
    libsvm_compare.py
    metrics.py
  scripts/
    run_experiments.py
  tests/
    test_svm_qp.py
  results/
    tables/
    figures/
  report/
    report.md
```

If PDF generation is available locally, `report/report.pdf` can also be produced. The Markdown report remains the source of truth.

## Architecture

`src/data.py` will load `data_a1.zip` directly, read `data.csv`, map labels to `+1` for malignant and `-1` for benign, drop the index and ID columns, and return NumPy arrays. It will provide standardization using training-set means and standard deviations so all solvers see the same scaled feature space.

`src/svm_qp.py` will contain the custom SVM implementation. It will expose small, explicit functions or classes for:

- Soft-margin primal QP with variables `w`, `b`, and `zeta`.
- Soft-margin dual QP with variables `alpha` and reconstruction of `w` and `b`.
- Hard-margin primal QP as the no-slack version.
- Hard-margin dual QP with nonnegative `alpha` and equality constraint.
- Prediction and decision-function helpers.

The assignment uses constraints of the form `y_i(w^T x_i - b) >= 1 - zeta_i`. libsvm uses the more common decision form `w^T x + rho`, where predictions depend on the sign. The code will make the sign convention explicit and compare decision values after converting parameters into the same convention.

`src/libsvm_compare.py` will train a linear libsvm model for comparison only. If the `libsvm` Python package is available, it will be used directly. If it is not available but scikit-learn is available, `sklearn.svm.SVC(kernel="linear")` can be used as a libsvm-backed fallback and the report will state this clearly. The custom implementation will not depend on either package.

`src/metrics.py` will compute accuracy, objective values, primal feasibility violation, support-vector counts, parameter differences, and duality gap.

`scripts/run_experiments.py` will run the full experiment pipeline and write reproducible outputs under `P2/results/`.

## Data Flow

1. Read WDBC data from `P2/data_a1.zip`.
2. Convert diagnosis labels to `+1/-1`.
3. Split data into train/test with a fixed seed.
4. Standardize features using train statistics.
5. Train custom primal and dual SVMs for selected `C` values.
6. Train libsvm linear models with corresponding `C` values.
7. Convert parameters to a shared sign convention.
8. Compare `w`, `b`, support vectors, accuracy, objective values, and duality gap.
9. Write tables and any small diagnostic plots for the report.

## Experiment Design

The main experiments will use soft-margin SVM because it is the most relevant formulation for real data. Candidate `C` values will be small and interpretable, such as `0.1`, `1`, and `10`. The report will emphasize one representative setting, likely `C = 1`, for detailed parameter comparison.

Hard-margin SVM will be tested on the full standardized WDBC data only if the QP is feasible and numerically stable. Because the dataset description says the selected features are linearly separable, this is plausible. If hard-margin optimization fails due numerical conditioning, the report will show hard-margin behavior on a small synthetic separable dataset and explain the practical issue.

The assignment's soft-margin objective uses `(C / N) sum zeta_i`. libsvm uses `C_libsvm sum xi_i`. The comparison will always set `C_libsvm = C / N` for a training set of size `N`, so both solvers use equivalent penalty strength. This correspondence will be documented in code and report.

## Error Handling

Data loading will fail with a clear message if `data_a1.zip` is missing or lacks `data.csv`. QP solver failures will include solver status and basic diagnostics. libsvm comparison will be optional at import time: the experiment script will explain which package is missing rather than failing silently.

Numerical comparisons will use tolerances instead of exact equality. Support vectors will be identified with alpha and margin tolerances because QP and libsvm may return slightly different but equivalent solutions.

## Testing Strategy

Tests will focus on correctness of the custom implementation rather than the external libsvm package.

Planned tests:

- A tiny linearly separable dataset where hard-margin primal and dual produce matching predictions.
- A soft-margin dataset with one noisy point where slack variables are nonnegative and constraints are feasible within tolerance.
- Primal and dual soft-margin objectives have a small duality gap.
- Reconstructed `w` from dual alpha equals the primal `w` within tolerance on a controlled dataset.
- Data loader returns the expected feature matrix shape and binary labels.

## Report Structure

The Chinese report will be written in `P2/report/report.md` and organized to fit within 8 pages when converted to PDF:

1. Binary SVM formulation and intuition.
2. Hard-margin primal and dual forms.
3. Soft-margin primal and dual forms, including the assignment's `(C / N)` scaling.
4. Support vectors, KKT conditions, and parameter recovery.
5. Why maximum margin helps and its relation to generalisation bounds.
6. Weak duality, strong duality, and duality gap.
7. Implementation details and libsvm correspondence.
8. Experimental comparison, including tables for accuracy, `w`, `b`, `alpha`, support vectors, and duality gap.

## Success Criteria

The project is complete when:

- The custom primal and dual SVM implementations run on the provided data.
- The experiment script produces comparison tables under `P2/results/`.
- Tests for core behavior pass.
- The report draft in Chinese covers all required assignment topics.
- The libsvm comparison explains the `C` scaling and sign convention clearly.
