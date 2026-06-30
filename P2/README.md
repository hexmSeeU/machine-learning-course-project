# P2: Hard- and Soft-Margin Linear SVM

This project implements binary linear SVMs in primal and dual QP forms and compares the result with libsvm on the WDBC dataset.

## Setup

```bash
cd P2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
cd P2
python3 -m pytest -q
```

## Run Experiments

```bash
cd P2
python scripts/run_experiments.py
```

Outputs are written to `results/tables/` and `results/figures/`.

## Report

The Chinese report draft is in `report/report.md`.

## Notes

- If `libsvm-official` is unavailable, the comparison uses `sklearn.svm.SVC`, which is backed by libsvm for `SVC`.
- The custom implementation solves explicit QP formulations with cvxpy; it does not call an existing SVM training API.
