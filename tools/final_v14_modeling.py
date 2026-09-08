"""Locked shallow-model helpers for FINAL-v14; no data is loaded at import."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error

PENALTIES = tuple(10.0**p for p in range(-4, 5))


def _arrays(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.shape != (len(X),) or not X.shape[1]:
        raise ValueError("nonempty two-dimensional features and aligned targets required")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite model input")
    return X, y


def _fit(X: Any, y: Any, penalty: float, kind: str) -> dict:
    X, y = _arrays(X, y)
    if kind == "logistic" and set(np.unique(y)) != {0.0, 1.0}:
        return {"status": "NOT_ESTIMABLE", "reason": "training roster lacks both binary classes"}
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    constant = scale == 0
    scale[constant] = 1.0
    Z = (X - mean) / scale
    if kind == "logistic":
        model = LogisticRegression(C=float(penalty), penalty="l2", solver="lbfgs",
                                   fit_intercept=True, class_weight=None,
                                   max_iter=10000, tol=1e-8)
    else:
        model = Ridge(alpha=float(penalty), fit_intercept=True, solver="svd")
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(Z, y)
        if any(issubclass(w.category, ConvergenceWarning) for w in recorded):
            return {"status": "NOT_ESTIMABLE", "reason": "nonconvergence"}
        coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        coef[constant] = 0.0
        intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
        prediction = Z @ coef + intercept
        if not (np.isfinite(coef).all() and np.isfinite(intercept) and np.isfinite(prediction).all()):
            return {"status": "NOT_ESTIMABLE", "reason": "nonfinite fitted parameters or predictions"}
        return {
            "status": "ESTIMABLE", "selected_penalty": float(penalty),
            "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
            "coef": coef.tolist(), "intercept": intercept,
            "zero_variance_coordinates": np.flatnonzero(constant).tolist(),
            "predictions": prediction.tolist(),
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return {"status": "NOT_ESTIMABLE", "reason": str(error)}


def fit_logistic(X: Any, y: Any, C: float) -> dict:
    return _fit(X, y, C, "logistic")


def predict(model: dict, X: Any) -> np.ndarray:
    if model["status"] != "ESTIMABLE":
        raise ValueError("cannot score a non-estimable model")
    X = np.asarray(X, dtype=np.float64)
    result = ((X - np.asarray(model["scaler_mean"])) / np.asarray(model["scaler_scale"])) @ np.asarray(model["coef"]) + model["intercept"]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite predictions")
    return result


def nested_fit(X_train: Any, y_train: Any, X_test: Any,
               inner_splits: Sequence[tuple[Any, Any]], kind: str) -> dict:
    X, y = _arrays(X_train, y_train)
    X_test = np.asarray(X_test, dtype=np.float64)
    if X_test.ndim != 2 or X_test.shape[1] != X.shape[1] or not np.isfinite(X_test).all():
        raise ValueError("invalid held-out feature matrix")
    splits = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in inner_splits]
    if len(splits) < 2:
        return {"status": "NOT_ESTIMABLE", "reason": "fewer than two inner folds", "candidates": []}
    seen = np.zeros(len(X), dtype=int)
    all_indices = set(range(len(X)))
    for train, val in splits:
        if (not len(train) or not len(val) or set(train) & set(val)
                or set(train) | set(val) != all_indices
                or len(set(train)) != len(train) or len(set(val)) != len(val)):
            raise ValueError("inner split must partition the complete outer-training roster")
        seen[val] += 1
    if not np.all(seen == 1):
        raise ValueError("inner validation folds must cover each training patient once")
    candidates = []
    for penalty in PENALTIES:
        losses = []
        failures = []
        for fold, (train, val) in enumerate(splits):
            fitted = _fit(X[train], y[train], penalty, kind)
            if fitted["status"] != "ESTIMABLE":
                losses.append(None)
                failures.append({"fold": fold, "reason": fitted["reason"]})
                continue
            try:
                scores = predict(fitted, X[val])
                loss = (log_loss(y[val], expit(scores), labels=[0, 1])
                        if kind == "logistic" else mean_squared_error(y[val], scores))
                if not np.isfinite(loss):
                    raise ValueError("nonfinite validation loss")
                losses.append(float(loss))
            except ValueError as error:
                losses.append(None)
                failures.append({"fold": fold, "reason": str(error)})
        candidates.append({
            "penalty": penalty, "eligible": not failures,
            "inner_losses": losses, "failures": failures,
            "mean_loss": None if failures else float(np.mean(losses)),
        })
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        return {"status": "NOT_ESTIMABLE", "reason": "no eligible penalty", "candidates": candidates}
    best_loss = min(row["mean_loss"] for row in eligible)
    ties = [row for row in eligible if abs(row["mean_loss"] - best_loss) <= 1e-12]
    chosen = (min if kind == "logistic" else max)(row["penalty"] for row in ties)
    fitted = _fit(X, y, chosen, kind)
    fitted["selected_penalty"] = chosen
    fitted["candidates"] = candidates
    if fitted["status"] == "ESTIMABLE":
        try:
            fitted["predictions"] = predict(fitted, X_test).tolist()
        except ValueError as error:
            fitted = {"status": "NOT_ESTIMABLE", "reason": str(error),
                      "selected_penalty": chosen, "candidates": candidates}
    return fitted


def nested_logistic(X_train: Any, y_train: Any, X_test: Any,
                    inner_splits: Sequence[tuple[Any, Any]]) -> dict:
    return nested_fit(X_train, y_train, X_test, inner_splits, "logistic")


def nested_ridge(X_train: Any, y_train: Any, X_test: Any,
                 inner_splits: Sequence[tuple[Any, Any]]) -> dict:
    return nested_fit(X_train, y_train, X_test, inner_splits, "ridge")
