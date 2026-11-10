from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

XGB_GRID = (
    {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.1},
    {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05},
)


def rows_matrix(rows: list[dict], columns: list[str]) -> list[list[float]]:
    return [[float(row["features"][column]) for column in columns] for row in rows]


def rows_labels(rows: list[dict]) -> list[int]:
    return [int(row["y"]) for row in rows]


def predict_positive(model, rows: list[dict], columns: list[str]) -> list[float]:
    if not rows:
        return []
    probabilities = model.predict_proba(rows_matrix(rows, columns))[:, 1]
    return [min(max(float(probability), 0.0), 1.0) for probability in probabilities]


def build_logistic(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(random_state=seed, max_iter=500),
            ),
        ]
    )


def xgboost_importable() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


def build_xgboost(seed: int, params: dict | None = None):
    from xgboost import XGBClassifier

    chosen = {
        "n_estimators": 30,
        "max_depth": 2,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if params:
        chosen.update(params)
    chosen["random_state"] = seed
    chosen["n_jobs"] = 1
    chosen["verbosity"] = 0
    chosen["subsample"] = 1.0
    chosen["colsample_bytree"] = 1.0
    return XGBClassifier(**chosen)
