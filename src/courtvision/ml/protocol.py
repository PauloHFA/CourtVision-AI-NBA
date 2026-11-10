from __future__ import annotations

from courtvision.ml.candidates import predict_positive, rows_labels, rows_matrix
from courtvision.ml.metrics import brier_score


def chronological(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["game_date"], row["game_id"]))


def later_window(rows: list[dict], fraction: float) -> tuple[list[dict], list[dict]]:
    """Fatia final da validação. Não inclui linhas do ajuste."""
    ordered = chronological(rows)
    if len(ordered) < 2:
        return ordered, []
    bounded = min(max(float(fraction), 0.1), 0.5)
    late_count = max(1, int(round(len(ordered) * bounded)))
    late_count = min(late_count, len(ordered) - 1)
    return ordered[:-late_count], ordered[-late_count:]


def expanding_windows(
    rows: list[dict],
    min_train: int = 8,
    min_score: int = 4,
) -> list[tuple[list[dict], list[dict]]]:
    ordered = chronological(rows)
    count = len(ordered)
    if count < min_train + min_score:
        return []
    windows: list[tuple[list[dict], list[dict]]] = []
    for train_ratio in (0.5, 0.75):
        train_end = int(count * train_ratio)
        if train_ratio >= 0.75:
            score_end = count
        else:
            score_end = min(count, train_end + max(min_score, int(count * 0.25)))
        train_part = ordered[:train_end]
        score_part = ordered[train_end:score_end]
        if len(train_part) < min_train or len(score_part) < min_score:
            continue
        train_ids = {row["game_id"] for row in train_part}
        score_ids = {row["game_id"] for row in score_part}
        if train_ids & score_ids:
            continue
        if max(row["game_date"] for row in train_part) >= min(row["game_date"] for row in score_part):
            continue
        windows.append((train_part, score_part))
    return windows


def seasons_of(rows: list[dict]) -> list[int]:
    return sorted({int(row["season"]) for row in rows})


def overlap_count(left: list[dict], right: list[dict]) -> int:
    left_ids = {row["game_id"] for row in left}
    return sum(1 for row in right if row["game_id"] in left_ids)


def both_classes(rows: list[dict]) -> bool:
    return {int(row["y"]) for row in rows} == {0, 1}


def walk_forward_brier(
    factory,
    rows: list[dict],
    columns: list[str],
    min_train: int = 8,
    min_score: int = 4,
) -> dict:
    windows = expanding_windows(rows, min_train=min_train, min_score=min_score)
    payload = {
        "used_for_selection": False,
        "used_validation": False,
        "seasons": seasons_of(rows),
        "folds": [],
        "mean_brier": None,
        "reason": "Amostra de treino curta para walk-forward.",
    }
    if not windows:
        return payload
    folds = []
    for index, (train_part, score_part) in enumerate(windows, start=1):
        if not both_classes(train_part) or not both_classes(score_part):
            continue
        model = factory()
        model.fit(rows_matrix(train_part, columns), rows_labels(train_part))
        probabilities = predict_positive(model, score_part, columns)
        folds.append(
            {
                "fold": index,
                "train_rows": len(train_part),
                "score_rows": len(score_part),
                "brier_score": brier_score(rows_labels(score_part), probabilities),
            }
        )
    payload["folds"] = folds
    if not folds:
        payload["reason"] = "Folds sem as duas classes. Walk-forward não calculado."
        return payload
    payload["mean_brier"] = sum(item["brier_score"] for item in folds) / len(folds)
    payload["reason"] = "Diagnóstico cronológico dentro do treino. Não promove modelo."
    return payload


def fit_calibrator(estimator, rows: list[dict], method: str, columns: list[str]):
    from sklearn.calibration import CalibratedClassifierCV

    normalized = method if method in {"sigmoid", "isotonic"} else "sigmoid"
    matrix = rows_matrix(rows, columns)
    labels = rows_labels(rows)
    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(FrozenEstimator(estimator), method=normalized)
    except ImportError:
        calibrator = CalibratedClassifierCV(estimator, method=normalized, cv="prefit")
    calibrator.fit(matrix, labels)
    return calibrator, normalized
