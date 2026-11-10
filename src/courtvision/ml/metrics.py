from __future__ import annotations

import math


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        raise ValueError("amostra vazia")
    return sum((prob - label) ** 2 for label, prob in zip(y_true, y_prob, strict=True)) / len(y_true)


def log_loss(y_true: list[int], y_prob: list[float], eps: float = 1e-15) -> float:
    if not y_true:
        raise ValueError("amostra vazia")
    total = 0.0
    for label, prob in zip(y_true, y_prob, strict=True):
        clipped = min(max(prob, eps), 1.0 - eps)
        total += -(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped))
    return total / len(y_true)


def accuracy(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> float:
    if not y_true:
        raise ValueError("amostra vazia")
    hits = 0
    for label, prob in zip(y_true, y_prob, strict=True):
        predicted = 1 if prob > threshold else 0
        hits += int(predicted == label)
    return hits / len(y_true)


def roc_auc(y_true: list[int], y_prob: list[float]) -> float | None:
    positives = [prob for label, prob in zip(y_true, y_prob, strict=True) if label == 1]
    negatives = [prob for label, prob in zip(y_true, y_prob, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def expected_calibration_error(
    y_true: list[int],
    y_prob: list[float],
    bins: int = 10,
) -> float | None:
    if not y_true:
        return None
    total = len(y_true)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = [
            (label, prob)
            for label, prob in zip(y_true, y_prob, strict=True)
            if (prob >= low and prob < high) or (index == bins - 1 and prob == 1.0)
        ]
        if not bucket:
            continue
        observed = sum(label for label, _prob in bucket) / len(bucket)
        predicted = sum(prob for _label, prob in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(observed - predicted)
    return error
