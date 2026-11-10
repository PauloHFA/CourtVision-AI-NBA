from courtvision.ml.metrics import accuracy, brier_score, log_loss


def test_brier_of_perfect_predictions_is_zero():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0


def test_brier_of_coin_flip_on_balanced_labels():
    assert brier_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.25


def test_accuracy_uses_strict_threshold():
    assert accuracy([1, 0], [0.5, 0.2]) == 0.5


def test_log_loss_is_finite_for_clipped_probabilities():
    value = log_loss([1, 0], [0.9, 0.1])
    assert value > 0
