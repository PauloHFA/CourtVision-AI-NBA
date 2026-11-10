from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml

from courtvision.data.records import GameRecord, dataset_hash
from courtvision.data.repository import MemoryRepository
from courtvision.features.builder import build_features
from courtvision.features.schema import FEATURE_COLUMNS, FEATURE_VERSION
from courtvision.ml.baselines import HomeRateBaseline, PredictContext
from courtvision.ml.candidates import (
    XGB_GRID,
    build_logistic,
    build_xgboost,
    predict_positive,
    rows_labels,
    rows_matrix,
    xgboost_importable,
)
from courtvision.ml.metrics import accuracy, brier_score, expected_calibration_error, log_loss, roc_auc
from courtvision.ml.protocol import (
    both_classes,
    fit_calibrator,
    later_window,
    overlap_count,
    seasons_of,
    walk_forward_brier,
)
from courtvision.ml.reliability import reliability_bins, reliability_svg


def load_games(fixture_path: str) -> list[GameRecord]:
    payload = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    return MemoryRepository.from_dataset(payload).all_game_records()


def supervised_rows(
    games: list[GameRecord],
    seasons: set[int],
    min_prior_games: int,
) -> list[dict]:
    rows = []
    for game in games:
        if game.season not in seasons or game.status != "final" or game.postseason:
            continue
        if game.home_score is None or game.visitor_score is None:
            continue
        moment = datetime(game.game_date.year, game.game_date.month, game.game_date.day, tzinfo=timezone.utc)
        features = build_features(games, game.game_id, moment)
        if float(features["min_prior_games"]) < min_prior_games:
            continue
        rows.append(
            {
                "features": features,
                "y": 1 if game.home_score > game.visitor_score else 0,
                "season": game.season,
                "game_id": game.game_id,
                "game_date": game.game_date,
            }
        )
    return rows


def _metric_bundle(y_true: list[int], y_prob: list[float]) -> dict:
    return {
        "brier_score": brier_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "accuracy": accuracy(y_true, y_prob),
        "roc_auc": roc_auc(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "n": len(y_true),
    }


def _home_rate_probs(rows: list[dict], games: list[GameRecord]) -> list[float]:
    model = HomeRateBaseline()
    probabilities = []
    for row in rows:
        context = PredictContext(games=games, cutoff=row["game_date"])
        probabilities.append(model.predict_home(row["features"], context))
    return probabilities


def _skipped(reason: str) -> dict:
    return {"status": "skipped", "reason": reason}


def _beats(score: float, baselines: list[float], margin: float) -> bool:
    return all(score < baseline - margin for baseline in baselines)


def experiment_markdown(report: dict) -> str:
    lines = [
        "# Experimento",
        "",
        "`performance_verified` permanece `false`. Os números abaixo são desta execução, não metas da especificação.",
        "",
        f"- Promovido: {str(report['promoted']).lower()}",
        f"- Motivo: {report['reason']}",
        f"- Algoritmo selecionado para o artefato: {report.get('selected_algorithm') or 'nenhum'}",
        f"- Temporadas de ajuste: {report['fit_seasons']}",
        f"- Validação: {report['validation_seasons']}",
        f"- Teste usado na seleção: {str(report['test_used_for_selection']).lower()}",
        "",
        "## Métricas de validação, sem calibrador",
        "",
    ]
    for name, metrics in report.get("metrics", {}).items():
        if "brier_score" not in metrics:
            lines.append(f"- {name}: {metrics.get('reason', metrics.get('status', 'sem métrica'))}")
            continue
        lines.append(
            f"- {name}: Brier {metrics['brier_score']:.4f}, log loss {metrics['log_loss']:.4f}, "
            f"accuracy {metrics['accuracy']:.3f}, n={metrics['n']}"
        )
    calibration = report.get("calibration", {})
    reliability = report.get("reliability", {})
    lines.extend(
        [
            "",
            "## Protocolo",
            "",
            "A temporada de teste não entra na promoção. "
            "A calibração, quando aplicada, usa só a fatia final da validação e também não promove o modelo.",
            f"- Calibração aplicada: {str(calibration.get('applied', False)).lower()}",
            f"- Motivo da calibração: {calibration.get('reason', '')}",
        ]
    )
    if reliability.get("note"):
        lines.append(f"- Confiabilidade: {reliability['note']}")
    lines.append("")
    return "\n".join(lines)


def _write_report(config: dict, report: dict, output_dir: Path | None) -> None:
    if output_dir is not None:
        json_path = output_dir / "experiment_latest.json"
        markdown_path = output_dir / "experiment_latest.md"
        svg_path = output_dir / "reliability_validation.svg"
    else:
        json_path = Path(config.get("report_path", "reports/experiment_latest.json"))
        markdown_path = json_path.with_suffix(".md")
        svg_path = json_path.parent / "reliability_validation.svg"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(experiment_markdown(report), encoding="utf-8")
    chart = report.get("reliability", {}).get("svg")
    if chart:
        svg_path.write_text(chart, encoding="utf-8")
        report["reliability"]["path"] = str(svg_path)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _empty_calibration(method: str) -> dict:
    return {
        "method": method,
        "applied": False,
        "used_for_selection": False,
        "disjoint_from_fit": True,
        "seasons": [],
        "rows": 0,
        "overlap_with_fit": 0,
        "reason": "Calibração desligada." if method in {"", "none"} else "Calibração ainda não aplicada.",
    }


def train_from_config(config: dict, output_dir: Path | None = None) -> dict:
    games = load_games(config["fixture_path"])
    min_prior = int(config.get("min_prior_games", 3))
    train_seasons = set(config["train_seasons"])
    valid_seasons = set(config["validation_seasons"])
    test_seasons = set(config.get("test_seasons") or [])
    if train_seasons & test_seasons:
        raise ValueError("Temporadas de teste não podem entrar no ajuste.")
    if train_seasons & valid_seasons:
        raise ValueError("Validação não pode repetir temporada de ajuste.")
    if valid_seasons & test_seasons:
        raise ValueError("Teste não pode repetir temporada de validação.")
    seed = int(config.get("random_seed", 42))
    calibration_method = str(config.get("calibration_method") or "none")
    train_rows = supervised_rows(games, train_seasons, min_prior)
    valid_rows = supervised_rows(games, valid_seasons, min_prior)
    report: dict = {
        "model_version": config.get("model_version", "ml-0.1.0"),
        "feature_version": config.get("feature_version", FEATURE_VERSION),
        "random_seed": seed,
        "primary_metric": config.get("primary_metric", "brier_score"),
        "material_brier_gain": float(config.get("material_brier_gain", 0.0)),
        "material_gain_status": "proposed_gate_not_a_measured_result",
        "fit_seasons": sorted(train_seasons),
        "validation_seasons": sorted(valid_seasons),
        "test_seasons": sorted(test_seasons),
        "test_used_for_selection": False,
        "performance_verified": False,
        "performance_note": "Promoção usa só a validação. Não é avaliação externa nem resultado verificado.",
        "dataset_hash": dataset_hash(games),
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
        "promoted": False,
        "selected_algorithm": None,
        "reason": "",
        "metrics": {},
        "calibration": _empty_calibration(calibration_method),
        "walk_forward": {"used_for_selection": False, "logistic": None, "xgboost_search": None},
        "reliability": {"bins": [], "note": "Validação interna, sem calibrador. Não é resultado externo."},
    }
    if len(train_rows) < 8 or len(valid_rows) < 4:
        report["reason"] = "Amostra insuficiente para ajustar e comparar modelos."
        _write_report(config, report, output_dir)
        return report

    y_valid = rows_labels(valid_rows)
    report["metrics"]["constant_half"] = _metric_bundle(y_valid, [0.5 for _ in valid_rows])
    report["metrics"]["home_rate"] = _metric_bundle(y_valid, _home_rate_probs(valid_rows, games))

    fitted: dict[str, object] = {}
    if both_classes(train_rows):
        logistic = build_logistic(seed)
        logistic.fit(rows_matrix(train_rows, FEATURE_COLUMNS), rows_labels(train_rows))
        fitted["logistic"] = logistic
        report["metrics"]["logistic"] = _metric_bundle(
            y_valid,
            predict_positive(logistic, valid_rows, FEATURE_COLUMNS),
        )
        report["walk_forward"]["logistic"] = walk_forward_brier(
            lambda: build_logistic(seed),
            train_rows,
            FEATURE_COLUMNS,
        )
    else:
        report["metrics"]["logistic"] = _skipped("Treino sem as duas classes.")
        report["walk_forward"]["logistic"] = _skipped("Treino sem as duas classes.")

    enable_xgboost = bool(config.get("enable_xgboost", True))
    if not enable_xgboost:
        report["metrics"]["xgboost"] = _skipped("XGBoost desligado na configuração.")
        report["walk_forward"]["xgboost_search"] = {
            "used_for_selection": False,
            "used_validation": False,
            "chosen": None,
            "candidates": [],
            "reason": "XGBoost desligado na configuração.",
        }
    elif not xgboost_importable():
        report["metrics"]["xgboost"] = _skipped("Dependência opcional xgboost não instalada.")
        report["walk_forward"]["xgboost_search"] = {
            "used_for_selection": False,
            "used_validation": False,
            "chosen": None,
            "candidates": [],
            "reason": "Dependência opcional xgboost não instalada.",
        }
    elif not both_classes(train_rows):
        report["metrics"]["xgboost"] = _skipped("Treino sem as duas classes.")
        report["walk_forward"]["xgboost_search"] = {
            "used_for_selection": False,
            "used_validation": False,
            "chosen": None,
            "candidates": [],
            "reason": "Treino sem as duas classes.",
        }
    else:
        chosen, search = _select_xgb_params(train_rows, seed)
        report["walk_forward"]["xgboost_search"] = search
        xgb_model = build_xgboost(seed, chosen)
        xgb_model.fit(rows_matrix(train_rows, FEATURE_COLUMNS), rows_labels(train_rows))
        fitted["xgboost"] = xgb_model
        report["metrics"]["xgboost"] = _metric_bundle(
            y_valid,
            predict_positive(xgb_model, valid_rows, FEATURE_COLUMNS),
        )

    challengers = [
        (name, report["metrics"][name]["brier_score"], model)
        for name, model in fitted.items()
        if "brier_score" in report["metrics"].get(name, {})
    ]
    margin = report["material_brier_gain"]
    baselines = [
        report["metrics"]["constant_half"]["brier_score"],
        report["metrics"]["home_rate"]["brier_score"],
    ]
    selected_name = None
    selected_model = None
    if not challengers:
        report["reason"] = "Nenhum challenger ajustado."
    else:
        selected_name, selected_score, selected_model = min(
            challengers,
            key=lambda item: (item[1], 0 if item[0] == "logistic" else 1),
        )
        report["selected_algorithm"] = selected_name
        if _beats(selected_score, baselines, margin):
            report["promoted"] = True
            report["reason"] = (
                f"{selected_name} teve Brier de validação menor que os baselines, "
                "com folga igual ou acima do gate proposto. Isso não é avaliação externa."
            )
        elif _beats(selected_score, baselines, 0.0):
            report["promoted"] = False
            report["reason"] = (
                "Houve melhora no Brier de validação, mas abaixo do gate proposto de ganho material. "
                "Champion não promovido."
            )
        else:
            report["promoted"] = False
            report["reason"] = "Nenhum challenger superou os baselines pré-registrados. Champion não promovido."

    inference_model = selected_model
    if selected_model is not None and calibration_method not in {"", "none"}:
        inference_model = _maybe_calibrate(
            report,
            selected_model,
            train_rows,
            valid_rows,
            calibration_method,
            float(config.get("calibration_fraction", 0.4)),
        )
    elif selected_model is not None:
        report["calibration"]["reason"] = "Calibração desligada."

    if selected_model is not None:
        selected_probs = predict_positive(selected_model, valid_rows, FEATURE_COLUMNS)
        bins = reliability_bins(y_valid, selected_probs)
        report["reliability"] = {
            "bins": bins,
            "algorithm": selected_name,
            "calibrated": False,
            "note": "Probabilidades não calibradas na validação. Não é resultado externo verificado.",
            "svg": reliability_svg(bins),
        }

    destination_dir = output_dir or Path(config.get("artifact_dir", "models"))
    if inference_model is not None:
        destination_dir.mkdir(parents=True, exist_ok=True)
        artifact_name = "champion.joblib" if report["promoted"] else "challenger.joblib"
        artifact_path = destination_dir / artifact_name
        joblib.dump(
            {
                "model": inference_model,
                "feature_columns": FEATURE_COLUMNS,
                "model_version": report["model_version"],
                "feature_version": report["feature_version"],
                "dataset_hash": report["dataset_hash"],
                "metrics": report["metrics"],
                "promoted": report["promoted"],
                "algorithm": selected_name,
                "calibrated": report["calibration"]["applied"],
                "calibration_method": calibration_method if report["calibration"]["applied"] else None,
            },
            artifact_path,
        )
        report["artifact_uri"] = str(artifact_path)

    if config.get("evaluate_heldout") and test_seasons and inference_model is not None:
        test_rows = supervised_rows(games, test_seasons, min_prior)
        if test_rows:
            report["heldout_metrics"] = _metric_bundle(
                rows_labels(test_rows),
                predict_positive(inference_model, test_rows, FEATURE_COLUMNS),
            )
            report["heldout_note"] = "Métricas de teste não foram usadas para promover o modelo."
            report["test_used_for_selection"] = False
    _write_report(config, report, output_dir)
    return report


def _select_xgb_params(train_rows: list[dict], seed: int) -> tuple[dict, dict]:
    candidates = []
    for params in XGB_GRID:
        folded = walk_forward_brier(lambda params=params: build_xgboost(seed, params), train_rows, FEATURE_COLUMNS)
        candidates.append({"params": dict(params), "mean_brier": folded["mean_brier"], "folds": len(folded["folds"])})
    scored = [item for item in candidates if item["mean_brier"] is not None]
    if scored:
        chosen = min(scored, key=lambda item: item["mean_brier"])["params"]
        reason = "Hiperparâmetros escolhidos só com folds do treino."
    else:
        chosen = dict(XGB_GRID[0])
        reason = "Busca ignorada por amostra curta. Usado o primeiro conjunto fixo, sem ler a validação."
    return chosen, {
        "used_for_selection": False,
        "used_validation": False,
        "seasons": seasons_of(train_rows),
        "chosen": chosen,
        "candidates": candidates,
        "reason": reason,
    }


def _maybe_calibrate(
    report: dict,
    model,
    train_rows: list[dict],
    valid_rows: list[dict],
    method: str,
    fraction: float,
):
    _early, late = later_window(valid_rows, fraction)
    overlap = overlap_count(train_rows, late)
    report["calibration"].update(
        {
            "method": method,
            "fraction": fraction,
            "applied": False,
            "used_for_selection": False,
            "disjoint_from_fit": overlap == 0,
            "seasons": seasons_of(late),
            "rows": len(late),
            "overlap_with_fit": overlap,
        }
    )
    if overlap:
        report["calibration"]["reason"] = "Calibração recusada: a janela sobrepõe o ajuste."
        return model
    if any(int(row["season"]) in set(report["test_seasons"]) for row in late):
        report["calibration"]["reason"] = "Calibração recusada: a janela inclui temporada de teste."
        return model
    if len(late) < 4 or not both_classes(late):
        report["calibration"]["reason"] = "Calibração não aplicada: janela posterior curta ou sem as duas classes."
        return model
    try:
        calibrated, normalized = fit_calibrator(model, late, method, FEATURE_COLUMNS)
    except ValueError as exc:
        report["calibration"]["reason"] = f"Calibração não aplicada: {exc}"
        return model
    report["calibration"]["method"] = normalized
    report["calibration"]["applied"] = True
    report["calibration"]["reason"] = (
        "Calibrador ajustado só na fatia final da validação, depois do fit. Não entrou na promoção."
    )
    return calibrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina baselines, logística e o challenger XGBoost.")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report = train_from_config(config)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "reason": report["reason"],
                "selected_algorithm": report.get("selected_algorithm"),
                "performance_verified": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
