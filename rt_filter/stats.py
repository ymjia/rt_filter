from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def rank_results(
    summary: pd.DataFrame,
    *,
    metric: str,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    if metric not in summary.columns:
        raise ValueError(f"metric '{metric}' not found in summary")
    return summary.sort_values(metric, ascending=lower_is_better).reset_index(drop=True)


def create_report(
    summary_csv: str | Path,
    *,
    output_dir: str | Path | None = None,
    metric: str = "to_reference_translation_rmse",
    lower_is_better: bool = True,
) -> Path:
    summary_path = Path(summary_csv)
    summary = load_summary(summary_path)
    if metric not in summary.columns:
        fallback = "filtered_acceleration_rms"
        if fallback not in summary.columns:
            raise ValueError(f"metric '{metric}' not found and no fallback metric exists")
        metric = fallback

    report_dir = Path(output_dir) if output_dir is not None else summary_path.parent / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_results(summary, metric=metric, lower_is_better=lower_is_better)
    ranked.to_csv(report_dir / "ranked_results.csv", index=False)
    _plot_algorithm_box(summary, metric=metric, output=report_dir / f"{metric}_by_algorithm.png")
    _plot_parameter_trends(summary, metric=metric, output_dir=report_dir)
    return report_dir


def _plot_algorithm_box(summary: pd.DataFrame, *, metric: str, output: Path) -> None:
    plt.figure(figsize=(10, 5))
    summary.boxplot(column=metric, by="algorithm", rot=30)
    plt.suptitle("")
    plt.title(metric)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _plot_parameter_trends(summary: pd.DataFrame, *, metric: str, output_dir: Path) -> None:
    parsed = summary.copy()
    params = parsed["params"].apply(_parse_params)
    keys = sorted({key for item in params for key in item})
    for key in keys:
        values = params.apply(lambda item: item.get(key))
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().all():
            continue
        parsed[f"param_{key}"] = numeric
        plt.figure(figsize=(10, 5))
        for algorithm, group in parsed.groupby("algorithm"):
            valid = group.dropna(subset=[f"param_{key}", metric]).sort_values(f"param_{key}")
            if valid.empty:
                continue
            plt.plot(valid[f"param_{key}"], valid[metric], marker="o", label=algorithm)
        plt.xlabel(key)
        plt.ylabel(metric)
        plt.title(f"{metric} vs {key}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_vs_{key}.png", dpi=150)
        plt.close()


def _parse_params(text: Any) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        import json

        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}
