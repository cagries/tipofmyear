#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DEFAULT_METRICS = [
    "window_acc",
    "performance_acc",
    "macro_performance_acc",
    "top5_performance_acc",
    "macro_top5_performance_acc",
]


def summarize_metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def select_metric_payload(payload: dict[str, object], metric_prefix: str | None = None) -> dict[str, object]:
    selected: object = payload
    if metric_prefix:
        for part in metric_prefix.split("."):
            if not isinstance(selected, dict):
                raise KeyError(f"Cannot descend into non-object at {part!r}.")
            selected = selected[part]
    if not isinstance(selected, dict):
        raise TypeError("Selected metric payload is not a JSON object.")
    return selected


def summarize_results(
    results_root: Path,
    folds: list[str],
    metric_keys: list[str] | None = None,
    metrics_path: str = "metrics.json",
    metric_prefix: str | None = None,
) -> dict[str, object]:
    metric_keys = metric_keys or DEFAULT_METRICS
    per_fold = []
    for fold in folds:
        fold_metrics_path = results_root / fold / metrics_path
        if not fold_metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {fold_metrics_path}")
        payload = json.loads(fold_metrics_path.read_text(encoding="utf-8"))
        metrics = select_metric_payload(payload, metric_prefix=metric_prefix)
        per_fold.append(
            {
                "fold": fold,
                **{key: float(metrics[key]) for key in metric_keys},
            }
        )

    summary = {
        key: summarize_metric([float(row[key]) for row in per_fold])
        for key in metric_keys
    }
    return {
        "results_root": str(results_root),
        "folds": folds,
        "metrics_path": metrics_path,
        "metric_prefix": metric_prefix,
        "metric_keys": metric_keys,
        "per_fold": per_fold,
        "summary": summary,
    }


def markdown_table(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    lines = [
        "| Metric | Mean | Std |",
        "| --- | ---: | ---: |",
    ]
    for key in payload["metric_keys"]:
        stats = summary[key]
        lines.append(f"| `{key}` | {stats['mean']:.4f} | {stats['std']:.4f} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--metrics-path", default="metrics.json")
    parser.add_argument("--metric-prefix")
    parser.add_argument("--metric-keys", nargs="+", default=DEFAULT_METRICS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = summarize_results(
        results_root=args.results_root,
        folds=args.folds,
        metric_keys=args.metric_keys,
        metrics_path=args.metrics_path,
        metric_prefix=args.metric_prefix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown_table(payload), encoding="utf-8")
    print(markdown_table(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
