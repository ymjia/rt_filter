from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rt_filter.batch import run_batch_config
from rt_filter.evaluation import compare_filter_result, trajectory_metrics, write_metrics
from rt_filter.filters import available_filters, run_filter
from rt_filter.io import read_trajectory, write_trajectory
from rt_filter.stats import create_report
from rt_filter.vtk_export import write_vtk_unstructured_grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rt-filter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog_parser = subparsers.add_parser("catalog", help="show available filters")
    catalog_parser.set_defaults(func=_cmd_catalog)

    filter_parser = subparsers.add_parser("filter", help="filter one trajectory")
    filter_parser.add_argument("input")
    filter_parser.add_argument("output")
    filter_parser.add_argument("--algorithm", "-a", required=True)
    filter_parser.add_argument("--param", "-p", action="append", default=[])
    filter_parser.set_defaults(func=_cmd_filter)

    batch_parser = subparsers.add_parser("batch", help="run batch config")
    batch_parser.add_argument("config")
    batch_parser.set_defaults(func=_cmd_batch)

    eval_parser = subparsers.add_parser("evaluate", help="evaluate one trajectory")
    eval_parser.add_argument("trajectory")
    eval_parser.add_argument("--reference")
    eval_parser.add_argument("--raw")
    eval_parser.add_argument("--output", "-o")
    eval_parser.set_defaults(func=_cmd_evaluate)

    report_parser = subparsers.add_parser("report", help="create plots and rankings from summary.csv")
    report_parser.add_argument("summary")
    report_parser.add_argument("--metric", default="to_reference_translation_rmse")
    report_parser.add_argument("--output-dir")
    report_parser.add_argument("--higher-is-better", action="store_true")
    report_parser.set_defaults(func=_cmd_report)

    vtk_parser = subparsers.add_parser(
        "export-vtk",
        help="export trajectory poses as a VTK UnstructuredGrid point cloud",
    )
    vtk_parser.add_argument("input")
    vtk_parser.add_argument("output")
    vtk_parser.add_argument("--normal-axis", default="z", help="x, y, z, -x, -y, -z")
    vtk_parser.add_argument("--no-axes", action="store_true")
    vtk_parser.add_argument("--no-path-distance", action="store_true")
    vtk_parser.set_defaults(func=_cmd_export_vtk)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def _cmd_catalog(_: argparse.Namespace) -> int:
    for info in available_filters().values():
        print(f"{info.name}: {info.description} defaults={info.defaults}")
    return 0


def _cmd_filter(args: argparse.Namespace) -> int:
    traj = read_trajectory(args.input)
    filtered = run_filter(args.algorithm, traj, _parse_params(args.param))
    write_trajectory(filtered, args.output)
    print(f"wrote {args.output}")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    run_dir = run_batch_config(args.config)
    print(f"batch output: {run_dir}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    traj = read_trajectory(args.trajectory)
    if args.raw:
        raw = read_trajectory(args.raw)
        reference = read_trajectory(args.reference) if args.reference else None
        metrics = compare_filter_result(raw, traj, reference=reference)
    elif args.reference:
        reference = read_trajectory(args.reference)
        from rt_filter.evaluation import delta_metrics

        metrics = trajectory_metrics(traj)
        metrics.update(delta_metrics(traj, reference))
    else:
        metrics = trajectory_metrics(traj)

    if args.output:
        write_metrics(metrics, args.output)
        print(f"wrote {args.output}")
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    report_dir = create_report(
        args.summary,
        output_dir=args.output_dir,
        metric=args.metric,
        lower_is_better=not args.higher_is_better,
    )
    print(f"report output: {report_dir}")
    return 0


def _cmd_export_vtk(args: argparse.Namespace) -> int:
    traj = read_trajectory(args.input)
    write_vtk_unstructured_grid(
        traj,
        args.output,
        normal_axis=args.normal_axis,
        include_axes=not args.no_axes,
        include_path_distance=not args.no_path_distance,
    )
    print(f"wrote {args.output}")
    return 0


def _parse_params(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"parameter must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        params[key] = _parse_value(raw_value)
    return params


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
