"""Rerender example forecast/system window plots from saved evaluation CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys as _sys

    _sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.reporting import build_example_window_report, build_system_example_window_report

DEFAULT_MODELS = ("lstm", "deepar", "tft", "chronos2", "seasonal_naive")
DEFAULT_KINDS = ("forecast",)


def _parse_csv(raw: str | None, *, default: tuple[str, ...]) -> list[str]:
    if raw is None:
        return list(default)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or list(default)


def _parse_window_spec(raw: str) -> dict[str, Any]:
    series, separator, start_text = raw.rpartition(":")
    if not separator or not series:
        raise ValueError(f"Invalid window selector '{raw}'. Expected format SERIES:START_INDEX.")
    return {"series": series, "start_index": int(start_text)}


def _parse_model_windows(raw: str | None) -> dict[str, dict[str, Any]]:
    selectors: dict[str, dict[str, Any]] = {}
    if not raw:
        return selectors
    for entry in [part.strip() for part in raw.split(",") if part.strip()]:
        model_name, separator, window_spec = entry.partition("=")
        if not separator or not model_name:
            raise ValueError(
                f"Invalid per-model window selector '{entry}'. Expected format MODEL=SERIES:START_INDEX."
            )
        selectors[model_name.strip()] = _parse_window_spec(window_spec.strip())
    return selectors


def _parse_model_row_indexes(raw: str | None) -> dict[str, dict[str, int]]:
    selectors: dict[str, dict[str, int]] = {}
    if not raw:
        return selectors
    for entry in [part.strip() for part in raw.split(",") if part.strip()]:
        model_name, separator, row_text = entry.partition(":")
        if not separator or not model_name:
            raise ValueError(f"Invalid per-model row selector '{entry}'. Expected format MODEL:ROW_INDEX.")
        selectors[model_name.strip()] = {"row_index": int(row_text)}
    return selectors


def _merge_model_selectors(*selector_maps: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for selector_map in selector_maps:
        for model_name, selector in selector_map.items():
            current = dict(merged.get(model_name, {}))
            current.update(selector)
            merged[model_name] = current
    return merged


def _load_report_index(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_report_index(path: Path, entries: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment directory containing forecast_eval/ and optionally system_eval/ artifacts.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to render from forecast_eval and system_eval directories.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names to render. Defaults to lstm,deepar,tft,chronos2,seasonal_naive.",
    )
    parser.add_argument(
        "--kinds",
        default=",".join(DEFAULT_KINDS),
        help="Comma-separated report kinds to build: forecast,system. Default: forecast.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Default row index to render within each saved windows CSV.",
    )
    parser.add_argument(
        "--shared-window",
        help="Use the same SERIES:START_INDEX selector across models.",
    )
    parser.add_argument(
        "--model-row-indexes",
        help="Per-model row selectors as MODEL:ROW_INDEX pairs separated by commas.",
    )
    parser.add_argument(
        "--model-windows",
        help="Per-model window selectors as MODEL=SERIES:START_INDEX pairs separated by commas.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested model has no matching saved window for the selected row/window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    models = _parse_csv(args.models, default=DEFAULT_MODELS)
    kinds = _parse_csv(args.kinds, default=DEFAULT_KINDS)

    invalid_kinds = [kind for kind in kinds if kind not in {"forecast", "system"}]
    if invalid_kinds:
        raise ValueError(f"Unsupported --kinds values: {', '.join(invalid_kinds)}")

    shared_selector: dict[str, Any] = {"row_index": int(args.row_index)}
    if args.shared_window:
        shared_selector.update(_parse_window_spec(args.shared_window))

    model_selectors = _merge_model_selectors(
        _parse_model_row_indexes(args.model_row_indexes),
        _parse_model_windows(args.model_windows),
    )

    outputs: dict[str, str] = {}
    if "forecast" in kinds:
        outputs.update(
            build_example_window_report(
                experiment_dir,
                split=args.split,
                models=models,
                shared_selector=shared_selector,
                model_selectors=model_selectors,
                strict=args.strict,
                include_extra_models=False,
                include_preferred_model=False,
            )
        )
    if "system" in kinds:
        outputs.update(
            build_system_example_window_report(
                experiment_dir,
                split=args.split,
                models=models,
                shared_selector=shared_selector,
                model_selectors=model_selectors,
                strict=args.strict,
                include_extra_models=False,
                include_preferred_model=False,
            )
        )

    report_index_path = experiment_dir / "reports" / "report_index.json"
    report_index = _load_report_index(report_index_path)
    report_index.update(outputs)
    _save_report_index(report_index_path, report_index)

    print(f"Rendered {len(outputs)} example plot artifacts for split='{args.split}' in '{experiment_dir}'.")
    for artifact_name, output_path in sorted(outputs.items()):
        print(f"  {artifact_name}: {output_path}")


if __name__ == "__main__":
    main()
