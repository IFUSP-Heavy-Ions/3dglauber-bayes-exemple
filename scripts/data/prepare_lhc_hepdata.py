#!/usr/bin/env python3
"""Convert ALICE LHC HEPData CSV tables into hic-bayes runtime CSVs.

The script writes ``experiment.csv`` and ``exp_error.csv`` from HEPData CSV
blocks and, when model predictions are supplied, aligns and writes
``simulation.csv``.  The RHIC design/ranges now live under ``model/``; pass
them explicitly if a prediction study needs local copies.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULTS = {
    "AliceXeXe5440": {
        "hepdata": "data/AliceXeXe5440/HEPData-ins1672756-v2-csv",
        "dest": "data/AliceXeXe5440/lhc_predictions",
        "patterns": ["DN-DETARAP.csv"],
        "system_label": "Xe-Xe-5440",
    },
    "AlicePbPb5020": {
        "hepdata": "data/AlicePbPb5020/HEPData-ins1759506-v1-csv",
        "dest": "data/AlicePbPb5020/lhc_predictions",
        "patterns": ["Table*.csv"],
        "system_label": "Pb-Pb-5020",
    },
}


@dataclass
class Row:
    column: str
    value: float
    error: float
    source_file: str
    table_name: str
    observable: str
    centrality: str
    x_name: str
    x: float
    x_low: float | None
    x_high: float | None
    dependent: str


def _slug(text: str) -> str:
    text = text.replace("\\", "")
    text = re.sub(r"\$|{|}|\^|_", "", text)
    text = re.sub(r"[^A-Za-z0-9.+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


def _num(text: str) -> float | None:
    try:
        if text == "":
            return None
        value = float(text)
        if math.isfinite(value):
            return value
    except ValueError:
        pass
    return None


def _centrality_key(value: str) -> str:
    return (
        value.strip()
        .replace("%", "")
        .replace(".0", "")
        .replace("-", "_")
        .replace(" ", "")
    )


def _read_hepdata_blocks(path: Path) -> list[tuple[dict[str, str], list[str], list[list[str]]]]:
    blocks: list[tuple[dict[str, str], list[str], list[list[str]]]] = []
    meta: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[list[str]] = []

    def flush() -> None:
        nonlocal header, rows
        if header and rows:
            blocks.append((dict(meta), header, rows))
        header = None
        rows = []

    with path.open(newline="") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#:"):
                if header and rows:
                    flush()
                payload = line[2:].strip()
                if ":" in payload:
                    key, value = payload.split(":", 1)
                    meta[key.strip()] = value.strip()
                elif ",,," in payload:
                    key, value = payload.split(",,,", 1)
                    meta[key.strip()] = value.strip()
                continue

            parsed = next(csv.reader([line]))
            if header is None:
                header = parsed
            else:
                rows.append(parsed)
    flush()
    return blocks


def _block_rows(path: Path, system_label: str) -> list[Row]:
    parsed_rows: list[Row] = []
    for block_idx, (meta, header, rows) in enumerate(_read_hepdata_blocks(path)):
        centrality = (
            meta.get("CENTRALITY [pct]")
            or meta.get("CENTRALITY [PCT]")
            or meta.get("centrality")
            or f"block{block_idx:02d}"
        )
        cent_key = _centrality_key(centrality)
        observable = meta.get("keyword observables", meta.get("name", path.stem))
        table_name = meta.get("name", path.stem)

        x_idx = 0
        x_name = _slug(header[x_idx])
        x_low_idx = next((i for i, h in enumerate(header) if i > 0 and " LOW" in h.upper()), None)
        x_high_idx = next((i for i, h in enumerate(header) if i > 0 and " HIGH" in h.upper()), None)

        value_idx = None
        for idx, name in enumerate(header):
            low_name = name.lower()
            if idx == x_idx or idx == x_low_idx or idx == x_high_idx:
                continue
            if any(token in low_name for token in ["stat", "syst", "sys ", "unc", "error"]):
                continue
            value_idx = idx
            break
        if value_idx is None:
            continue

        error_indices = [
            idx
            for idx, name in enumerate(header)
            if idx != value_idx
            and any(token in name.lower() for token in ["stat", "syst", "sys ", "unc", "error"])
            and "+" in name
        ]
        dependent = _slug(header[value_idx])
        obs_key = _slug(observable)
        table_key = _slug(table_name)

        for bin_idx, row in enumerate(rows):
            if len(row) <= value_idx:
                continue
            x = _num(row[x_idx])
            value = _num(row[value_idx])
            if x is None or value is None:
                continue
            err2 = 0.0
            for err_idx in error_indices:
                if err_idx < len(row):
                    err = _num(row[err_idx])
                    if err is not None:
                        err2 += abs(err) ** 2
            error = math.sqrt(err2) if err2 > 0.0 else 0.0
            x_low = _num(row[x_low_idx]) if x_low_idx is not None and x_low_idx < len(row) else None
            x_high = _num(row[x_high_idx]) if x_high_idx is not None and x_high_idx < len(row) else None
            column = f"{system_label}_{table_key}_{obs_key}_cen_{cent_key}_{dependent}__bin{bin_idx:03d}"
            parsed_rows.append(
                Row(
                    column=column,
                    value=value,
                    error=error,
                    source_file=path.name,
                    table_name=table_name,
                    observable=observable,
                    centrality=centrality,
                    x_name=x_name,
                    x=x,
                    x_low=x_low,
                    x_high=x_high,
                    dependent=header[value_idx],
                )
            )
    return parsed_rows


def _copy_if_requested(src: Path | None, dest: Path, name: str) -> None:
    if src is None:
        return
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dest / name)


def _write_one_row_csv(path: Path, columns: list[str], values: list[float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow(values)


def _write_metadata(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "column",
                "source_file",
                "table_name",
                "observable",
                "centrality",
                "x_name",
                "x",
                "x_low",
                "x_high",
                "dependent",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "column": row.column,
                    "source_file": row.source_file,
                    "table_name": row.table_name,
                    "observable": row.observable,
                    "centrality": row.centrality,
                    "x_name": row.x_name,
                    "x": row.x,
                    "x_low": "" if row.x_low is None else row.x_low,
                    "x_high": "" if row.x_high is None else row.x_high,
                    "dependent": row.dependent,
                }
            )


def _write_simulation(src: Path, dest: Path, columns: list[str]) -> None:
    with src.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    if header == columns:
        output_header = header
        output_rows = rows
    elif len(header) == len(columns):
        output_header = columns
        output_rows = rows
    else:
        col_index = {name: idx for idx, name in enumerate(header)}
        missing = [col for col in columns if col not in col_index]
        if missing:
            raise ValueError(
                f"Simulation table cannot be aligned: {len(missing)} missing columns "
                f"out of {len(columns)}."
            )
        output_header = columns
        output_rows = [[row[col_index[col]] for col in columns] for row in rows]

    with (dest / "simulation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(output_header)
        writer.writerows(output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--hepdata-dir", help="Directory containing HEPData CSV files.")
    parser.add_argument("--dest", help="Destination directory for hic-bayes CSVs.")
    parser.add_argument(
        "--pattern",
        action="append",
        help="CSV glob to include. May be repeated. Defaults depend on --system.",
    )
    parser.add_argument("--simulation", help="Optional model predictions CSV to align/write.")
    parser.add_argument(
        "--design-points",
        default=None,
        help="Design points CSV to copy into the destination.",
    )
    parser.add_argument(
        "--design-ranges",
        default=None,
        help="Design ranges CSV to copy into the destination.",
    )
    parser.add_argument(
        "--allow-zero-errors",
        action="store_true",
        help="Keep observables with no parsed positive error column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = DEFAULTS[args.system]
    hepdata_dir = Path(args.hepdata_dir or defaults["hepdata"])
    dest = Path(args.dest or defaults["dest"])
    patterns = args.pattern or defaults["patterns"]
    system_label = defaults["system_label"]

    rows: list[Row] = []
    for pattern in patterns:
        for path in sorted(hepdata_dir.glob(pattern)):
            rows.extend(_block_rows(path, system_label))
    if not rows:
        raise RuntimeError(f"No HEPData rows parsed from {hepdata_dir}")
    if not args.allow_zero_errors:
        rows = [row for row in rows if row.error > 0.0]
    if not rows:
        raise RuntimeError("All parsed rows had zero error; use --allow-zero-errors to keep them.")

    dest.mkdir(parents=True, exist_ok=True)
    columns = [row.column for row in rows]
    _write_one_row_csv(dest / "experiment.csv", columns, [row.value for row in rows])
    _write_one_row_csv(dest / "exp_error.csv", columns, [row.error for row in rows])
    _write_metadata(dest / "observable_metadata.csv", rows)
    _copy_if_requested(Path(args.design_points) if args.design_points else None, dest, "design_points.csv")
    _copy_if_requested(Path(args.design_ranges) if args.design_ranges else None, dest, "design_ranges.csv")

    if args.simulation:
        _write_simulation(Path(args.simulation), dest, columns)

    metadata = [
        f"system: {args.system}",
        f"source_dir: {hepdata_dir}",
        f"n_observables: {len(columns)}",
        "observable_columns:",
        *[f"  - {column}" for column in columns],
        "",
    ]
    (dest / "conversion_metadata.yaml").write_text("\n".join(metadata))
    print(f"Wrote {len(columns)} observables to {dest}")
    if not args.simulation:
        print("No --simulation supplied; simulation.csv was not written.")


if __name__ == "__main__":
    main()
