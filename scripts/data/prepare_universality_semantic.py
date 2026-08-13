#!/usr/bin/env python3
"""Convert Universality_Github data to hic-bayes CSVs with semantic columns.

The original backup converter writes observable columns as 0..286.  This
variant keeps the same numerical data, but names each observable by its
physical block:

    UxT_cen_00_05_ALICE__bin00
    ...
    UxT_cen_50_60_ALICE__bin40

It also writes small metadata tables that preserve the source design-point
ids and the x_T value associated with each observable column.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


IDF_NAMES = {
    0: "Grad",
    1: "CE",
    2: "PTM",
    3: "PTB",
}

IDF_LABELS = {
    0: "Grad",
    1: "Chapman-Enskog R.T.A",
    2: "Pratt-Torrieri-McNelis",
    3: "Pratt-Torrieri-Bernhard",
}

NAN_SETS: dict[int, set[int]] = {
    0: {334, 341, 377, 429, 447, 483},
    1: {285, 334, 341, 447, 483, 495},
    2: {209, 280, 322, 334, 341, 412, 421, 424, 429, 432, 446, 447, 453, 468, 483, 495},
    3: {60, 232, 280, 285, 322, 324, 341, 377, 432, 447, 464, 468, 482, 483, 485, 495},
}
UNFINISHED = {289, 324, 326, 459, 462, 242, 406, 440, 123}
STRANGE = {289, 324, 440, 459, 462}

NUM_DESIGN_POINTS = 325
N_CENTRALITIES = 7
N_XT_BINS = 41
CENTRALITIES = ["00_05", "05_10", "10_20", "20_30", "30_40", "40_50", "50_60"]
CENTRALITY_LABELS = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%"]


def delete_set(idf: int) -> set[int]:
    return NAN_SETS.get(idf, set()) | UNFINISHED | STRANGE


def observable_columns() -> list[str]:
    return [
        f"UxT_cen_{centrality}_ALICE__bin{bin_idx:02d}"
        for centrality in CENTRALITIES
        for bin_idx in range(N_XT_BINS)
    ]


def load_experimental(src: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(src / "Bayesian_data" / "ALICE_PbPb2p76.dat", comments="#")
    xt_values = data[:, 0::3][:N_CENTRALITIES, :N_XT_BINS]
    u_values = data[:, 1::3][:N_CENTRALITIES, :N_XT_BINS]
    u_errors = data[:, 2::3][:N_CENTRALITIES, :N_XT_BINS]
    return xt_values, u_values.reshape(-1), u_errors.reshape(-1)


def load_simulation(src: Path, idf: int) -> tuple[np.ndarray, list[int], list[int]]:
    bad = delete_set(idf)
    rows: list[np.ndarray] = []
    valid_ids: list[int] = []
    missing_ids: list[int] = []

    for dp in range(NUM_DESIGN_POINTS):
        if dp in bad:
            continue

        path = src / "Bayesian_data" / str(dp) / f"universal_alicecut_{idf}.dat"
        if not path.exists():
            missing_ids.append(dp)
            continue

        data = np.loadtxt(path, comments="#")
        rows.append(data[:N_CENTRALITIES, :N_XT_BINS].reshape(-1))
        valid_ids.append(dp)

    return np.asarray(rows), valid_ids, missing_ids


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, list(reader)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def load_design(src: Path, valid_ids: list[int]) -> tuple[list[str], list[list[str]]]:
    path = src / "design_pts_Pb_Pb_2760_production" / "design_points_main_PbPb-2760.dat"
    header, rows = read_csv_rows(path)
    data_header = header[1:]

    by_id: dict[int, list[str]] = {}
    for row in rows[:NUM_DESIGN_POINTS]:
        by_id[int(row[0])] = row[1:]

    return data_header, [by_id[idx] for idx in valid_ids]


def load_design_ranges(src: Path) -> tuple[list[str], list[list[str]]]:
    path = src / "design_pts_Pb_Pb_2760_production" / "design_ranges_main_PbPb-2760.dat"
    return read_csv_rows(path)


def write_observable_metadata(dest: Path, xt_values: np.ndarray, cols: list[str]) -> None:
    rows = []
    for cent_idx, (cent_key, cent_label) in enumerate(zip(CENTRALITIES, CENTRALITY_LABELS)):
        for bin_idx in range(N_XT_BINS):
            flat_idx = cent_idx * N_XT_BINS + bin_idx
            rows.append(
                {
                    "column": cols[flat_idx],
                    "observable": "U(x_T)",
                    "system": "Pb-Pb-2760",
                    "experiment": "ALICE",
                    "centrality": cent_label,
                    "centrality_key": cent_key,
                    "centrality_index": cent_idx,
                    "bin": bin_idx,
                    "x_T": xt_values[cent_idx, bin_idx],
                    "source_file": "Bayesian_data/ALICE_PbPb2p76.dat",
                    "source_layout": "triplets x_T,U(x_T),error",
                }
            )
    write_csv(
        dest / "observable_metadata.csv",
        [
            "column",
            "observable",
            "system",
            "experiment",
            "centrality",
            "centrality_key",
            "centrality_index",
            "bin",
            "x_T",
            "source_file",
            "source_layout",
        ],
        [[row[key] for key in row] for row in rows],
    )


def write_design_point_ids(dest: Path, idf: int, valid_ids: list[int]) -> None:
    write_csv(
        dest / "design_point_ids.csv",
        ["row", "universality_design_point_id", "idf", "delta_f"],
        [[row_idx, dp, idf, IDF_NAMES[idf]] for row_idx, dp in enumerate(valid_ids)],
    )


def write_conversion_metadata(
    dest: Path,
    src: Path,
    idf: int,
    cols: list[str],
    valid_ids: list[int],
    missing_ids: list[int],
) -> None:
    removed = sorted(delete_set(idf))
    lines = [
        "source_root: " + str(src),
        "source_system: Pb-Pb-2760",
        "source_experiment_file: Bayesian_data/ALICE_PbPb2p76.dat",
        "source_simulation_pattern: Bayesian_data/<design_point>/universal_alicecut_<idf>.dat",
        "source_design_file: design_pts_Pb_Pb_2760_production/design_points_main_PbPb-2760.dat",
        "source_design_ranges_file: design_pts_Pb_Pb_2760_production/design_ranges_main_PbPb-2760.dat",
        f"idf: {idf}",
        f"delta_f_short: {IDF_NAMES[idf]}",
        f"delta_f_label: {IDF_LABELS[idf]}",
        f"raw_design_rows: {NUM_DESIGN_POINTS}",
        f"design_rows: {len(valid_ids)}",
        f"n_centralities: {N_CENTRALITIES}",
        f"n_xt_bins: {N_XT_BINS}",
        f"n_observables: {len(cols)}",
        "centralities:",
    ]
    lines.extend(f"  - {label}" for label in CENTRALITY_LABELS)
    lines.append("design_remove_idx:")
    lines.extend(f"  - {idx}" for idx in removed)
    lines.append("missing_design_idx:")
    if missing_ids:
        lines.extend(f"  - {idx}" for idx in missing_ids)
    else:
        lines.append("  []")
    lines.append("obs_order:")
    lines.extend(f"  - UxT_cen_{cent}_ALICE" for cent in CENTRALITIES)
    lines.append("observable_columns:")
    lines.extend(f"  - {col}" for col in cols)
    lines.append("")
    (dest / "conversion_metadata.yaml").write_text("\n".join(lines))


def prepare_one(src: Path, base_dest: Path, idf: int, scale: bool, copy_emulator: bool) -> None:
    label = IDF_NAMES[idf]
    dest = base_dest / label
    dest.mkdir(parents=True, exist_ok=True)

    cols = observable_columns()
    xt_values, y_exp, y_err = load_experimental(src)
    simulation, valid_ids, missing_ids = load_simulation(src, idf)
    design_header, design_rows = load_design(src, valid_ids)
    ranges_header, ranges_rows = load_design_ranges(src)

    if simulation.shape != (len(valid_ids), len(cols)):
        raise ValueError(f"Unexpected simulation shape for idf={idf}: {simulation.shape}")
    if y_exp.shape[0] != len(cols) or y_err.shape[0] != len(cols):
        raise ValueError("Experimental vectors do not match semantic observable columns.")

    if scale:
        mean = simulation.mean(axis=0)
        std = simulation.std(axis=0, ddof=0)
        if np.any(std == 0.0):
            raise ValueError("Cannot scale data with zero-variance observable columns.")
        simulation = (simulation - mean) / std
        y_exp = (y_exp - mean) / std
        y_err = y_err / std

    write_csv(dest / "design_points.csv", design_header, design_rows)
    write_csv(dest / "design_ranges.csv", ranges_header, ranges_rows)
    write_csv(dest / "simulation.csv", cols, simulation.tolist())
    write_csv(dest / "experiment.csv", cols, [y_exp.tolist()])
    write_csv(dest / "exp_error.csv", cols, [y_err.tolist()])

    write_observable_metadata(dest, xt_values, cols)
    write_design_point_ids(dest, idf, valid_ids)
    write_conversion_metadata(dest, src, idf, cols, valid_ids, missing_ids)

    emulator_src = src / "emulators" / f"emulators_{label}_6_pcs.pkl"
    (dest / "emulator_src.txt").write_text(str(emulator_src) + "\n")
    if copy_emulator and emulator_src.exists():
        shutil.copy2(emulator_src, dest / emulator_src.name)

    print(f"{label}: wrote {len(valid_ids)} design rows x {len(cols)} observables -> {dest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Universality_Github data with semantic observable columns."
    )
    parser.add_argument("--src", required=True, help="Path to Universality_Github checkout.")
    parser.add_argument(
        "--dest-base",
        default="data/AtlasPbPb2760/universality_PbPb2760",
        help="Destination base directory. Model subdirectories are created inside it.",
    )
    parser.add_argument(
        "--idf",
        type=int,
        choices=sorted(IDF_NAMES),
        action="append",
        help="Viscous correction idf to prepare. Repeat to select multiple. Default: all.",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Apply per-observable z-score scaling using the simulation ensemble.",
    )
    parser.add_argument(
        "--copy-emulator",
        action="store_true",
        help="Copy emulators_<model>_6_pcs.pkl from Universality_Github into each output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    dest_base = Path(args.dest_base).expanduser()
    if not dest_base.is_absolute():
        dest_base = Path.cwd() / dest_base
    dest_base.mkdir(parents=True, exist_ok=True)

    idfs = args.idf if args.idf is not None else sorted(IDF_NAMES)
    for idf in idfs:
        prepare_one(src, dest_base, idf, args.scale, args.copy_emulator)


if __name__ == "__main__":
    main()
