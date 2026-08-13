#!/usr/bin/env python3
"""Check migrated RHIC Au-Au/d-Au tables used by AllDetectorsAuAu200 studies."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


SYSTEMS = {
    "Au-Au-200": {
        "data_dir": "data/AllDetectorsAuAu200/Au-Au-200",
        "hic_model_dir": "model/RHICAuAu200/Au-Au-200",
        "model_dir": "model_calculations/production_414pts_Au_Au_200/Obs",
        "design_dir": "production_designs/ProductionMaxPro-Mod/design_pts_Au_Au_200_production",
        "active_observables": {
            "dNdeta_eta_cen_00_03_PHOB": 54,
            "dNdeta_eta_cen_03_06_PHOB": 54,
            "dNdeta_eta_cen_06_10_PHOB": 54,
            "dNdeta_eta_cen_10_15_PHOB": 54,
            "dNdeta_eta_cen_15_20_PHOB": 54,
            "dNdeta_eta_cen_20_25_PHOB": 54,
            "dNdeta_eta_cen_25_30_PHOB": 54,
            "dNdeta_eta_cen_30_35_PHOB": 54,
            "dNdeta_eta_cen_35_40_PHOB": 54,
            "dNdeta_eta_cen_40_45_PHOB": 54,
            "dNdeta_eta_cen_45_50_PHOB": 54,
            "v22_eta_cen_20_70_STAR": 21,
            "v22_eta_cen_03_15_PHOB": 16,
            "v22_eta_cen_15_25_PHOB": 16,
            "v22_eta_cen_25_50_PHOB": 16,
            "v22_pt_cen_00_10_PHEN": 5,
            "v22_pt_cen_10_20_PHEN": 5,
            "v22_pt_cen_20_30_PHEN": 5,
            "v22_pt_cen_30_40_PHEN": 5,
            "v22_pt_cen_40_50_PHEN": 5,
            "v22_pt_cen_50_60_PHEN": 5,
            "v22_pt_cen_00_05_STAR": 8,
            "v22_pt_cen_05_10_STAR": 8,
            "v22_pt_cen_10_20_STAR": 8,
            "v22_pt_cen_20_30_STAR": 8,
            "v22_pt_cen_30_40_STAR": 8,
            "v22_pt_cen_40_50_STAR": 8,
            "v22_pt_cen_50_60_STAR": 8,
            "v22_int_STAR": 6,
            "v32_int_STAR": 6,
            "meanpT_pi_STAR": 7,
            "meanpT_k_STAR": 7,
            "meanpT_k_PHEN": 8,
            "meanpT_p_PHEN": 8,
        },
    },
    "d-Au-200": {
        "data_dir": "data/AllDetectorsAuAu200/d-Au-200",
        "hic_model_dir": "model/RHICAuAu200/d-Au-200",
        "model_dir": "model_calculations/production_414pts_d_Au_200/Obs",
        "design_dir": "production_designs/ProductionMaxPro-Mod/design_pts_d_Au_200_production",
        "active_observables": {
            "dNdeta_eta_cen_00_05_PHEN": 34,
            "dNdeta_eta_cen_05_10_PHEN": 34,
            "dNdeta_eta_cen_10_20_PHEN": 34,
            "v22_eta_cen_00_05_PHEN": 14,
            "v22_pt_cen_00_05_PHEN": 5,
            "v22_pt_cen_00_10_STAR": 5,
        },
    },
}


def csv_shape(path: Path) -> tuple[int, list[str]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, header


def data_rows(path: Path) -> int:
    rows = 0
    with path.open(errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows += 1
    return rows


def read_metadata(path: Path) -> dict[str, object]:
    meta: dict[str, object] = {}
    current_list: str | None = None
    values: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if current_list and stripped.startswith("- "):
            values.append(stripped[2:].strip().strip("'\""))
            continue
        if current_list:
            meta[current_list] = values
            current_list = None
            values = []
        if ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        raw = raw.strip()
        if raw == "":
            current_list = key
            values = []
        elif raw.isdigit():
            meta[key] = int(raw)
        elif raw.lower() in {"true", "false"}:
            meta[key] = raw.lower() == "true"
        else:
            meta[key] = raw.strip("'\"")
    if current_list:
        meta[current_list] = values
    return meta


def group_columns(columns: list[str]) -> dict[str, int]:
    groups: dict[str, int] = {}
    for column in columns:
        obs = column.rsplit("__bin", 1)[0] if "__bin" in column else column
        groups[obs] = groups.get(obs, 0) + 1
    return groups


def check_system(repo: Path, jetscape: Path, system: str) -> list[str]:
    info = SYSTEMS[system]
    data_dir = repo / info["data_dir"]
    hic_model_dir = repo / info["hic_model_dir"]
    messages: list[str] = [f"\n{system}"]
    required_data = [
        "experiment.csv",
        "exp_error.csv",
        "conversion_metadata.yaml",
    ]
    required_model = [
        "design_points.csv",
        "design_ranges.csv",
        "simulation.csv",
        "validation_design_points.csv",
        "validation_simulation.csv",
    ]
    missing = [f"data/{name}" for name in required_data if not (data_dir / name).exists()]
    missing.extend(f"model/{name}" for name in required_model if not (hic_model_dir / name).exists())
    if missing:
        messages.append(f"  FAIL missing files: {', '.join(missing)}")
        return messages

    design_rows, design_cols = csv_shape(hic_model_dir / "design_points.csv")
    sim_rows, sim_cols = csv_shape(hic_model_dir / "simulation.csv")
    exp_rows, exp_cols = csv_shape(data_dir / "experiment.csv")
    err_rows, err_cols = csv_shape(data_dir / "exp_error.csv")
    val_design_rows, val_design_cols = csv_shape(hic_model_dir / "validation_design_points.csv")
    val_rows, val_cols = csv_shape(hic_model_dir / "validation_simulation.csv")
    ranges_rows, ranges_cols = csv_shape(hic_model_dir / "design_ranges.csv")
    meta = read_metadata(data_dir / "conversion_metadata.yaml")

    messages.append(f"  design rows:      {design_rows}")
    messages.append(f"  simulation rows:  {sim_rows}")
    messages.append(f"  validation rows:  {val_rows}")
    messages.append(f"  observables:      {len(sim_cols)}")
    messages.append(f"  parameters:       {len(design_cols)}")

    checks = [
        (design_rows == sim_rows, "design_points and simulation have same row count"),
        (exp_rows == 1 and err_rows == 1, "experiment and exp_error have one row"),
        (exp_cols == sim_cols, "experiment columns exactly match simulation"),
        (err_cols == sim_cols, "exp_error columns exactly match simulation"),
        (val_design_rows == val_rows, "validation_design_points and validation_simulation have same row count"),
        (val_design_cols == design_cols, "validation_design_points columns match design_points"),
        (val_cols == sim_cols, "validation_simulation columns match simulation"),
        (ranges_rows == len(design_cols), "design_ranges rows match parameter count"),
        ("param" in ranges_cols and "min" in ranges_cols and "max" in ranges_cols, "design_ranges has param/min/max"),
    ]

    raw_rows = meta.get("raw_design_rows")
    removed = meta.get("design_remove_idx", [])
    if isinstance(raw_rows, int) and isinstance(removed, list):
        if raw_rows - len(removed) != design_rows:
            messages.append(
                "  NOTE design_remove_idx is recorded, but design_points.csv still has "
                f"{design_rows} rows; leaving design migration untouched for now"
            )

    simulation_columns = meta.get("simulation_columns", [])
    if isinstance(simulation_columns, list):
        checks.append((simulation_columns == sim_cols, "metadata simulation_columns match simulation columns"))

    active_observables = info["active_observables"]
    groups = group_columns(sim_cols)
    checks.append((groups == active_observables, "simulation columns match active calibration observables"))

    model_dir = jetscape / info["model_dir"]
    design_dir = jetscape / info["design_dir"]
    source_files = [
        model_dir / "main.dat",
        model_dir / "validation.dat",
        model_dir / "Simulation",
        model_dir / "Validation",
    ]
    source_files.extend(sorted(design_dir.glob("design_points_main_*.dat")))
    source_files.extend(sorted(design_dir.glob("design_ranges_main_*.dat")))
    checks.append((all(path.exists() for path in source_files), "JETSCAPE source files are present"))

    if source_files and (model_dir / "main.dat").exists():
        messages.append(f"  source main.dat nonempty lines: {data_rows(model_dir / 'main.dat')}")

    failed = False
    for ok, label in checks:
        messages.append(f"  {'OK  ' if ok else 'FAIL'} {label}")
        failed = failed or not ok
    if failed:
        messages.append("  RESULT: migration should be inspected or regenerated")
    else:
        messages.append("  RESULT: migrated tables look internally consistent")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument(
        "--jetscape",
        default=Path("../jetscape-emulator"),
        type=Path,
        help="Path to the sibling jetscape-emulator checkout.",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    jetscape = (repo / args.jetscape).resolve() if not args.jetscape.is_absolute() else args.jetscape
    for system in SYSTEMS:
        print("\n".join(check_system(repo, jetscape, system)))


if __name__ == "__main__":
    main()
