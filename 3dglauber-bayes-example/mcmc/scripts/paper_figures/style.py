"""Shared style for the paper-comparison figures (img/DESIGN_SYSTEM.md, section 1).

Every figure script imports this module; no colour, size or limit is defined elsewhere.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRIOR = "#BFBFBF"
BASELINE = "#2CA02C"
CS = "#E41A1C"
RBF = "#1F3FD8"
EXTRA = "#7B2D8E"
OLIVE = "#B5B83A"
LEARNED = "#FF7F00"  # proposal 3 (covinf), learned PHOBOS covariance (next free Set1 colour)
DASH = (0, (5, 3))
DASHDOT = (0, (5, 2, 1, 2))
HATCH = "++++"

CI = 0.90
SEED = 12345

FONT = dict(label=14, tick=11, title=12, legend=11)


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": FONT["label"],
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "axes.titlesize": FONT["title"],
        "legend.fontsize": FONT["legend"],
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
    })


def save(fig, outdir, stem):
    """PDF (vector) + PNG 300 dpi, same stem. Returns the two paths.

    Rendered to a local temporary file, then copied and size-checked: overwriting files in place
    in an iCloud Drive folder with savefig silently kept the old content (24/11/2026).
    """
    import shutil
    import tempfile
    from pathlib import Path
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    with tempfile.TemporaryDirectory() as tmp:
        for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
            local = Path(tmp) / f"{stem}.{ext}"
            fig.savefig(local, bbox_inches="tight", pad_inches=0.05, **kw)
            p = outdir / f"{stem}.{ext}"
            shutil.copyfile(local, p)
            if p.stat().st_size != local.stat().st_size:
                raise OSError(f"copy to {p} did not take effect")
            paths.append(p)
            print("Saved:", p, flush=True)
    plt.close(fig)
    return paths
