"""Render the first submitted month from the verified competition CSV; no fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["submission", "coastline", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = json.loads((root / "competition/reference.json").read_text(encoding="utf-8"))
    coast_ref = json.loads((root / "assets/forecast_october_2026.json").read_text(encoding="utf-8"))["coastline"]
    for path, digest in [(args.submission, ref["hash_solucao"]), (args.coastline, coast_ref["sha256"])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Different input bytes: " + path.name)
    if args.output.exists():
        raise FileExistsError("Choose a new figure path.")
    rows = pd.read_csv(args.submission, dtype={"id": str})
    rows = rows[rows.id.str.startswith("2023_01_")].copy()
    ids = rows.id.str.split("_", expand=True)
    rows["lat"], rows["lon"] = ids[2].astype(float), ids[3].astype(float)
    field = rows.pivot(index="lat", columns="lon", values="tp_mm_day").sort_index().sort_index(axis=1)
    if not (field.shape == (301, 261) and np.isfinite(field.values).all() and (field.values >= 0).all()
            and np.array_equal(field.index, np.arange(-60, 15.25, .25))
            and np.array_equal(field.columns, np.arange(-90, -24.75, .25))):
        raise ValueError("Incomplete or invalid forecast grid.")
    coast = json.loads(args.coastline.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.8, 8.3))
    fig.patch.set_facecolor("#f7f8fa")
    levels = [0, .5, 1, 2, 4, 6, 8, 12, 20, 40, 80, 120]
    cmap = plt.get_cmap("YlGnBu", len(levels)-1)
    im = ax.pcolormesh(field.columns, field.index, field.values,
                      cmap=cmap, norm=BoundaryNorm(levels, cmap.N), shading="auto")
    for feature in coast["features"]:
        g = feature["geometry"]
        for line in ([g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]):
            xy = np.asarray(line)
            ax.plot(xy[:, 0], xy[:, 1], color="#374151", lw=.45)
    ax.set(xlim=(-90, -25), ylim=(-60, 15), xlabel="Longitude (°)", ylabel="Latitude (°)")
    ax.set_aspect("equal")
    ax.grid(alpha=.15)
    fig.suptitle("Competition hybrid · January 2023", fontsize=17)
    ax.set_title("First target month in the submitted CSV", fontsize=11)
    cb = fig.colorbar(im, ax=ax, pad=.04, shrink=.85)
    cb.set_label("Monthly mean precipitation (mm/day)")
    fig.text(.5, .025, "Archived prediction, not observations · Unequal color intervals\nCoastline: Natural Earth · No new training", ha="center", fontsize=9)
    fig.subplots_adjust(left=.12, right=.91, top=.88, bottom=.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
