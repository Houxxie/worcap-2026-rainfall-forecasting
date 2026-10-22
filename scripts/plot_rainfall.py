"""Shared presentation for saved rainfall fields; no resampling or value changes."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def render_panel(lon, lat, field, coast, output, *, title, subtitle, levels, anomaly=False):
    fig = plt.figure(figsize=(9, 9.2), facecolor="#f5f8f7")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, .046], left=.10, right=.86,
                           bottom=.12, top=.84, wspace=.13)
    ax, cax = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    cmap = plt.get_cmap("BrBG" if anomaly else "YlGnBu", len(levels) - 1)
    im = ax.pcolormesh(lon, lat, field, cmap=cmap,
                      norm=BoundaryNorm(levels, cmap.N), shading="auto")
    for feature in coast["features"]:
        geometry = feature["geometry"]
        lines = [geometry["coordinates"]] if geometry["type"] == "LineString" else geometry["coordinates"]
        for line in lines:
            xy = np.asarray(line)
            ax.plot(xy[:, 0], xy[:, 1], color="#374151", lw=.5, alpha=.8)
    ax.set(xlim=(-90, -25), ylim=(-60, 15), xlabel="Longitude (°)", ylabel="Latitude (°)")
    ax.set_aspect("equal")
    ax.grid(alpha=.12)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    for spine in ax.spines.values():
        spine.set_color("#526874")
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-20, -5, -1, 0, 1, 5, 20] if anomaly else [0, 1, 4, 8, 20, 80, levels[-1]])
    cb.ax.tick_params(labelsize=10)
    cb.set_label("Difference (mm/day)" if anomaly else "Mean precipitation (mm/day)", fontsize=12, labelpad=12)
    cb.outline.set_edgecolor("#526874")
    fig.text(.5, .955, title, ha="center", va="top", fontsize=22, weight="bold", color="#17333d")
    fig.text(.5, .902, subtitle, ha="center", va="top", fontsize=15, color="#526874")
    fig.text(.5, .035, "Color scale uses unequal intervals", ha="center", fontsize=10, color="#526874")
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
