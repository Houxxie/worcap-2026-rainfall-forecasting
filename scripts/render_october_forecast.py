"""Display the frozen forecast; never modify stored prediction values."""
from pathlib import Path
import json
import argparse
import hashlib
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--forecast', type=Path, required=True)
parser.add_argument('--coastline', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
metadata = json.loads((Path(__file__).resolve().parents[1] / 'assets/forecast_october_2026.json').read_text(encoding='utf-8'))
for path, expected in [(args.forecast, metadata['forecast_sha256']), (args.coastline, metadata['coastline']['sha256'])]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Input differs from the recorded October figure: ' + path.name)
if args.output.exists():
    raise FileExistsError('Choose a new output path; this script does not overwrite existing figures.')
args.output.parent.mkdir(parents=True, exist_ok=True)
with xr.open_dataset(args.forecast, engine='h5netcdf') as ds:
    ds = ds.load()
coast = json.loads(args.coastline.read_text(encoding='utf-8'))
fig, axes = plt.subplots(1, 3, figsize=(15, 8))
fig.patch.set_facecolor('#f7f8fa')
fig.subplots_adjust(left=.055, right=.98, bottom=.17, top=.81, wspace=.22)
p, c = ds.precipitacao.values[0], ds.climatologia.values[0]
levels = [0, .5, 1, 2, 4, 6, 8, 12, 20, 40, 80, 100]
delta_levels = [-20, -10, -5, -3, -2, -1, -.5, 0, .5, 1, 2, 3, 5, 10, 20]
for i, (field, title) in enumerate([(p, 'Frozen forecast'), (c, 'Training climatology (1993–2022)'),
                                    (p-c, 'Forecast − climatology')]):
    lev = levels if i < 2 else delta_levels
    cmap = plt.get_cmap('YlGnBu' if i < 2 else 'BrBG', len(lev)-1)
    ax = axes[i]
    im = ax.pcolormesh(ds.lon, ds.lat, field, cmap=cmap, norm=BoundaryNorm(lev, cmap.N), shading='auto')
    for feature in coast['features']:
        g = feature['geometry']
        for line in ([g['coordinates']] if g['type']=='LineString' else g['coordinates']):
            points = np.asarray(line)
            ax.plot(points[:, 0], points[:, 1], color='#374151', lw=.45, alpha=.8)
    ax.set(xlim=(-90, -25), ylim=(-60, 15), title=title, xlabel='Longitude (°)')
    ax.set_aspect('equal')
    ax.grid(alpha=.17)
    ax.set_yticks([-60, -45, -30, -15, 0, 15])
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=.98, pad=.13, fraction=.05)
    cb.set_ticks([0, 1, 2, 4, 8, 20, 80, 100] if i < 2 else [-20, -5, -2, -.5, .5, 2, 5, 20])
    cb.set_label('Monthly mean precipitation (mm/day)' if i < 2 else 'Difference (mm/day)')
axes[0].set_ylabel('Latitude (°)')
fig.suptitle('October 2026 · South America rainfall forecast', fontsize=19, y=.96)
fig.text(.5, .89, 'Frozen 30 September, 18:18 UTC / 15:18 Brasília', ha='center', fontsize=13)
fig.text(.5, .075, 'Forecast, not observations · Unequal color intervals · Coastline: Natural Earth', ha='center', fontsize=10)
fig.savefig(args.output, dpi=170, facecolor=fig.get_facecolor())
plt.close(fig)
