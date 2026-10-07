"""Download exact historical climate snapshots; no credentials or model libraries."""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import urllib.request
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_io import digest, install_zip, installed

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/data_snapshots.json"


def download(row, root, archive_dir=None):
    if installed(root, row["files"]):
        return "already verified (no download)"
    with tempfile.TemporaryDirectory(prefix="worcap-download-") as temp:
        if archive_dir:
            archive = Path(archive_dir) / row["archive"]
        else:
            archive = Path(temp) / row["archive"]
            request = urllib.request.Request(row["url"], headers={"User-Agent": "WorCAP-data-downloader"})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("xb") as out:
                received = 0
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > row["bytes"]:
                        raise ValueError("Download exceeds recorded size")
                    out.write(chunk)
        if archive.stat().st_size != row["bytes"] or digest(archive) != row["sha256"]:
            raise ValueError(f"Archive hash or size mismatch: {row['archive']}")
        return install_zip(archive, root, row["files"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data", help="Data root, relative to current directory")
    parser.add_argument("--include-sst", action="store_true", help="Also get the optional research-only SST snapshot")
    parser.add_argument("--archive-dir", help="Read previously downloaded release ZIPs from this directory")
    args = parser.parse_args(argv)
    try:
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        for source in ["seas5", "cfsv2"] + (["sst"] if args.include_sst else []):
            print(f"{source}: {download(catalog['snapshots'][source], args.output, args.archive_dir)}", flush=True)
        print("Historical snapshots ready. Official competition inputs must be obtained separately; see docs/DATA.md.")
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Data preparation stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
