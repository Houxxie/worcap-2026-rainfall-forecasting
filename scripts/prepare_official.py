"""Verify and unpack your own Kaggle competition download. Does not download or share it."""
from pathlib import Path
import argparse
import json
import sys
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_io import install_zip

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Kaggle Download All ZIP, with the 13 input files at its root")
    parser.add_argument("--output", default="data/official")
    args = parser.parse_args(argv)
    reference = json.loads((ROOT / "competition/reference.json").read_text(encoding="utf-8"))
    files = [dict(path=r["nome"], bytes=r["bytes"], sha256=r["sha256"]) for r in reference["entradas"]]
    try:
        print(install_zip(args.archive, args.output, files))
        print(f"{len(files)} official files verified. Destination: {Path(args.output).resolve()}")
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Official input preparation stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
