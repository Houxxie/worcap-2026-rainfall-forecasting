"""Maintainer: package retained, hash-matching climate snapshots, never official inputs."""
from pathlib import Path
import argparse
import json
import sys
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_io import digest

ROOT = Path(__file__).resolve().parents[1]
TAG = "data-snapshots-v1"
BASE = f"https://github.com/Houxxie/worcap-2026-rainfall-forecasting/releases/download/{TAG}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seas5", type=Path, required=True)
    parser.add_argument("--cfsv2", type=Path, required=True)
    parser.add_argument("--sst", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    notices = (ROOT / "docs/DATA_RIGHTS.md").read_bytes()
    catalog = {"schema": "worcap_public_snapshots_v1", "release": TAG, "snapshots": {}}
    for source, folder, prefix in [("seas5", "SEAS5", "seas5_51"), ("cfsv2", "CFSv2", "cfsv2"), ("sst", None, None)]:
        contents = {f"{source}/NOTICE.txt": notices}
        if folder:
            manifest_path = ROOT / "competition/metadata" / folder / f"{prefix}_manifesto.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            contents[f"{source}/{manifest_path.name}"] = manifest_path.read_bytes()
            for record in manifest["arquivos"].values():
                name = record["arquivo"]
                if Path(name).name != name or "\\" in name:
                    raise ValueError("Expected a manifest basename")
                path = getattr(args, source) / name
                if digest(path) != record["sha256"]:
                    raise ValueError(f"Snapshot mismatch: {path.name}")
                contents[f"{source}/{name}"] = path.read_bytes()
        else:
            protocol = json.loads((ROOT / "research/lagged_sources/protocol.json").read_text(encoding="utf-8"))
            if digest(args.sst) != protocol["hash_sst"]:
                raise ValueError("SST snapshot mismatch")
            contents[f"sst/{args.sst.name}"] = args.sst.read_bytes()
        filename = f"{source}-snapshot-v1.zip"
        archive = args.output / filename
        rows = []
        import hashlib
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 7, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                z.writestr(info, data, compresslevel=6)
                rows.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        catalog["snapshots"][source] = dict(archive=filename, url=f"{BASE}/{filename}", bytes=archive.stat().st_size, sha256=digest(archive), files=rows)
        print(filename, archive.stat().st_size)
    (args.output / "data_snapshots.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
