"""Check exact historical inputs without importing model libraries or training."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = {"numpy": "2.0.2", "pandas": "2.3.3", "xarray": "2025.12.0", "lightgbm": "4.6.0"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(config_path, root=ROOT):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if set(config) != {"profile", "inputs"} or config["profile"] not in {"competition", "research", "operational-fit"}:
        raise ValueError("Use profile competition, research or operational-fit and an inputs object.")
    if not isinstance(config["inputs"], dict) or set(config["inputs"]) != {"official", "seas5", "cfsv2"}:
        raise ValueError("inputs must specify official, seas5 and cfsv2 directories.")
    paths = {}
    for key, value in config["inputs"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Set a nonempty directory for inputs.{key}.")
        p = Path(value).expanduser()
        paths[key] = (config_path.parent / p).resolve() if not p.is_absolute() else p.resolve()
    reference = json.loads((root / "competition/reference.json").read_text(encoding="utf-8"))
    profile = config["profile"]
    expected = [("official", r["nome"], r["sha256"]) for r in reference["entradas"]
                if profile == "competition" or r["nome"].startswith("treino_")]
    phases = ["desenvolvimento"]
    if profile in {"competition", "operational-fit"}:
        phases.append("somente_ajuste_final")
    if profile == "competition":
        phases.append("teste")
    for key, folder, name in [("seas5", "SEAS5", "seas5_51_manifesto.json"),
                               ("cfsv2", "CFSv2", "cfsv2_manifesto.json")]:
        source = root / "competition/metadata" / folder / name
        manifest = json.loads(source.read_text(encoding="utf-8"))
        expected.append((key, name, sha256(source)))
        for phase in phases:
            row = manifest["arquivos"][phase]
            name = row["arquivo"]
            if Path(name).name != name or "\\" in name:
                raise ValueError("A manifest filename must be a basename.")
            expected.append((key, name, row["sha256"]))
    files = []
    for source, name, expected_hash in expected:
        p = paths[source] / name
        status = "missing"
        if p.is_file():
            status = "ok" if sha256(p) == expected_hash else "hash-mismatch"
        files.append(dict(source=source, file=name, path=str(p), status=status))
    versions = []
    for name, required in CORE.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        versions.append(dict(package=name, required=required, installed=installed,
                             status="ok" if installed == required else "missing-or-different"))
    ready = all(r["status"] == "ok" for r in files + versions)
    return dict(profile=profile, ready=ready, paths={k: str(v) for k, v in paths.items()},
                files=files, core_versions=versions,
                scope="Historical input bytes and core versions only. No training, downloads, source-arrival validation or forecast issuance.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        result = check(args.config)
        print(json.dumps(result, indent=2))
        if not result["ready"]:
            print("Not ready. Check missing/changed files and packages above. See docs/DATA.md; manifests alone are not datasets.", file=sys.stderr)
        return 0 if result["ready"] else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Input check stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
