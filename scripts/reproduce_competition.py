"""Check inputs and reproduce the competition CSV without editing the model code."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_inputs import check

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/competition.inputs.json"))
    parser.add_argument("--output", default="outputs/reproduction")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--validation", action="store_true", help="Also refit seven historical blocks (much slower)")
    args = parser.parse_args(argv)
    result = check(args.config)
    print(json.dumps(result, indent=2), flush=True)
    if result["profile"] != "competition" or not result["ready"]:
        print("Not ready for competition reproduction. Fix the files/packages listed above.", file=sys.stderr)
        return 2
    if args.check_only:
        return 0
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for source, key in [("official", "WORCAP_OFFICIAL_DIR"), ("seas5", "WORCAP_SEAS5_DIR"), ("cfsv2", "WORCAP_CFSV2_DIR")]:
        env[key] = result["paths"][source]
    env.update(WORCAP_REVALIDATE="1" if args.validation else "0", WORCAP_OUTPUT_ROOT=str(output), MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    return subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "competition/hybrid_forecast.py")], env=env, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
