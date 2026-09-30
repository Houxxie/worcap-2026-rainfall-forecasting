"""Build an offline results viewer from checked, saved evidence (standard library)."""
from pathlib import Path
import argparse
import base64
import csv
import hashlib
import io
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    manifest = json.loads((root / "demo/sources.json").read_text(encoding="utf-8"))
    checked = {}
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Demo input leaves the repository.")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Demo evidence changed: " + name)
        checked[name] = data

    def table(name, model=None):
        rows = list(csv.DictReader(io.StringIO(checked[name].decode("utf-8"))))
        out = []
        for r in rows:
            if model and r.get("modelo", r.get("model")) != model:
                continue
            row = dict(rmse=float(r["rmse"]), mae=float(r["mae"]), bias=float(r.get("bias", r.get("vies"))))
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError("Nonfinite saved metric.")
            if "year" in r or "ano" in r:
                row["year"] = int(r.get("year", r.get("ano")))
            out.append(row)
        if not out:
            raise ValueError("No matching model rows in " + name)
        return out

    data = {
        "competition": {
            "global": table("competition/results/global.csv", "hybrid")[0],
            "years": table("competition/results/years.csv", "hybrid")},
        "current": {
            "global": table("research/lagged_sources/evidence/metricas_globais.csv", "controle_defasado")[0],
            "years": table("research/lagged_sources/evidence/metricas_anos.csv", "controle_defasado")}}
    for group in data.values():
        if sorted(r["year"] for r in group["years"]) != list(range(2007, 2021)):
            raise ValueError("Demo needs the complete 2007–2020 evaluation.")
        group["years"].sort(key=lambda r: r["year"])
    event = json.loads(checked["research/prospective/evidence/issued_forecast_2026_10.json"])
    if not event["skill_not_yet_measured"] or event["observed_rainfall_used_for_scoring"]:
        raise ValueError("Review the demo text before changing October's evaluation status.")
    page = (root / "demo/template.html").read_text(encoding="utf-8")
    replacements = {
        "@@DATA@@": json.dumps(data, allow_nan=False).replace("<", "\\u003c"),
        "@@COMPETITION_MAP@@": base64.b64encode(checked["assets/competition_forecast_example.png"]).decode(),
        "@@OCTOBER_MAP@@": base64.b64encode(checked["assets/forecast_october_2026.png"]).decode(),
        "@@ISSUED@@": event["issued_at_utc"],
        "@@VERIFY_FROM@@": event["first_verification_eligible_from"][:10],
        "@@SOURCES@@": "\n".join('<li><code>' + name + '</code></li>' for name in checked),
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    if "@@" in page:
        raise ValueError("Unfilled demo template field.")
    return page


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "demo/index.html")
    parser.add_argument("--check", action="store_true", help="Compare against the existing HTML without writing.")
    args = parser.parse_args(argv)
    try:
        page = build()
        if args.check:
            if args.output.read_bytes() != page.encode("utf-8"):
                raise ValueError("The saved viewer differs; regenerate it from reviewed sources.")
            print("Offline viewer matches its checked sources.")
        else:
            if args.output.suffix.lower() != ".html":
                raise ValueError("Choose an .html output file.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(page, encoding="utf-8", newline="\n")
            print("Open in a browser:", args.output.resolve())
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("Demo stopped:", error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
