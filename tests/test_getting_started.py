"""Contract checks for offline viewing and historical-input preflight."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import shutil
import tempfile
import unittest

from demo.explore import ROOT, build
from scripts.check_inputs import CORE, check


class GettingStartedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value), encoding="utf-8")
        return p

    def inputs(self, profile):
        digest = hashlib.sha256(b"expected bytes").hexdigest()
        self.write("competition/reference.json", {"entradas": [
            {"nome": "treino_tp.nc", "sha256": digest},
            {"nome": "teste_features.nc", "sha256": digest}]})
        for folder, prefix in [("SEAS5", "seas5_51"), ("CFSv2", "cfsv2")]:
            self.write(f"competition/metadata/{folder}/{prefix}_manifesto.json", {
                "arquivos": {phase: {"arquivo": f"{prefix}_{phase}.nc", "sha256": digest}
                             for phase in ["desenvolvimento", "somente_ajuste_final", "teste"]}})
        return self.write("configs/check.json", dict(profile=profile, inputs={
            "official": "../data/official", "seas5": "../data/seas5", "cfsv2": "../data/cfsv2"}))

    def test_missing_files_are_reported_together_without_creating_data(self):
        cfg = self.inputs("competition")
        result = check(cfg, self.root)
        self.assertFalse(result["ready"])
        self.assertEqual(len(result["files"]), 10)
        self.assertTrue(all(r["status"] == "missing" for r in result["files"]))
        self.assertFalse((self.root / "data").exists())
        self.assertEqual(result["paths"]["official"], str((self.root / "data/official").resolve()))

    def test_operational_profile_excludes_future_test_partitions(self):
        cfg = self.inputs("operational-fit")
        rows = check(cfg, self.root)["files"]
        self.assertEqual(len(rows), 7)
        self.assertFalse(any("teste" in r["file"] for r in rows))
        self.assertTrue(any("somente_ajuste_final" in r["file"] for r in rows))

    def test_research_profile_never_requires_final_fit_partition(self):
        cfg = self.inputs("research")
        rows = check(cfg, self.root)["files"]
        self.assertEqual(len(rows), 5)
        self.assertFalse(any("somente_ajuste_final" in r["file"] for r in rows))

    def test_matching_inputs_pass_but_changed_manifest_cannot_approve_changed_data(self):
        cfg = self.inputs("competition")
        rows = check(cfg, self.root)["files"]
        for r in rows:
            p = Path(r["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.suffix == ".json":
                original = next((self.root / "competition/metadata").rglob(p.name))
                shutil.copyfile(original, p)
            else:
                p.write_bytes(b"expected bytes")
        with patch("scripts.check_inputs.importlib.metadata.version", side_effect=CORE.__getitem__):
            self.assertTrue(check(cfg, self.root)["ready"])
            path = self.root / "data/cfsv2/cfsv2_desenvolvimento.nc"
            path.write_bytes(b"changed bytes")
            manifest = self.root / "data/cfsv2/cfsv2_manifesto.json"
            m = json.loads(manifest.read_text())
            m["arquivos"]["desenvolvimento"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(m))
            result = check(cfg, self.root)
        self.assertFalse(result["ready"])
        bad = {r["file"] for r in result["files"] if r["status"] == "hash-mismatch"}
        self.assertEqual(bad, {path.name, manifest.name})

    def test_offline_demo_uses_only_bundled_sources_and_keeps_status_distinct(self):
        manifest = json.loads((ROOT / "demo/sources.json").read_text())
        for name in ["demo/sources.json", "demo/template.html", *manifest["files"]]:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        page = build(self.root)
        self.assertEqual(page, build(ROOT))
        self.assertIn("ISSUED · NOT YET EVALUATED", page)
        self.assertIn("Competition model", page)
        self.assertNotIn('src="http', page)
        self.assertNotIn("@@", page)
        path = self.root / "competition/results/global.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Demo evidence changed"):
            build(self.root)


if __name__ == "__main__":
    unittest.main()
