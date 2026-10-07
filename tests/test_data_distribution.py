"""Data integrity, safe extraction and reproduction entry-point contracts."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import tempfile
import unittest
import zipfile

from scripts.data_io import destination, install_zip
from scripts.download_data import download
from scripts.reproduce_competition import main as reproduce


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "data"

    def archive(self, values, recorded=None):
        archive = self.root / "snapshot.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name, data in values.items():
                z.writestr(name, data)
        files = [dict(path=n, bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
                 for n, b in (recorded or values).items()]
        row = dict(archive=archive.name, bytes=archive.stat().st_size,
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=files, url="https://invalid.example/snapshot.zip")
        return archive, row

    def test_install_and_cached_second_run_need_no_network(self):
        _, row = self.archive({"seas5/a.nc": b"one", "seas5/NOTICE.txt": b"attribution"})
        self.assertEqual(download(row, self.target, self.root), "installed and verified")
        with patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network")):
            self.assertEqual(download(row, self.target), "already verified (no download)")

    def test_wrong_archive_hash_is_rejected_before_install(self):
        archive, row = self.archive({"sst/a.nc": b"one"})
        archive.write_bytes(archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Archive hash"):
            download(row, self.target, self.root)
        self.assertFalse(self.target.exists())

    def test_all_members_checked_before_any_destination_write(self):
        archive, row = self.archive({"sst/a.nc": b"one", "sst/b.nc": b"bad"},
                                    {"sst/a.nc": b"one", "sst/b.nc": b"two"})
        with self.assertRaisesRegex(ValueError, "Member hash"):
            install_zip(archive, self.target, row["files"])
        self.assertFalse(self.target.exists())

    def test_existing_different_file_is_not_overwritten(self):
        archive, row = self.archive({"cfsv2/a.nc": b"one"})
        path = self.target / "cfsv2/a.nc"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"mine")
        with self.assertRaisesRegex(ValueError, "preserved"):
            install_zip(archive, self.target, row["files"])
        self.assertEqual(path.read_bytes(), b"mine")

    def test_extra_members_and_traversal_are_rejected(self):
        archive, row = self.archive({"a.nc": b"one", "../outside": b"bad"}, {"a.nc": b"one"})
        with self.assertRaisesRegex(ValueError, "members differ"):
            install_zip(archive, self.target, row["files"])
        for name in ["../outside", "/absolute", "C:/absolute", "x\\y", "a/../b", "./x"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                destination(self.target, name)

    def test_missing_official_input_cannot_start_training(self):
        result = dict(profile="competition", ready=False, paths={})
        with patch("scripts.reproduce_competition.check", return_value=result), patch("subprocess.run") as run:
            self.assertEqual(reproduce(["--config", "unused"]), 2)
            run.assert_not_called()

    def test_launcher_passes_checked_paths_without_enabling_validation(self):
        paths = {k: str(self.root / k) for k in ["official", "seas5", "cfsv2"]}
        result = dict(profile="competition", ready=True, paths=paths)
        with patch("scripts.reproduce_competition.check", return_value=result), patch("subprocess.run") as run:
            run.return_value.returncode = 0
            self.assertEqual(reproduce(["--output", str(self.root / "run")]), 0)
            env = run.call_args.kwargs["env"]
            self.assertEqual(env["WORCAP_OFFICIAL_DIR"], paths["official"])
            self.assertEqual(env["WORCAP_REVALIDATE"], "0")

    def test_catalog_preserves_recorded_scientific_hashes(self):
        root = Path(__file__).resolve().parents[1]
        catalog = json.loads((root / "configs/data_snapshots.json").read_text())
        for source, folder, prefix in [("seas5", "SEAS5", "seas5_51"), ("cfsv2", "CFSv2", "cfsv2")]:
            path = root / "competition/metadata" / folder / f"{prefix}_manifesto.json"
            manifest = json.loads(path.read_text())
            rows = {Path(r["path"]).name: r["sha256"] for r in catalog["snapshots"][source]["files"]}
            self.assertEqual(rows[path.name], hashlib.sha256(path.read_bytes()).hexdigest())
            for entry in manifest["arquivos"].values():
                self.assertEqual(rows[entry["arquivo"]], entry["sha256"])
        protocol = json.loads((root / "research/lagged_sources/protocol.json").read_text())
        sst = next(r for r in catalog["snapshots"]["sst"]["files"] if r["path"].endswith(".nc"))
        self.assertEqual(sst["sha256"], protocol["hash_sst"])


if __name__ == "__main__":
    unittest.main()
