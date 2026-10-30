"""Synthetic checks for the standalone splitter; no recording inputs required.

Run directly with the repository's designated Python environment.
"""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "preprocessing/identify_and_separate_sessions.py"
spec = importlib.util.spec_from_file_location("session_splitter", SCRIPT)
splitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitter)


class SessionSplitterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "pipeline with spaces"
        self.adc = self.root / "OneBox-ADC"
        self.probe = self.root / "ProbeA"
        self.ks = self.probe / "kilosort"
        self.adc.mkdir(parents=True)
        self.ks.mkdir(parents=True)
        np.save(self.adc / "periods_samples.npy", np.array([[0, 100], [100, 200]]))
        np.save(self.adc / "periods_timestamps.npy", np.array([[10., 11.], [11., 12.]]))
        np.save(self.probe / "adc_spike_times.npy",
                np.array([9.9, 10., 10.5, 11., 11.9, 12., np.nan]))
        np.save(self.ks / "spike_times.npy", np.arange(7, dtype=np.int64) * 100)
        np.save(self.ks / "spike_clusters.npy", np.array([5, 7, 7, 5, 7, 8, 9]))
        np.save(self.ks / "amplitudes.npy", np.arange(7, dtype=float)[:, None])
        (self.ks / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n5\tgood\n7\tgood\n")

    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            splitter.cli(["--adc-dir", str(self.adc), *args])
        return output.getvalue()

    def test_cli_preserves_export_layout_times_and_joint_sort_identity(self):
        self.run_cli("--session-names", "free moving", "RF mapping")
        for number, rows in ((1, [1, 2]), (2, [3, 4])):
            session = self.root / "sessions" / f"session_{number:02d}"
            neural = session / "neural/ProbeA"
            np.testing.assert_array_equal(np.load(neural / "original_spike_indices.npy"), rows)
            np.testing.assert_array_equal(np.load(neural / "spike_times_global.npy"), np.array(rows) * 100)
            np.testing.assert_array_equal(np.load(neural / "amplitudes.npy"), np.array(rows)[:, None])
            np.testing.assert_array_equal(np.load(neural / "spike_clusters.npy"),
                                          np.array([5, 7, 7, 5, 7, 8, 9])[rows])
            np.testing.assert_allclose(np.load(neural / "adc_spike_times.npy"),
                                       [0., 0.5] if number == 1 else [0., 0.9])
            self.assertEqual((neural / "cluster_KSLabel.tsv").read_bytes(),
                             (self.ks / "cluster_KSLabel.tsv").read_bytes())
            self.assertTrue((session / "stimulus").is_dir())
            with np.load(session / "session_info.npz", allow_pickle=False) as info:
                self.assertEqual(info["session_id"].item(), number - 1)
                self.assertEqual(info["time_reference"].item(), "session_local_adc_seconds")
                self.assertEqual(info["session_name"].item(),
                                 "free moving" if number == 1 else "RF mapping")
        with np.load(self.probe / "session_assignments/spike_session_assignments.npz") as info:
            np.testing.assert_array_equal(info["spike_session_id"], [-1, 0, 0, 1, 1, -1, -1])
            np.testing.assert_array_equal(info["unassigned_original_spike_indices"], [0, 5, 6])

    def test_explicit_pipeline_directory_and_generic_names(self):
        moved_adc = Path(self.temp.name) / "separate timing inputs"
        self.adc.rename(moved_adc)
        self.adc = moved_adc
        self.run_cli("--pipeline-dir", str(self.root))
        with np.load(self.root / "sessions/session_01/session_info.npz") as info:
            self.assertEqual(info["session_name"].item(), "session_01")
        self.assertFalse((moved_adc.parent / "sessions").exists())

    def test_dry_run_writes_nothing(self):
        before = {p.relative_to(self.root): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        output = self.run_cli("--dry-run")
        after = {p.relative_to(self.root): p.read_bytes()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.root / "sessions").exists())
        self.assertFalse((self.probe / "session_assignments").exists())
        self.assertIn("Unassigned: 3", output)

    def test_no_arguments_use_editable_settings(self):
        with patch.multiple(splitter, ADC_DIR=self.adc, PIPELINE_DIR=self.root,
                            SESSION_NAMES=["first", "second"], DRY_RUN=True):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                splitter.cli([])
        self.assertIn("first", output.getvalue())
        self.assertIn("Dry run complete", output.getvalue())
        self.assertFalse((self.root / "sessions").exists())

    def test_editable_settings_can_export(self):
        with patch.multiple(splitter, ADC_DIR=self.adc, PIPELINE_DIR=self.root,
                            SESSION_NAMES=None, DRY_RUN=False):
            with contextlib.redirect_stdout(io.StringIO()):
                splitter.cli([])
        self.assertTrue((self.root / "sessions/session_01/session_info.npz").is_file())

    def test_invalid_names_fail_before_export(self):
        with self.assertRaises(ValueError):
            self.run_cli("--session-names", "only one")
        self.assertFalse((self.root / "sessions").exists())

    def test_mismatched_spike_rows_fail_even_in_dry_run(self):
        np.save(self.ks / "spike_clusters.npy", np.array([7]))
        with self.assertRaisesRegex(ValueError, "Spike count mismatch"):
            self.run_cli("--dry-run")
        self.assertFalse((self.root / "sessions").exists())

    def test_shank_output_layout(self):
        self.ks.rename(self.probe / "temporary-sort")
        self.ks.mkdir()
        (self.probe / "temporary-sort").rename(self.ks / "shank_2")
        (self.probe / "adc_spike_times.npy").rename(self.probe / "adc_spike_times_2.npy")
        self.run_cli()
        self.assertTrue((self.root / "sessions/session_01/neural/ProbeA/shank_2/spike_clusters.npy").is_file())
        self.assertTrue((self.probe / "session_assignments/spike_session_assignments_shank_2.npz").is_file())


if __name__ == "__main__":
    unittest.main()
