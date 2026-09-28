"""Output safety checks for staged Physion++ jobs."""
import tempfile
import unittest
from pathlib import Path

from src.experiments.awm_physionpp_fullpatch_probe_seed239.output_guard import ensure_no_artifacts


class OutputGuardTest(unittest.TestCase):
    def test_launcher_bookkeeping_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("checkpoints", "configs", "logs"):
                (output / name).mkdir()
            for name in ("manifest.json", "environment.json", "command.txt"):
                (output / name).touch()
            ensure_no_artifacts(output)

    def test_existing_cache_and_probe_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "manifest.json").touch()
            for name in ("sample_000001.pt", "best.pt", "cache_manifest.json", "other.txt"):
                artifact = output / name
                artifact.touch()
                with self.assertRaisesRegex(FileExistsError, name):
                    ensure_no_artifacts(output)
                artifact.unlink()


if __name__ == "__main__":
    unittest.main()
