"""Contract tests for the thin per-task launcher scripts.

Replaces the per-recipe tests that asserted the previous ``run_all.sh``
launcher contract. The new contract (docs/manual/requirement.md section 5.3) is:

* every script under ``scripts/<dataset>/<experiment>/`` only selects CONFIG and
  TASK and then sources ``tools/launcher/launch_from_config.sh``;
* ``--dry-run`` resolves the config and prints the resolved output directories
  without creating or modifying anything.

Written with ``unittest`` because the vjepa2-312 environment does not ship
pytest.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def script_paths() -> list[Path]:
    return sorted(SCRIPTS_ROOT.rglob("*.sh"))


class ThinLauncherScriptsTest(unittest.TestCase):
    def test_scripts_exist(self):
        self.assertTrue(script_paths(), "no launcher scripts found under scripts/")

    def test_scripts_are_thin(self):
        """A thin script must not hardcode GPU, module, checkpoint or master values."""
        forbidden = ("torchrun", "CUDA_VISIBLE_DEVICES", "MASTER_PORT", "-m src.", "-m recipe")
        for script in script_paths():
            text = script.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{script} hardcodes {token!r}")

    def test_dry_run_is_side_effect_free(self):
        outputs_before = sorted((REPO_ROOT / "outputs").rglob("*"))
        for script in script_paths():
            with self.subTest(script=str(script.relative_to(REPO_ROOT))):
                result = subprocess.run(
                    ["bash", str(script), "--dry-run"],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for field in (
                    "module:",
                    "output_mode:",
                    "large_data_output_dir:",
                    "light_data_output_dir:",
                    "command:",
                ):
                    self.assertIn(field, result.stdout, f"{script} missing {field!r}")
        self.assertEqual(outputs_before, sorted((REPO_ROOT / "outputs").rglob("*")))


if __name__ == "__main__":
    unittest.main()
