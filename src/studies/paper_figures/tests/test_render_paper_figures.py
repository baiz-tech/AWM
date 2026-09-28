#!/usr/bin/env python3
"""Tests for the paper-figure renderers.

Run with the standard library runner (pytest is not installed)::

    cd makeup
    python -m unittest src.studies.paper_figures.tests.test_render_paper_figures -v

The PDF-writing tests need matplotlib; they are skipped (with an explicit
reason) when the active interpreter cannot import it.  The schema-validation
tests never touch matplotlib, so they always run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[4]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.studies.paper_figures import (  # noqa: E402  (path bootstrap must run first)
    render_all,
    render_entity_intervention,
    render_interaction_prediction,
)


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


MPL_AVAILABLE = _matplotlib_available()
MPL_REASON = (
    "matplotlib is not importable from the active interpreter; "
    "install it with `python -m pip install matplotlib`"
)

ENTITY_SCORES = {
    "visual_only": (0.6100, 0.6000, 0.6500),
    "visual_probe": (0.7100, 0.7000, 0.7900),
    "target_mask": (0.6300, 0.6200, 0.7300),
    "irrelevant_mask": (0.7000, 0.6900, 0.7800),
    "random_slot": (0.6900, 0.6800, 0.7700),
}

INTERACTION_SCORES = {
    "Entity": (0.8100, 0.2000),
    "Dynamic": (0.8300, 0.1800),
    "Relation-only": (0.9673, 0.1054),
    "Entity+Dynamic": (0.8600, 0.1700),
    "Entity+Relation": (0.9650, 0.1060),
    "Dynamic+Relation": (0.9660, 0.1058),
    "Full": (0.9680, 0.1049),
}


def entity_payload() -> dict:
    return {
        "protocol": "physionpp3_ocp_intervention_v1",
        "split": "readout_data_v1",
        "samples": 8,
        "metrics": {
            mode: {
                "samples": 8,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "auroc": auroc,
            }
            for mode, (accuracy, balanced, auroc) in ENTITY_SCORES.items()
        },
        "paired_delta_vs_visual_probe": {
            "target_mask": {"mean_logit_delta": -0.31, "mean_abs_logit_delta": 0.55}
        },
    }


def interaction_payload() -> dict:
    return {
        "protocol": "clevrer_relation_controls_v1",
        "train_pairs": 40,
        "validation_pairs": 16,
        "checkpoint": "/tmp/synthetic/structured_probe/best.pt",
        "metrics": {
            variant: {
                "contact": {"auroc": auroc, "positive_rate": 0.4, "samples": 16},
                "first_contact": {"accuracy": 0.30, "samples": 16, "classes": 16},
                "ttc": {"mae": mae, "samples": 16},
                "feature_dim": 32,
            }
            for variant, (auroc, mae) in INTERACTION_SCORES.items()
        },
    }


class PaperFigureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def run_module(self, module: str, *arguments: str) -> subprocess.CompletedProcess:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["MPLCONFIGDIR"] = str(self.tmp / "mplconfig")
        return subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=str(PACKAGE_ROOT),
            env=environment,
            capture_output=True,
            text=True,
        )

    def assert_pdf(self, path: Path) -> None:
        self.assertTrue(path.is_file(), f"missing output: {path}")
        self.assertGreater(path.stat().st_size, 0, f"empty output: {path}")
        self.assertEqual(path.read_bytes()[:4], b"%PDF", f"not a PDF: {path}")


class TestEntityInterventionValidation(PaperFigureTestCase):
    def test_condition_order_follows_mode_order(self) -> None:
        payload = entity_payload()
        payload["metrics"] = {mode: payload["metrics"][mode] for mode in ("target_mask", "visual_probe")}
        path = self.write_json("entity_subset.json", payload)
        conditions, scores = render_entity_intervention.load_condition_scores(path)
        self.assertEqual(conditions, ["visual_probe", "target_mask"])
        self.assertAlmostEqual(scores["target_mask"]["auroc"], 0.7300)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            render_entity_intervention.render(self.tmp / "absent.json", self.tmp / "out.pdf")

    def test_missing_metrics_object_raises(self) -> None:
        path = self.write_json("entity_no_metrics.json", {"protocol": "physionpp3_ocp_intervention_v1"})
        with self.assertRaises(KeyError):
            render_entity_intervention.load_condition_scores(path)

    def test_missing_score_key_raises(self) -> None:
        payload = entity_payload()
        del payload["metrics"]["target_mask"]["auroc"]
        path = self.write_json("entity_missing_key.json", payload)
        with self.assertRaises(KeyError):
            render_entity_intervention.load_condition_scores(path)
        with self.assertRaises(KeyError):
            render_entity_intervention.render(path, self.tmp / "out.pdf")

    def test_non_numeric_score_raises(self) -> None:
        payload = entity_payload()
        payload["metrics"]["visual_only"]["auroc"] = None
        path = self.write_json("entity_null_auroc.json", payload)
        with self.assertRaises(ValueError):
            render_entity_intervention.load_condition_scores(path)

    def test_unknown_conditions_raise(self) -> None:
        path = self.write_json(
            "entity_unknown.json",
            {"metrics": {"bogus_condition": {"accuracy": 0.5, "balanced_accuracy": 0.5, "auroc": 0.5}}},
        )
        with self.assertRaises(KeyError):
            render_entity_intervention.load_condition_scores(path)


class TestInteractionPredictionValidation(PaperFigureTestCase):
    def test_variant_order_follows_known_variants(self) -> None:
        payload = interaction_payload()
        payload["metrics"] = {name: payload["metrics"][name] for name in ("Full", "Entity")}
        path = self.write_json("interaction_subset.json", payload)
        variants, scores = render_interaction_prediction.load_variant_scores(path)
        self.assertEqual(variants, ["Entity", "Full"])
        self.assertAlmostEqual(scores["Full"]["ttc.mae"], 0.1049)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            render_interaction_prediction.render(self.tmp / "absent.json", self.tmp / "out.pdf")

    def test_missing_metrics_object_raises(self) -> None:
        path = self.write_json("interaction_no_metrics.json", {"protocol": "clevrer_relation_controls_v1"})
        with self.assertRaises(KeyError):
            render_interaction_prediction.load_variant_scores(path)

    def test_missing_ttc_key_raises(self) -> None:
        payload = interaction_payload()
        del payload["metrics"]["Relation-only"]["ttc"]["mae"]
        path = self.write_json("interaction_missing_ttc.json", payload)
        with self.assertRaises(KeyError):
            render_interaction_prediction.load_variant_scores(path)

    def test_missing_contact_section_raises(self) -> None:
        payload = interaction_payload()
        del payload["metrics"]["Full"]["contact"]
        path = self.write_json("interaction_missing_contact.json", payload)
        with self.assertRaises(KeyError):
            render_interaction_prediction.load_variant_scores(path)

    def test_non_numeric_mae_raises(self) -> None:
        payload = interaction_payload()
        payload["metrics"]["Dynamic"]["ttc"]["mae"] = None
        path = self.write_json("interaction_null_mae.json", payload)
        with self.assertRaises(ValueError):
            render_interaction_prediction.load_variant_scores(path)

    def test_unknown_variants_raise(self) -> None:
        path = self.write_json(
            "interaction_unknown.json",
            {"metrics": {"Bogus": {"contact": {"auroc": 0.5}, "ttc": {"mae": 0.1}}}},
        )
        with self.assertRaises(KeyError):
            render_interaction_prediction.load_variant_scores(path)


class TestCliFailures(PaperFigureTestCase):
    def test_entity_intervention_cli_missing_key_exits_nonzero(self) -> None:
        payload = entity_payload()
        del payload["metrics"]["irrelevant_mask"]["balanced_accuracy"]
        path = self.write_json("entity_cli_missing_key.json", payload)
        result = self.run_module(
            "src.studies.paper_figures.render_entity_intervention",
            "--metrics",
            str(path),
            "--output",
            str(self.tmp / "out.pdf"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("balanced_accuracy", result.stderr)
        self.assertFalse((self.tmp / "out.pdf").exists())

    def test_render_all_missing_input_raises(self) -> None:
        interaction = self.write_json("interaction_for_all.json", interaction_payload())
        argv = [
            "render_all",
            "--entity-intervention-metrics",
            str(self.tmp / "absent_entity.json"),
            "--interaction-metrics",
            str(interaction),
            "--output-dir",
            str(self.tmp / "figures"),
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(FileNotFoundError):
                render_all.main()


@unittest.skipUnless(MPL_AVAILABLE, MPL_REASON)
class TestRenderedPdfs(PaperFigureTestCase):
    def test_entity_intervention_writes_pdf(self) -> None:
        path = self.write_json("entity.json", entity_payload())
        output = self.tmp / "entity_intervention.pdf"
        result = self.run_module(
            "src.studies.paper_figures.render_entity_intervention",
            "--metrics",
            str(path),
            "--output",
            str(output),
            "--title",
            "Entity intervention",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_pdf(output)

    def test_interaction_prediction_writes_pdf(self) -> None:
        path = self.write_json("interaction.json", interaction_payload())
        output = self.tmp / "interaction_prediction.pdf"
        result = self.run_module(
            "src.studies.paper_figures.render_interaction_prediction",
            "--metrics",
            str(path),
            "--output",
            str(output),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_pdf(output)

    def test_render_all_writes_both_pdfs(self) -> None:
        entity = self.write_json("entity_all.json", entity_payload())
        interaction = self.write_json("interaction_all.json", interaction_payload())
        output_dir = self.tmp / "figures"
        result = self.run_module(
            "src.studies.paper_figures.render_all",
            "--entity-intervention-metrics",
            str(entity),
            "--interaction-metrics",
            str(interaction),
            "--output-dir",
            str(output_dir),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_pdf(output_dir / render_entity_intervention.DEFAULT_OUTPUT_NAME)
        self.assert_pdf(output_dir / render_interaction_prediction.DEFAULT_OUTPUT_NAME)


if __name__ == "__main__":
    unittest.main()
