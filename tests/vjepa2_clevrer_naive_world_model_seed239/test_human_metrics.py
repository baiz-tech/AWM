import csv
import math
import tempfile
import unittest
from pathlib import Path

import torch

from src.data.physionpp.evaluate.eval_ocp import load_human_accuracy, human_subset_metrics, stimulus_id


class HumanMetricsTest(unittest.TestCase):
    def test_weighting_and_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = ["stim_ID", "correct", "c", "target_hit_zone_label"]
            rows = (
                ("human_accuracy-a.csv", [["family_scenario_0000_img", 1.0, 1, "True"], ["family_scenario_0001_img", 0.0, 2, "False"]]),
                ("human_accuracy-b.csv", [["family_scenario_0000_img", 0.5, 2, "True"], ["family_scenario_0001_img", 1.0, 1, "False"]]),
            )
            for name, values in rows:
                with (root / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    writer.writerows(values)
            human = load_human_accuracy(root)

        self.assertEqual(human["family_scenario_0000_img"]["accuracy"], 2.0 / 3.0)
        self.assertEqual(human["family_scenario_0001_img"]["accuracy"], 1.0 / 3.0)
        paths = ["/root/family/scenario/0000_img.mp4", "/root/family/scenario/0001_img.mp4"]
        self.assertEqual(stimulus_id(paths[0]), "family_scenario_0000_img")
        metrics, accuracy, labels = human_subset_metrics(
            torch.tensor([0.9, 0.1]), torch.tensor([0, 0]), paths, 0.5, human, 0.5,
        )
        self.assertEqual(metrics["human_all_accuracy"], 1.0)
        self.assertEqual(metrics["human_hard_accuracy"], 1.0)
        self.assertEqual(metrics["human_all_count"], 2)
        self.assertEqual(metrics["human_hard_count"], 1)
        self.assertEqual(metrics["human_label_disagreements_with_final_frame"], 1)
        self.assertTrue(all(math.isclose(actual, expected, rel_tol=1.0e-6) for actual, expected in zip(
            accuracy.tolist(), [2.0 / 3.0, 1.0 / 3.0]
        )))
        self.assertEqual(labels.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
