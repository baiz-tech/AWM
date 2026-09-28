import unittest
import json
import tempfile
from pathlib import Path

import yaml

from src.data.clevrer.v1.qa_dataset import load_clevrer_annotations
from src.data.clevrer.v1.qa_pipeline import (
    _archive_existing_paths,
    _trajectory_export_complete,
)
from src.data.clevrer.v1.qa_utils import make_submission
from tools.launcher.resolve_run import resolve
from tools.launcher.build_launch_command import _template_context


RECIPE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = RECIPE_DIR / "scripts"
SHARED_EVAL_DIR = RECIPE_DIR.parent / "shared" / "evaluate"
SHARED_TOOLS_DIR = RECIPE_DIR.parent / "shared" / "tools"
SHARED_LAUNCHER_DIR = SHARED_TOOLS_DIR / "launcher"


class ScriptStructureTest(unittest.TestCase):
    def test_entries_are_direct_and_method_local(self):
        expected = {
            "clevrer/stride2/qa_eval_8gpu.sh",
            "clevrer/stride2/train_8gpu.sh",
            "clevrer/v2/qa_eval_8gpu.sh",
            "intphys2/physion_stride2_4/eval_surprise_8gpu.sh",
            "physionpp/gap32/train_8gpu.sh",
            "physionpp/gap32/eval_temporal_retrieval_8gpu.sh",
            "physionpp/no_gap/train_4gpu.sh",
            "physionpp/no_gap/train_8gpu.sh",
            "physionpp/no_gap/train_fair_rerun_8gpu.sh",
            "physionpp/no_gap/eval_future_prediction_8gpu.sh",
            "physionpp/no_gap/eval_full_future_ocp.sh",
            "physionpp/no_gap/eval_single_future_ocp_8gpu.sh",
            "physionpp/no_gap/eval_single_future_all_future_ocp_8gpu.sh",
            "physionpp/no_gap/eval_temporal_retrieval_8gpu.sh",
        }
        scripts = sorted(SCRIPTS_DIR.rglob("*.sh"))
        self.assertEqual(
            {path.relative_to(SCRIPTS_DIR).as_posix() for path in scripts}, expected
        )
        for path in scripts:
            source = path.read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", source)
            self.assertIn("src/experiments/vjepa2_clevrer_naive_world_model_seed239/configs/", source)
            self.assertIn("TASK=", source)
            self.assertIn("source tools/launcher/launch_from_config.sh", source)
            self.assertNotIn("REPO_ROOT=", source)
            self.assertNotIn("PYTHON_BIN", source)
            self.assertNotIn("torch.distributed.run", source)
            self.assertNotIn("src.shared.evaluate", source)
            self.assertNotIn("src.experiments.vjepa2_clevrer_naive_world_model_seed239.train", source)
            self.assertNotIn("launcher.py", source)
            self.assertNotIn("recipe.vjepa2_ef", source)

    def test_intphys2_eval_config_uses_shared_entrypoint(self):
        path = (
            RECIPE_DIR
            / "configs"
            / "intphys2"
            / "physion-vith-16to16-intphys2-surprise-8gpu.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        task = config["tasks"]["intphys2_surprise"]
        self.assertEqual(
            task["launch"]["module"],
            "src.shared.evaluate.intphys2.eval_surprise",
        )
        self.assertTrue(task["launch"]["distributed"])
        self.assertEqual(
            task["experiment"]["evaluation"]["world_model_adapter"],
            "src.experiments.vjepa2_clevrer_naive_world_model_seed239.world_model_adapter",
        )
        self.assertEqual(task["experiment"]["intphys2"]["split"], "Main")
        resolved = resolve(config, "intphys2_surprise")
        self.assertEqual(
            resolved["WORKSPACE_OUTPUT_DIR_RESOLVED"],
            "outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/"
            "evaluations/intphys2_main_surprise",
        )

    def test_fresh_outputs_are_archived_after_live_pid_check(self):
        common = (SHARED_LAUNCHER_DIR / "launcher_common.sh").read_text(encoding="utf-8")
        self.assertIn("/legacy", common)
        self.assertIn('mv "$source_dir" "$archive_dir"', common)
        self.assertLess(
            common.index('for pid_file in "${pid_files[@]}"'),
            common.index('for output_dir in "${output_dirs[@]}"'),
        )
        self.assertIn("LAUNCH_ARCHIVE_PRESERVE_SUBPATHS", common)
        self.assertIn("launcher_restore_preserved_subpaths", common)
        train_entries = [
            *sorted(SCRIPTS_DIR.rglob("train_*.sh")),
        ]
        for path in train_entries:
            source = path.read_text(encoding="utf-8")
            self.assertIn("launch_from_config.sh", source)
            self.assertNotRegex(source, r'MASTER_PORT="\$\{MASTER_PORT:-[0-9]')
            self.assertNotRegex(source, r'NUM_GPUS="\$\{NUM_GPUS:-[0-9]')
            self.assertNotRegex(source, r'CUDA_VISIBLE_DEVICES="\$\{CUDA_VISIBLE_DEVICES:-0')

        for path in sorted(SCRIPTS_DIR.rglob("*.sh")):
            if path.name.startswith("train_"):
                continue
            if path.name.startswith("eval_") or path.name == "qa_eval_8gpu.sh":
                source = path.read_text(encoding="utf-8")
                self.assertIn("launch_from_config.sh", source)
                continue
            source = path.read_text(encoding="utf-8")
            self.assertIn("/legacy", source)
            self.assertIn('mv "$OUTPUT_DIR" "$ARCHIVE_DIR"', source)
            self.assertLess(
                source.index('kill -0 "$OLD_PID"'),
                source.index('mv "$OUTPUT_DIR" "$ARCHIVE_DIR"'),
            )

    def test_training_resume_preserves_output_directory(self):
        train_common = (SHARED_LAUNCHER_DIR / "launch_from_config.sh").read_text(encoding="utf-8")
        shared_common = (SHARED_LAUNCHER_DIR / "launcher_common.sh").read_text(encoding="utf-8")
        self.assertIn('"$ARG" == "--resume"', train_common)
        self.assertIn('"$ARG" == --resume=*', train_common)
        self.assertIn("Resume requested; preserving output directory", shared_common)
        for path in sorted(SCRIPTS_DIR.rglob("train_*.sh")):
            source = path.read_text(encoding="utf-8")
            self.assertIn("launch_from_config.sh", source)

    def test_validation_bypasses_ddp_forward_for_exact_shards(self):
        source = (RECIPE_DIR / "train.py").read_text(encoding="utf-8")
        self.assertIn("eval_loader,\n            module,", source)

    def test_training_launcher_mirrors_required_outputs(self):
        shared = (SHARED_LAUNCHER_DIR / "launcher_common.sh").read_text(encoding="utf-8")
        self.assertNotIn("OUTPUT_ROOT=\"${OUTPUT_ROOT:-/data/shyang/outputs}\"", shared)
        self.assertNotIn("PROJECT_NAME=\"${PROJECT_NAME:-$(basename \"$REPO_ROOT\")}\"", shared)
        self.assertIn('tee -a "${logs[@]}" >/dev/null', shared)
        self.assertIn('for pid_file in "${pid_files[@]}"', shared)
        self.assertIn("Workspace output:", shared)
        self.assertIn("Data output:", shared)

        source = (SHARED_LAUNCHER_DIR / "launch_from_config.sh").read_text(encoding="utf-8")
        self.assertIn("tools/launcher/launcher_common.sh", source)
        self.assertIn("tools.launcher.resolve_run", source)
        self.assertIn("tools.launcher.build_launch_command", source)
        self.assertIn("DATA_RUN_DIR=", source)
        self.assertIn("launcher_run_background", source)
        self.assertNotIn("torch.distributed.run", source)
        self.assertNotIn("src.shared.evaluate", source)
        self.assertNotIn("src.experiments.vjepa2_clevrer_naive_world_model_seed239.train", source)
        builder = (SHARED_LAUNCHER_DIR / "build_launch_command.py").read_text(encoding="utf-8")
        self.assertIn("preserve_on_archive", builder)
        self.assertIn("LAUNCH_ARCHIVE_PRESERVE_SUBPATHS", builder)

    def test_eval_scripts_live_in_experiment_and_call_shared_code(self):
        common = (SHARED_LAUNCHER_DIR / "launcher_common.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("LAUNCH_WORKSPACE_DIR", common)
        self.assertIn("LAUNCH_DATA_DIR", common)
        self.assertIn('tee -a "${logs[@]}" >/dev/null', common)
        self.assertIn('cp -a "$LAUNCH_DATA_DIR"/. "$LAUNCH_WORKSPACE_DIR"/', common)
        self.assertIn("launcher_mirror_lightweight_outputs", common)
        self.assertIn('for pid_file in "${pid_files[@]}"', common)
        self.assertIn("Workspace output:", common)
        self.assertIn("Data output:", common)

        self.assertFalse(
            list(SHARED_EVAL_DIR.rglob("*.sh")),
            "shared/evaluate should contain implementation code only; shell entrypoints live under src/experiments/vjepa2_clevrer_naive_world_model_seed239/scripts",
        )
        for relative_path in (
            "physionpp/no_gap/eval_future_prediction_8gpu.sh",
            "physionpp/no_gap/eval_full_future_ocp.sh",
            "physionpp/no_gap/eval_single_future_ocp_8gpu.sh",
            "physionpp/no_gap/eval_single_future_all_future_ocp_8gpu.sh",
            "physionpp/no_gap/eval_temporal_retrieval_8gpu.sh",
            "physionpp/gap32/eval_temporal_retrieval_8gpu.sh",
        ):
            source = (SCRIPTS_DIR / relative_path).read_text(encoding="utf-8")
            self.assertIn("source tools/launcher/launch_from_config.sh", source)
            self.assertIn("OUTPUT_MODE=${OUTPUT_MODE:-workspace}", source)
            self.assertIn("WORKSPACE_MIRROR_MODE=${WORKSPACE_MIRROR_MODE:-lightweight}", source)
            self.assertNotIn('OUTPUT_ROOT="${OUTPUT_ROOT:-/data/shyang/outputs}"', source)
            self.assertNotRegex(source, r'MASTER_PORT="\$\{MASTER_PORT:-[0-9]')
            self.assertNotRegex(source, r'NUM_GPUS="\$\{NUM_GPUS:-[0-9]')
            self.assertNotRegex(source, r'CUDA_VISIBLE_DEVICES="\$\{CUDA_VISIBLE_DEVICES:-0')

    def test_output_mode_selects_one_or_both_path_sets(self):
        config_path = RECIPE_DIR / "configs/physionpp/physion-vith-16to16-8gpu.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        resolved = resolve(config, "single_future_ocp")

        workspace = _template_context(
            str(config_path), "single_future_ocp", resolved, "python", "workspace"
        )
        self.assertEqual(workspace["data_run_dir"], workspace["workspace_run_dir"])
        self.assertEqual(workspace["data_output_dir"], workspace["workspace_output_dir"])
        self.assertTrue(workspace["checkpoint"].startswith("outputs/runs/"))

        data = _template_context(
            str(config_path), "single_future_ocp", resolved, "python", "data"
        )
        self.assertEqual(data["workspace_run_dir"], data["data_run_dir"])
        self.assertEqual(data["workspace_output_dir"], data["data_output_dir"])
        self.assertTrue(data["checkpoint"].startswith("/data/shyang/outputs/"))

        both = _template_context(
            str(config_path), "single_future_ocp", resolved, "python", "both"
        )
        self.assertNotEqual(both["workspace_run_dir"], both["data_run_dir"])

    def test_future_prediction_uses_test_split_and_shared_entrypoint(self):
        config = yaml.safe_load(
            (
                RECIPE_DIR
                / "configs/physionpp/physion-vith-16to16-8gpu.yaml"
            ).read_text(encoding="utf-8")
        )
        task = config["tasks"]["future_prediction"]["launch"]
        self.assertEqual(
            task["module"],
            "src.data.physionpp.evaluate.eval_future_prediction",
        )
        self.assertTrue(task["distributed"])
        self.assertEqual(task["launcher"]["gpus"], 8)
        resolved = resolve(config, "future_prediction")
        self.assertTrue(
            resolved["WORKSPACE_OUTPUT_DIR_RESOLVED"].endswith(
                "/evaluations/future_prediction_test"
            )
        )

        clevrer = (
            SCRIPTS_DIR / "clevrer" / "stride2" / "qa_eval_8gpu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("source tools/launcher/launch_from_config.sh", clevrer)
        self.assertNotRegex(clevrer, r'BASE_PORT="\$\{BASE_PORT:-[0-9]')
        self.assertNotRegex(clevrer, r'NUM_GPUS="\$\{NUM_GPUS:-[0-9]')
        self.assertNotRegex(clevrer, r'CUDA_VISIBLE_DEVICES="\$\{CUDA_VISIBLE_DEVICES:-0')

        shared_launcher = (SHARED_LAUNCHER_DIR / "launch_from_config.sh").read_text(encoding="utf-8")
        self.assertIn("tools.launcher.build_launch_command", shared_launcher)
        for forbidden in (
            "src.shared.evaluate.physionpp",
            "src.shared.evaluate.clevrer",
            "full_future_ocp)",
            "same_video_temporal_retrieval)",
            "qa_eval)",
            "qa_${variant}",
            "mirror_json_outputs",
            "launch_physionpp",
            "launch_clevrer",
        ):
            self.assertNotIn(forbidden, shared_launcher)

    def test_temporal_retrieval_uses_saved_config(self):
        for relative_path in (
            "physionpp/no_gap/eval_temporal_retrieval_8gpu.sh",
            "physionpp/gap32/eval_temporal_retrieval_8gpu.sh",
        ):
            source = (SCRIPTS_DIR / relative_path).read_text(encoding="utf-8")
            self.assertIn("TASK=same_video_temporal_retrieval", source)
            self.assertNotIn("config.resolved.yaml", source)

    def test_script_groups_match_clip_gap(self):
        no_gap_train = (SCRIPTS_DIR / "physionpp/no_gap/train_4gpu.sh").read_text(
            encoding="utf-8"
        )
        no_gap_config = RECIPE_DIR / "configs/physionpp/physion-vith-16to16-4gpu.yaml"
        self.assertIn(str(no_gap_config.relative_to(RECIPE_DIR.parents[1])), no_gap_train)
        self.assertIn("clip_gap: 0", no_gap_config.read_text(encoding="utf-8"))

        gap32_train = (SCRIPTS_DIR / "physionpp/gap32/train_8gpu.sh").read_text(
            encoding="utf-8"
        )
        gap32_config = RECIPE_DIR / "configs/physionpp/physion-vith-16to16-gap32-8gpu.yaml"
        self.assertIn(str(gap32_config.relative_to(RECIPE_DIR.parents[1])), gap32_train)
        self.assertIn("clip_gap: 32", gap32_config.read_text(encoding="utf-8"))

    def test_legacy_flat_entries_are_removed(self):
        self.assertFalse((SCRIPTS_DIR / "no_gap").exists())
        self.assertFalse((SCRIPTS_DIR / "gap32").exists())
        self.assertFalse((SCRIPTS_DIR / "launch_fair_rerun_8gpu.sh").exists())
        self.assertFalse((SCRIPTS_DIR / "train_launcher_common.sh").exists())
        self.assertFalse(list((RECIPE_DIR / "configs").glob("physion-vith-16to16-*.yaml")))
        for legacy_name in (
            "launch_from_config.sh",
            "launcher_common.sh",
            "resolve_run.py",
            "build_launch_command.py",
            "config_utils.py",
        ):
            self.assertFalse((SHARED_TOOLS_DIR / legacy_name).exists(), legacy_name)

    def test_dataset_configs_use_explicit_shared_schema(self):
        required = {
            "dataset",
            "root",
            "train_split",
            "eval_split",
            "video_glob",
            "clip_frames",
            "sampling_mode",
            "current_frame_step",
            "future_frame_step",
            "clip_gap",
            "train_num_future_chunks",
            "eval_num_future_chunks",
        }
        configs = [
            *sorted((RECIPE_DIR / "configs" / "physionpp").glob("physion-vith-16to16-*gpu.yaml")),
            *(RECIPE_DIR / "configs" / "clevrer").glob("*.yaml"),
        ]
        self.assertTrue(configs)
        for path in configs:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(set(config), {"launch", "tasks"}, path)
            launch = config["launch"]
            tasks = config["tasks"]
            self.assertIn("run", launch, path)
            self.assertIn("resources", launch, path)
            self.assertIn("outputs", launch, path)
            run = launch["run"]
            self.assertEqual(run["project"], "vjepa2-baiz", path)
            self.assertEqual(run["experiment"], "vjepa2_naive", path)
            self.assertEqual(run["workspace_root"], "outputs/runs", path)
            self.assertEqual(run["data_root"], "/data/shyang/outputs", path)
            self.assertNotIn("output_dir", launch["outputs"], path)
            resources = launch["resources"]
            self.assertIn("gpus", resources, path)
            self.assertIn("master_port", resources, path)
            self.assertNotIn("tasks", launch, path)
            self.assertNotIn("evaluations", launch, path)
            self.assertIn("train", tasks, path)
            train_task = tasks["train"]
            self.assertIn("launch", train_task, path)
            self.assertIn("experiment", train_task, path)
            self.assertEqual(train_task["launch"]["module"], "src.experiments.vjepa2_clevrer_naive_world_model_seed239.train", path)
            self.assertTrue(train_task["launch"]["distributed"], path)
            experiment = train_task["experiment"]
            data = experiment["data"]
            self.assertFalse(required - set(data), path)
            if data["dataset"] == "physionpp":
                self.assertEqual(
                    experiment["evaluation"]["world_model_adapter"],
                    "src.experiments.vjepa2_clevrer_naive_world_model_seed239.world_model_adapter",
                    path,
                )
            temporal_task = tasks.get("same_video_temporal_retrieval")
            if temporal_task is not None:
                temporal_args = temporal_task["launch"]["args"]
                adapter_index = temporal_args.index("--world-model-adapter")
                self.assertEqual(
                    temporal_args[adapter_index + 1],
                    "src.experiments.vjepa2_clevrer_naive_world_model_seed239.world_model_adapter",
                    path,
                )
            for key in ("meta", "protocol", "model", "data", "data_aug", "loss", "optimization", "logging"):
                self.assertIn(key, experiment, path)
        clevrer = yaml.safe_load(
            (RECIPE_DIR / "configs" / "clevrer" / "clevrer-vith-16to16-stride2-8gpu.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(clevrer["tasks"]["train"]["experiment"]["data"]["sampling_mode"], "random")
        self.assertNotIn("qa", clevrer["tasks"]["train"]["experiment"])
        self.assertIn("qa", clevrer["tasks"]["qa_eval"]["experiment"])
        required_qa = {
            "annotation_root",
            "question_types",
            "skip_existing_trajectories",
            "archive_existing_probe_outputs",
            "spatial_pool_grid",
            "export_batch_size",
            "export_num_workers",
            "video_len",
            "n_sample_frames",
            "max_n_objects",
            "max_visual_tokens",
            "max_question_len",
            "max_choice_len",
            "scene_batch_size",
            "train_batch_size",
            "val_batch_size",
            "num_workers",
            "min_token_frequency",
            "input_dim",
            "num_layers",
            "num_heads",
            "ffn_dim",
            "cls_mlp_size",
            "dropout",
            "epochs",
            "learning_rate",
            "weight_decay",
            "grad_clip",
        }
        qa_config = clevrer["tasks"]["qa_eval"]["experiment"]["qa"]
        self.assertFalse(required_qa - set(qa_config))
        self.assertEqual(qa_config["protocol"], "compatibility_stride2")
        self.assertEqual(qa_config["data"]["token_source"], "online")
        self.assertEqual(qa_config["representation"]["type"], "global_tokens")
        self.assertEqual(qa_config["question_types"], ["predictive"])
        self.assertIs(qa_config["skip_existing_trajectories"], True)
        self.assertIs(qa_config["archive_existing_probe_outputs"], True)
        qa_launch = clevrer["tasks"]["qa_eval"]["launch"]
        self.assertEqual(
            qa_launch["module"],
            "src.data.clevrer.v1.qa_pipeline",
        )
        self.assertEqual(qa_launch["output_scope"], "evaluation")
        self.assertNotIn("output_dir", qa_launch)
        self.assertEqual(qa_launch["log_name"], "pipeline.log")
        self.assertEqual(qa_launch["pid_name"], "pipeline.pid")
        self.assertIs(qa_launch["archive_existing"], True)
        self.assertEqual(qa_launch["preserve_on_archive"], ["qa_trajectories"])
        self.assertEqual(clevrer["launch"]["outputs"], {})
        self.assertNotIn("extra_config", clevrer["launch"])
        self.assertNotIn("extra_config", qa_launch)
        self.assertIn("--workspace-output-dir", qa_launch["args"])
        self.assertIn("{workspace_output_dir}", qa_launch["args"])
        self.assertIn("--data-output-dir", qa_launch["args"])
        self.assertIn("{data_output_dir}", qa_launch["args"])
        self.assertNotIn("--trajectory-root", qa_launch["args"])
        self.assertNotIn("--qa-checkpoint-root", qa_launch["args"])
        resolved_qa = resolve(clevrer, "qa_eval")
        self.assertEqual(
            resolved_qa["WORKSPACE_OUTPUT_DIR_RESOLVED"],
            "outputs/runs/vjepa2_naive/clevrer_vith_16to16_stride2_8gpu/evaluations/qa_naive",
        )
        self.assertEqual(
            resolved_qa["DATA_OUTPUT_DIR_RESOLVED"],
            "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/clevrer_vith_16to16_stride2_8gpu/evaluations/qa_naive",
        )
        self.assertNotIn("qa_trajectory_dir", clevrer["launch"]["outputs"])
        self.assertNotIn("qa_checkpoint_dir", clevrer["launch"]["outputs"])
        self.assertNotIn("qa_trajectory_root", clevrer["launch"]["outputs"])
        self.assertNotIn("qa_checkpoint_root", clevrer["launch"]["outputs"])
        self.assertNotIn("trajectory_root", qa_config)
        self.assertNotIn("checkpoint_root", qa_config)

    def test_shared_physion_evaluation_uses_recipe_world_model_adapter(self):
        physion_dir = SHARED_EVAL_DIR / "physionpp"
        loader = (physion_dir / "world_model_adapter.py").read_text(encoding="utf-8")
        self.assertIn('config.get("evaluation", {}).get("world_model_adapter")', loader)
        common_loader = (SHARED_EVAL_DIR / "world_model.py").read_text(encoding="utf-8")
        self.assertIn('getattr(adapter, "load_world_model", None)', common_loader)
        for name in ("eval_ocp.py", "eval_same_video_temporal_retrieval.py"):
            source = (physion_dir / name).read_text(encoding="utf-8")
            self.assertIn("load_world_model", source)
            self.assertNotIn("from src.experiments.vjepa2_clevrer_naive_world_model_seed239", source)
            self.assertNotIn("checkpoint[\"predictor\"]", source)
        ocp = (physion_dir / "eval_ocp.py").read_text(encoding="utf-8")
        retrieval = (physion_dir / "eval_same_video_temporal_retrieval.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("model.predict_next(context)", ocp)
        self.assertIn("model.encode_current(current)", retrieval)
        self.assertIn("model.predict_next(context)", retrieval)
        self.assertIn("model.encode_target(currents, futures)", retrieval)
        self.assertNotIn("model(currents, futures)", retrieval)

    def test_clevrer_qa_question_type_filter_is_predictive_only(self):
        scenes = [
            {
                "scene_index": 0,
                "questions": [
                    {
                        "question_id": 1,
                        "question_type": "descriptive",
                        "question": "What color is it?",
                        "answer": "red",
                    },
                    {
                        "question_id": 2,
                        "question_type": "predictive",
                        "question": "What will happen?",
                        "choices": [
                            {"choice_id": 0, "choice": "yes", "answer": "correct"},
                            {"choice_id": 1, "choice": "no", "answer": "wrong"},
                        ],
                    },
                    {
                        "question_id": 3,
                        "question_type": "counterfactual",
                        "question": "What if?",
                        "choices": [
                            {"choice_id": 0, "choice": "yes", "answer": "wrong"},
                        ],
                    },
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "validation.json").write_text(json.dumps(scenes), encoding="utf-8")
            filtered = load_clevrer_annotations(root, "validation", question_types=["predictive"])
            self.assertEqual(len(filtered), 1)
            self.assertEqual(
                [question["question_type"] for question in filtered[0]["questions"]],
                ["predictive"],
            )
            records = [
                {
                    "scene_index": 0,
                    "question_id": 2,
                    "choice_id": 0,
                    "task": 1,
                    "label": 1,
                    "prediction": 1,
                    "probability": 0.8,
                },
                {
                    "scene_index": 0,
                    "question_id": 2,
                    "choice_id": 1,
                    "task": 1,
                    "label": 0,
                    "prediction": 0,
                    "probability": 0.2,
                },
            ]
            submission = make_submission(
                root / "validation.json",
                records,
                [],
                question_types=["predictive"],
            )
            self.assertEqual(len(submission[0]["questions"]), 1)
            self.assertEqual(submission[0]["questions"][0]["question_id"], 2)

    def test_clevrer_qa_existing_trajectory_skip_requires_manifest_and_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "naive" / "train"
            split_dir.mkdir(parents=True)
            self.assertFalse(_trajectory_export_complete(split_dir))
            (split_dir / "manifest.json").write_text(
                json.dumps({"scenes": 2}), encoding="utf-8"
            )
            self.assertFalse(_trajectory_export_complete(split_dir))
            (split_dir / "scene_00000.pt").write_bytes(b"placeholder")
            self.assertFalse(_trajectory_export_complete(split_dir))
            (split_dir / "scene_00001.pt").write_bytes(b"placeholder")
            self.assertTrue(_trajectory_export_complete(split_dir))
            self.assertTrue(_trajectory_export_complete(split_dir, min_scenes=2))
            self.assertFalse(_trajectory_export_complete(split_dir, min_scenes=3))

    def test_clevrer_qa_archives_probe_outputs_without_touching_trajectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            qa_dir = root / "qa_checkpoints" / "naive"
            eval_dir = root / "evaluations" / "qa_naive_validation"
            log_file = root / "qa_naive_train.log"
            trajectory_file = root / "qa_trajectories" / "naive" / "train" / "scene_00000.pt"
            for path in (qa_dir, eval_dir, trajectory_file.parent):
                path.mkdir(parents=True)
            (qa_dir / "best.pt").write_bytes(b"checkpoint")
            (eval_dir / "metrics.json").write_text("{}", encoding="utf-8")
            log_file.write_text("old log", encoding="utf-8")
            trajectory_file.write_bytes(b"trajectory")

            archived = _archive_existing_paths([qa_dir, eval_dir, log_file])
            self.assertEqual(len(archived), 3)
            self.assertFalse(qa_dir.exists())
            self.assertFalse(eval_dir.exists())
            self.assertFalse(log_file.exists())
            self.assertTrue(trajectory_file.exists())
            self.assertTrue(list((root / "qa_checkpoints" / "legacy").glob("naive_*")))
            self.assertTrue(list((root / "evaluations" / "legacy").glob("qa_naive_validation_*")))
            self.assertTrue(list((root / "legacy").glob("qa_naive_train.log_*")))

    def test_evaluations_default_to_best_checkpoint(self):
        for relative_path in (
            "physionpp/no_gap/eval_temporal_retrieval_8gpu.sh",
            "physionpp/gap32/eval_temporal_retrieval_8gpu.sh",
        ):
            source = (SCRIPTS_DIR / relative_path).read_text(encoding="utf-8")
            self.assertIn("TASK=same_video_temporal_retrieval", source)
            self.assertNotIn("$RUN_DIR/latest.pt", source)

        ocp_config = yaml.safe_load(
            (RECIPE_DIR / "configs/physionpp/physion-vith-16to16-full-future-ocp.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(set(ocp_config), {"launch", "tasks"})
        self.assertEqual(set(ocp_config["tasks"]), {"full_future_ocp"})
        ocp_task = ocp_config["tasks"]["full_future_ocp"]
        self.assertEqual(ocp_task["launch"]["checkpoint"], "best.pt")
        self.assertNotIn("/latest.pt", ocp_task["launch"]["checkpoint"])
        self.assertNotIn("output_dir", ocp_task["launch"])
        evaluation = ocp_task["experiment"]["evaluation"]
        self.assertEqual(evaluation["probe_protocol"], "unified")
        self.assertEqual(evaluation["probe_validation_fraction"], 0.25)
        self.assertEqual(evaluation["probe_variants"], ["full", "current", "rollout_mean", "last_chunk", "delta"])
        resolved_ocp = resolve(ocp_config, "full_future_ocp")
        self.assertEqual(
            resolved_ocp["WORKSPACE_OUTPUT_DIR_RESOLVED"],
            "outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/evaluations/full_future_ocp",
        )
        self.assertEqual(
            resolved_ocp["DATA_OUTPUT_DIR_RESOLVED"],
            "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/evaluations/full_future_ocp",
        )

        train_source = (RECIPE_DIR / "train.py").read_text(encoding="utf-8")
        self.assertIn("workspace_run_dir =", train_source)
        self.assertIn("data_run_dir =", train_source)
        self.assertIn('data_run_dir / "best.pt"', train_source)
        self.assertIn('data_run_dir / "latest.pt"', train_source)
        self.assertNotIn("mirror_path", train_source)
        self.assertIn('payload.get("best_validation_loss"', train_source)
        self.assertIn('payload.get("best_epoch"', train_source)

        shared_launcher = (SHARED_LAUNCHER_DIR / "launch_from_config.sh").read_text(encoding="utf-8")
        self.assertNotIn('TRAIN_RUN_CONFIG="${TRAIN_RUN_CONFIG:-$WORKSPACE_RUN_DIR/config.yaml}"', shared_launcher)
        self.assertNotIn('CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_RESOLVED}"', shared_launcher)
        self.assertNotIn('local world_checkpoint="$DATA_RUN_DIR/best.pt"', shared_launcher)
        self.assertNotIn('local world_checkpoint="$WORKSPACE_RUN_DIR/best.pt"', shared_launcher)
        self.assertNotIn('WORKSPACE_OUTPUT_DIR="${WORKSPACE_OUTPUT_DIR:-${OUTPUT_DIR:-$WORKSPACE_OUTPUT_DIR_RESOLVED}}"', shared_launcher)

    def test_single_future_ocp_is_shared_and_always_8gpu(self):
        train_config = yaml.safe_load(
            (
                RECIPE_DIR
                / "configs/physionpp/physion-vith-16to16-8gpu.yaml"
            ).read_text(encoding="utf-8")
        )
        task = train_config["tasks"]["single_future_ocp"]["launch"]
        self.assertEqual(
            task["module"],
            "src.data.physionpp.evaluate.eval_single_future_ocp",
        )
        self.assertTrue(task["distributed"])
        self.assertEqual(task["launcher"]["gpus"], 8)
        self.assertEqual(
            task["launcher"]["cuda_visible_devices"], "0,1,2,3,4,5,6,7"
        )
        self.assertEqual(
            task["checkpoint"],
            {"base": "data_run_dir", "path": "best.pt"},
        )
        resolved = resolve(train_config, "single_future_ocp")
        self.assertEqual(
            resolved["CHECKPOINT_RESOLVED"],
            "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/best.pt",
        )
        self.assertEqual(
            resolved["WORKSPACE_OUTPUT_DIR_RESOLVED"],
            "outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/"
            "evaluations/single_future_clip_ocp",
        )
        eval_config = yaml.safe_load(
            (
                RECIPE_DIR
                / "configs/physionpp/physion-vith-16to16-single-future-ocp.yaml"
            ).read_text(encoding="utf-8")
        )
        eval_task = eval_config["tasks"]["single_future_ocp"]
        self.assertTrue(eval_task["launch"]["distributed"])
        self.assertEqual(
            eval_task["launch"]["checkpoint"],
            {"base": "workspace_run_dir", "path": "best.pt"},
        )
        self.assertEqual(eval_task["launch"]["launcher"]["gpus"], 8)
        self.assertEqual(
            eval_task["experiment"]["evaluation"]["probe_variants"],
            ["full", "current", "predicted_future", "delta"],
        )
        self.assertNotIn("humans_dir", eval_task["experiment"]["evaluation"])

    def test_single_future_all_future_ocp_is_shared_and_always_8gpu(self):
        train_config = yaml.safe_load(
            (
                RECIPE_DIR
                / "configs/physionpp/physion-vith-16to16-8gpu.yaml"
            ).read_text(encoding="utf-8")
        )
        task = train_config["tasks"]["single_future_all_future_ocp"]["launch"]
        self.assertEqual(
            task["module"],
            "src.data.physionpp.evaluate.eval_single_future_all_future_ocp",
        )
        self.assertTrue(task["distributed"])
        self.assertEqual(task["launcher"]["gpus"], 8)
        self.assertEqual(
            task["checkpoint"],
            {"base": "workspace_run_dir", "path": "best.pt"},
        )
        resolved = resolve(train_config, "single_future_all_future_ocp")
        self.assertEqual(
            resolved["CHECKPOINT_RESOLVED"],
            "outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/best.pt",
        )
        self.assertEqual(
            resolved["WORKSPACE_OUTPUT_DIR_RESOLVED"],
            "outputs/runs/vjepa2_naive/physion_vith_native_predictor_16to16_8gpu/"
            "evaluations/single_future_all_future_ocp",
        )
        eval_config = yaml.safe_load(
            (
                RECIPE_DIR
                / "configs/physionpp/physion-vith-16to16-single-future-all-future-ocp.yaml"
            ).read_text(encoding="utf-8")
        )
        eval_task = eval_config["tasks"]["single_future_all_future_ocp"]
        self.assertEqual(eval_task["experiment"]["evaluation"]["label_scope"], "all_future")
        self.assertEqual(
            eval_task["experiment"]["evaluation"]["probe_variants"],
            ["full", "current", "predicted_future", "delta"],
        )

    def test_launch_outputs_paths_are_resolved_from_roots(self):
        resolver_source = (SHARED_LAUNCHER_DIR / "resolve_run.py").read_text(encoding="utf-8")
        self.assertNotIn('"QA_TRAJECTORY_ROOT_RESOLVED"', resolver_source)
        self.assertNotIn('"QA_CHECKPOINT_ROOT_RESOLVED"', resolver_source)
        self.assertIn("extra_config", resolver_source)

        config = {
            "launch": {
                "run": {
                    "project": "vjepa2-baiz",
                    "experiment": "vjepa2_naive",
                    "name": "path_base_test",
                    "workspace_root": "outputs/runs",
                    "data_root": "/data/shyang/outputs",
                },
                "resources": {"gpus": 1},
                "outputs": {
                },
                "extra_config": {
                    "qa_trajectory_root": {
                        "base": "data_run_dir",
                        "path": "qa_trajectories",
                    },
                    "qa_checkpoint_root": {
                        "base": "data_run_dir",
                        "path": "qa_checkpoints",
                    },
                },
            },
            "tasks": {
                "manual_eval": {
                    "launch": {
                        "output_dir": "evaluations/manual",
                    },
                },
            },
        }
        resolved = resolve(config, "manual_eval")
        self.assertEqual(
            resolved["WORKSPACE_OUTPUT_DIR_RESOLVED"],
            "outputs/runs/vjepa2_naive/path_base_test/evaluations/manual",
        )
        self.assertEqual(
            resolved["DATA_OUTPUT_DIR_RESOLVED"],
            "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/path_base_test/evaluations/manual",
        )
        self.assertEqual(
            resolved["QA_TRAJECTORY_ROOT_RESOLVED"],
            "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/path_base_test/qa_trajectories",
        )
        self.assertEqual(
            resolved["QA_CHECKPOINT_ROOT_RESOLVED"],
            "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/path_base_test/qa_checkpoints",
        )

    def test_launch_outputs_run_dirs_override_defaults(self):
        config = {
            "launch": {
                "run": {
                    "project": "vjepa2-baiz",
                    "experiment": "vjepa2_naive",
                    "name": "explicit_run_dir_test",
                    "workspace_root": "outputs/runs",
                    "data_root": "/data/shyang/outputs",
                },
                "resources": {"gpus": 1},
                "outputs": {
                    "workspace_run_dir": "/tmp/custom-workspace-run",
                    "data_run_dir": "/tmp/custom-data-run",
                },
            },
            "tasks": {"same_video_temporal_retrieval": {"launch": {}}},
        }
        resolved = resolve(config, "same_video_temporal_retrieval")
        self.assertEqual(resolved["WORKSPACE_RUN_DIR_RESOLVED"], "/tmp/custom-workspace-run")
        self.assertEqual(resolved["DATA_RUN_DIR_RESOLVED"], "/tmp/custom-data-run")
        self.assertEqual(
            resolved["WORKSPACE_OUTPUT_DIR_RESOLVED"],
            "/tmp/custom-workspace-run/evaluations/same_video_temporal_retrieval",
        )
        self.assertEqual(
            resolved["DATA_OUTPUT_DIR_RESOLVED"],
            "/tmp/custom-data-run/evaluations/same_video_temporal_retrieval",
        )

    def test_default_checkpoints_resolve_to_data_run_dir(self):
        config = {
            "launch": {
                "run": {
                    "project": "vjepa2-baiz",
                    "experiment": "vjepa2_naive",
                    "name": "checkpoint_base_test",
                    "workspace_root": "outputs/runs",
                    "data_root": "/data/shyang/outputs",
                },
                "resources": {"gpus": 1},
            },
            "tasks": {"same_video_temporal_retrieval": {"launch": {}}},
        }
        resolved = resolve(config, "same_video_temporal_retrieval")
        self.assertEqual(
            resolved["CHECKPOINT_RESOLVED"],
            "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/checkpoint_base_test/best.pt",
        )
        self.assertEqual(
            resolved["WORKSPACE_OUTPUT_DIR_RESOLVED"],
            "outputs/runs/vjepa2_naive/checkpoint_base_test/evaluations/same_video_temporal_retrieval",
        )
        self.assertEqual(
            resolved["DATA_OUTPUT_DIR_RESOLVED"],
            "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive/checkpoint_base_test/evaluations/same_video_temporal_retrieval",
        )


if __name__ == "__main__":
    unittest.main()
