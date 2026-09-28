#!/usr/bin/env bash
# Common launcher used by experiment scripts.
#
# Expected usage:
#
#   CONFIG=configs/<dataset>/<experiment>/config.yaml
#   TASK=train
#   source tools/launcher/launch_from_config.sh "$@"
#
# Machine-specific multi-node values are provided through environment variables:
#   NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE CUDA_VISIBLE_DEVICES
#
# Task arguments
# --------------
# `tasks.<task>.launch.args` is rendered through tools/launcher/render_task.py
# before being passed to the task module, so scripts never hardcode paths. The
# following placeholders are available (see render_task.py for the full list):
#
#   {config} {resolved_config} {task} {project} {dataset} {experiment} {stage}
#   {experiment_root} {data_experiment_root} {workspace_experiment_root}
#   {large_output_dir} {light_output_dir} {checkpoint} {seed}
#   {nnodes} {nproc} {master_addr} {master_port}
#   {cfg:<dotted.path>}
#
# `tasks.<task>.launch.config_template` / `.config_overrides` build the
# task-resolved config that is passed as `{resolved_config}` and recorded as
# `<light_output_dir>/configs/<task>.resolved.yaml`. `tasks.<task>.launch.env`
# is rendered into `<light_output_dir>/configs/<task>.env` and sourced before
# the task starts.
#
# Requirements:
#   - bash
#   - python
#   - PyYAML
#   - torchrun when tasks.<task>.launch.distributed=true

_launcher_die() {
    echo "[launcher] ERROR: $*" >&2
    return 1
}

# Resolve the directory holding this script so that helper modules can be found
# regardless of the caller's working directory.
LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_launcher_info() {
    echo "[launcher] $*" >&2
}

_launcher_require_value() {
    local option="$1"
    local value="${2-}"
    if [[ -z "$value" ]]; then
        _launcher_die "$option requires a value"
        return 1
    fi
}

_launcher_normalize_bool() {
    case "${1,,}" in
        1|true|yes|on)  printf '1' ;;
        0|false|no|off) printf '0' ;;
        *)
            _launcher_die "invalid boolean value: $1"
            return 1
            ;;
    esac
}

_launcher_is_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

_launcher_is_nonnegative_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

_launcher_abs_path() {
    python - "$1" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

_launcher_shell_join() {
    local out=""
    local arg
    for arg in "$@"; do
        printf -v out '%s%q ' "$out" "$arg"
    done
    printf '%s' "${out% }"
}

_launcher_validate_delete_target() {
    local target="$1"
    local experiment_root="$2"

    [[ -n "$target" ]] || {
        _launcher_die "refusing to delete an empty path"
        return 1
    }

    [[ "$target" != "/" ]] || {
        _launcher_die "refusing to delete /"
        return 1
    }

    [[ "$target" != "$experiment_root" ]] || {
        _launcher_die "refusing to delete experiment root: $target"
        return 1
    }

    case "$target" in
        "$experiment_root"/*) ;;
        *)
            _launcher_die "refusing to delete path outside experiment root: $target"
            return 1
            ;;
    esac
}

_launcher_handle_existing_output() {
    local target="$1"
    local experiment_root="$2"
    local mode="$3"
    local archive_tag="$4"
    local timestamp="$5"

    [[ -e "$target" ]] || return 0

    case "$mode" in
        archive)
            local legacy_dir="${experiment_root%/}/legacy"
            local archive_dir="${legacy_dir}/${archive_tag}_archived_${timestamp}"

            if [[ -e "$archive_dir" ]]; then
                archive_dir="${archive_dir}_$$"
            fi

            mkdir -p "$legacy_dir"
            mv "$target" "$archive_dir"
            _launcher_info "archived: $target -> $archive_dir"
            ;;
        delete)
            _launcher_validate_delete_target "$target" "$experiment_root" || return 1
            rm -rf -- "$target"
            _launcher_info "deleted existing output: $target"
            ;;
        raise)
            _launcher_die "output already exists: $target"
            return 1
            ;;
        *)
            _launcher_die "unknown existing_output mode: $mode"
            return 1
            ;;
    esac
}

_launcher_write_status() {
    local state="$1"
    local exit_code="${2-}"
    local pid="${3-}"

    LAUNCH_STATUS_STATE="$state" \
    LAUNCH_STATUS_EXIT_CODE="$exit_code" \
    LAUNCH_STATUS_PID="$pid" \
    python - "$LAUNCH_STATUS_FILE" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone

path = sys.argv[1]

def maybe_int(value):
    if value in ("", None):
        return None
    try:
        return int(value)
    except ValueError:
        return value

exit_code = os.environ.get("LAUNCH_STATUS_EXIT_CODE", "")
pid = os.environ.get("LAUNCH_STATUS_PID", "")

payload = {
    "task": os.environ["TASK"],
    "project": os.environ.get("LAUNCH_PROJECT"),
    "dataset": os.environ.get("LAUNCH_DATASET"),
    "experiment": os.environ.get("LAUNCH_EXPERIMENT"),
    "stage": maybe_int(os.environ.get("LAUNCH_STAGE", "")),
    "state": os.environ["LAUNCH_STATUS_STATE"],
    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "exit_code": maybe_int(exit_code),
    "nodes": [
        {
            "host": socket.gethostname(),
            "node_rank": maybe_int(os.environ.get("LAUNCH_NODE_RANK", "0")),
            "pid": maybe_int(pid),
            "log": os.environ.get("LAUNCH_LOG_FILE"),
            "exit_status": maybe_int(exit_code),
        }
    ],
}

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
os.replace(tmp, path)
PY
}

_launcher_write_environment() {
    python - "$LAUNCH_ENVIRONMENT_FILE" <<'PY'
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

path = sys.argv[1]

def run(cmd):
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return {
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }

torch_info = {}
try:
    import torch
    torch_info = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
except Exception as exc:
    torch_info = {"error": f"{type(exc).__name__}: {exc}"}

payload = {
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": sys.version,
    "executable": sys.executable,
    "torch": torch_info,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "distributed": {
        "nnodes": os.environ.get("LAUNCH_NNODES"),
        "node_rank": os.environ.get("LAUNCH_NODE_RANK"),
        "nproc_per_node": os.environ.get("LAUNCH_NPROC_PER_NODE"),
        "master_addr": os.environ.get("LAUNCH_MASTER_ADDR"),
        "master_port": os.environ.get("LAUNCH_MASTER_PORT"),
    },
    "git": {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status": run(["git", "status", "--short"]),
    },
    "nvidia_smi": run(["nvidia-smi", "-L"]),
}

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
os.replace(tmp, path)
PY
}

_launcher_write_manifest() {
    python - "$LAUNCH_MANIFEST_FILE" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path = sys.argv[1]

payload = {
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "project": os.environ.get("LAUNCH_PROJECT"),
    "dataset": os.environ.get("LAUNCH_DATASET"),
    "experiment": os.environ.get("LAUNCH_EXPERIMENT"),
    "stage": int(os.environ.get("LAUNCH_STAGE", "0")),
    "task": os.environ["TASK"],
    "config": os.environ["CONFIG"],
    "resolved_config": os.environ.get("LAUNCH_RESOLVED_CONFIG"),
    "module": os.environ.get("LAUNCH_MODULE"),
    "distributed": os.environ.get("LAUNCH_DISTRIBUTED") == "1",
    "output_mode": os.environ.get("LAUNCH_OUTPUT_MODE"),
    "experiment_root": os.environ.get("LAUNCH_EXPERIMENT_ROOT"),
    "large_data_output_dir": os.environ["LARGE_DATA_OUTPUT_DIR"],
    "light_data_output_dir": os.environ["LIGHT_DATA_OUTPUT_DIR"],
    "resume": os.environ.get("RESUME") == "1",
    "checkpoint": os.environ.get("CHECKPOINT") or None,
}

tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
os.replace(tmp, path)
PY
}

_launcher_sync_light_outputs() {
    [[ -n "${LAUNCH_LIGHT_MIRROR_DIR:-}" ]] || return 0
    [[ "$LAUNCH_LIGHT_MIRROR_DIR" != "$LIGHT_DATA_OUTPUT_DIR" ]] || return 0

    mkdir -p "$LAUNCH_LIGHT_MIRROR_DIR"

    local item
    for item in configs logs manifest.json command.txt environment.json; do
        local src="${LIGHT_DATA_OUTPUT_DIR%/}/$item"
        local dst="${LAUNCH_LIGHT_MIRROR_DIR%/}/$item"

        if [[ -d "$src" ]]; then
            rm -rf -- "$dst"
            cp -a -- "$src" "$dst"
        elif [[ -f "$src" ]]; then
            mkdir -p "$(dirname "$dst")"
            cp -a -- "$src" "$dst"
        fi
    done
}

_launcher_validate_gpu_count() {
    local nproc="$1"

    # CUDA_VISIBLE_DEVICES is the most direct description of what workers can see.
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local visible_count
        visible_count="$(
            python - "$CUDA_VISIBLE_DEVICES" <<'PY'
import sys
items = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(len(items))
PY
        )"

        if (( visible_count < nproc )); then
            _launcher_die \
                "NPROC_PER_NODE=$nproc but CUDA_VISIBLE_DEVICES exposes only $visible_count device(s): $CUDA_VISIBLE_DEVICES"
            return 1
        fi
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_count
        gpu_count="$(nvidia-smi -L | grep -c '^GPU ' || true)"
        if (( gpu_count < nproc )); then
            _launcher_die \
                "NPROC_PER_NODE=$nproc but only $gpu_count GPU(s) were detected"
            return 1
        fi
        return 0
    fi

    _launcher_die "cannot verify GPU count: neither CUDA_VISIBLE_DEVICES nor nvidia-smi is available"
    return 1
}

_launcher_main() {
    if [[ -z "${CONFIG:-}" ]]; then
        _launcher_die "CONFIG is not set"
        return 1
    fi

    if [[ -z "${TASK:-}" ]]; then
        _launcher_die "TASK is not set"
        return 1
    fi

    command -v python >/dev/null 2>&1 || {
        _launcher_die "python is not available in PATH"
        return 1
    }

    if [[ ! -f "$CONFIG" ]]; then
        _launcher_die "config not found: $CONFIG"
        return 1
    fi

    CONFIG="$(_launcher_abs_path "$CONFIG")"
    export CONFIG TASK

    python - <<'PY' >/dev/null 2>&1 || {
import yaml
PY
        _launcher_die "PyYAML is required to parse config YAML"
        return 1
    }

    # Parse scalar config values into shell-safe assignments.
    local parsed
    parsed="$(
        python - "$CONFIG" "$TASK" <<'PY'
import os
import shlex
import sys
import yaml

config_path, task_name = sys.argv[1], sys.argv[2]

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def require(mapping, key, path):
    if key not in mapping:
        raise SystemExit(f"missing config field: {path}.{key}")
    return mapping[key]

launch = require(cfg, "launch", "root")
run = require(launch, "run", "launch")
resources = require(launch, "resources", "launch")
outputs = require(launch, "outputs", "launch")
tasks = require(cfg, "tasks", "root")

if task_name not in tasks:
    raise SystemExit(f"task not found in config: {task_name}")

task = tasks[task_name] or {}
task_launch = require(task, "launch", f"tasks.{task_name}")

project = require(run, "project", "launch.run")
dataset = require(run, "dataset", "launch.run")
experiment = require(run, "experiment", "launch.run")
stage = int(require(run, "stage", "launch.run"))
workspace_root = require(run, "workspace_root", "launch.run")
data_root = require(run, "data_root", "launch.run")

module = require(task_launch, "module", f"tasks.{task_name}.launch")
distributed = bool(task_launch.get("distributed", False))
output_mode = task_launch.get("output_mode", "workspace")
existing_output = task_launch.get("existing_output", "archive")
background = bool(task_launch.get("background", False))
log_name = task_launch.get("log_name", f"{task_name}.log")
pid_name = task_launch.get("pid_name", f"{task_name}.pid")

resume_cfg = task_launch.get("resume") or {}
resume_checkpoint = resume_cfg.get("checkpoint")

gpus_per_node = int(resources.get("gpus_per_node", 1))
cuda_visible_devices = resources.get("cuda_visible_devices")
nnodes = int(resources.get("nnodes", 1))
node_rank = int(resources.get("node_rank", 0))
master_addr = str(resources.get("master_addr", "127.0.0.1"))
master_port = int(resources.get("master_port", 29500))

ctx = {
    "project": project,
    "dataset": dataset,
    "experiment": experiment,
    "stage": stage,
    "workspace_root": workspace_root,
    "data_root": data_root,
}

def render(value):
    if value is None:
        return ""
    value = str(value)
    for key, val in ctx.items():
        value = value.replace("{" + key + "}", str(val))
        value = value.replace("<" + key + ">", str(val))
    return value

workspace_output_root = render(
    require(outputs, "workspace_output_root", "launch.outputs")
)
data_output_root = render(
    require(outputs, "data_output_root", "launch.outputs")
)

if resume_checkpoint is not None:
    resume_checkpoint = render(resume_checkpoint)

def emit(name, value):
    if isinstance(value, bool):
        value = "1" if value else "0"
    elif value is None:
        value = ""
    else:
        value = str(value)
    print(f"{name}={shlex.quote(value)}")

emit("CFG_PROJECT", project)
emit("CFG_DATASET", dataset)
emit("CFG_EXPERIMENT", experiment)
emit("CFG_STAGE", stage)
emit("CFG_MODULE", module)
emit("CFG_DISTRIBUTED", distributed)
emit("CFG_OUTPUT_MODE", output_mode)
emit("CFG_EXISTING_OUTPUT", existing_output)
emit("CFG_BACKGROUND", background)
emit("CFG_LOG_NAME", log_name)
emit("CFG_PID_NAME", pid_name)
emit("CFG_RESUME_CHECKPOINT", resume_checkpoint)
emit("CFG_GPUS_PER_NODE", gpus_per_node)
emit("CFG_CUDA_VISIBLE_DEVICES", cuda_visible_devices)
emit("CFG_NNODES", nnodes)
emit("CFG_NODE_RANK", node_rank)
emit("CFG_MASTER_ADDR", master_addr)
emit("CFG_MASTER_PORT", master_port)
emit("CFG_WORKSPACE_OUTPUT_ROOT", workspace_output_root)
emit("CFG_DATA_OUTPUT_ROOT", data_output_root)
PY
    )" || return 1

    eval "$parsed"

    local cli_output_mode=""
    local cli_existing_output=""
    local cli_background=""
    local resume="0"
    local dry_run="0"
    local -a passthrough_args=()

    while (($#)); do
        case "$1" in
            --output-mode)
                _launcher_require_value "$1" "${2-}" || return 1
                cli_output_mode="$2"
                shift 2
                ;;
            --output-mode=*)
                cli_output_mode="${1#*=}"
                shift
                ;;
            --existing-output)
                _launcher_require_value "$1" "${2-}" || return 1
                cli_existing_output="$2"
                shift 2
                ;;
            --existing-output=*)
                cli_existing_output="${1#*=}"
                shift
                ;;
            --background)
                _launcher_require_value "$1" "${2-}" || return 1
                cli_background="$(_launcher_normalize_bool "$2")" || return 1
                shift 2
                ;;
            --background=*)
                cli_background="$(_launcher_normalize_bool "${1#*=}")" || return 1
                shift
                ;;
            --resume)
                resume="1"
                shift
                ;;
            --dry-run)
                dry_run="1"
                shift
                ;;
            --)
                shift
                passthrough_args+=("$@")
                break
                ;;
            *)
                passthrough_args+=("$1")
                shift
                ;;
        esac
    done

    local output_mode="${cli_output_mode:-$CFG_OUTPUT_MODE}"
    local existing_output="${cli_existing_output:-$CFG_EXISTING_OUTPUT}"
    local background="${cli_background:-$CFG_BACKGROUND}"

    case "$output_mode" in
        workspace|data|both) ;;
        *)
            _launcher_die "output_mode must be one of: workspace, data, both"
            return 1
            ;;
    esac

    case "$existing_output" in
        archive|delete|raise) ;;
        *)
            _launcher_die "existing_output must be one of: archive, delete, raise"
            return 1
            ;;
    esac

    if [[ "$background" != "0" && "$background" != "1" ]]; then
        _launcher_die "background must resolve to true/false"
        return 1
    fi

    local nnodes="${NNODES:-$CFG_NNODES}"
    local node_rank="${NODE_RANK:-$CFG_NODE_RANK}"
    local master_addr="${MASTER_ADDR:-$CFG_MASTER_ADDR}"
    local master_port="${MASTER_PORT:-$CFG_MASTER_PORT}"
    local nproc_per_node="${NPROC_PER_NODE:-$CFG_GPUS_PER_NODE}"

    _launcher_is_positive_int "$nnodes" || {
        _launcher_die "NNODES must be a positive integer: $nnodes"
        return 1
    }

    _launcher_is_nonnegative_int "$node_rank" || {
        _launcher_die "NODE_RANK must be a non-negative integer: $node_rank"
        return 1
    }

    _launcher_is_positive_int "$master_port" || {
        _launcher_die "MASTER_PORT must be a positive integer: $master_port"
        return 1
    }

    _launcher_is_positive_int "$nproc_per_node" || {
        _launcher_die "NPROC_PER_NODE must be a positive integer: $nproc_per_node"
        return 1
    }

    if (( node_rank >= nnodes )); then
        _launcher_die "NODE_RANK=$node_rank must be smaller than NNODES=$nnodes"
        return 1
    fi

    if [[ -z "${CUDA_VISIBLE_DEVICES+x}" && -n "$CFG_CUDA_VISIBLE_DEVICES" ]]; then
        export CUDA_VISIBLE_DEVICES="$CFG_CUDA_VISIBLE_DEVICES"
    fi

    local workspace_output_root="${CFG_WORKSPACE_OUTPUT_ROOT%/}"
    local data_output_root="${CFG_DATA_OUTPUT_ROOT%/}"

    local workspace_output_dir
    local data_output_dir
    local archive_tag

    if [[ "$CFG_STAGE" == "0" ]]; then
        workspace_output_dir="${workspace_output_root}/${TASK}"
        data_output_dir="${data_output_root}/${TASK}"
        archive_tag="${TASK}"
    else
        workspace_output_dir="${workspace_output_root}/stage_${CFG_STAGE}/${TASK}"
        data_output_dir="${data_output_root}/stage_${CFG_STAGE}/${TASK}"
        archive_tag="stage_${CFG_STAGE}_${TASK}"
    fi

    local large_data_output_dir
    local light_data_output_dir
    local light_mirror_dir=""

    case "$output_mode" in
        workspace)
            large_data_output_dir="$workspace_output_dir"
            light_data_output_dir="$workspace_output_dir"
            ;;
        data)
            large_data_output_dir="$data_output_dir"
            light_data_output_dir="$data_output_dir"
            ;;
        both)
            large_data_output_dir="$data_output_dir"
            light_data_output_dir="$data_output_dir"
            light_mirror_dir="$workspace_output_dir"
            ;;
    esac

    LARGE_DATA_OUTPUT_DIR="$large_data_output_dir"
    LIGHT_DATA_OUTPUT_DIR="$light_data_output_dir"
    RESUME="$resume"
    CHECKPOINT=""

    export LARGE_DATA_OUTPUT_DIR LIGHT_DATA_OUTPUT_DIR RESUME CHECKPOINT
    export TASK CONFIG

    export LAUNCH_PROJECT="$CFG_PROJECT"
    export LAUNCH_DATASET="$CFG_DATASET"
    export LAUNCH_EXPERIMENT="$CFG_EXPERIMENT"
    export LAUNCH_STAGE="$CFG_STAGE"
    export LAUNCH_MODULE="$CFG_MODULE"
    export LAUNCH_DISTRIBUTED="$CFG_DISTRIBUTED"
    export LAUNCH_OUTPUT_MODE="$output_mode"
    export LAUNCH_NNODES="$nnodes"
    export LAUNCH_NODE_RANK="$node_rank"
    export LAUNCH_NPROC_PER_NODE="$nproc_per_node"
    export LAUNCH_MASTER_ADDR="$master_addr"
    export LAUNCH_MASTER_PORT="$master_port"
    export LAUNCH_LIGHT_MIRROR_DIR="$light_mirror_dir"

    # Resolve checkpoint before existing-output handling so resume can use the
    # existing output tree safely.
    if [[ "$resume" == "1" ]]; then
        if [[ -n "$CFG_RESUME_CHECKPOINT" ]]; then
            CHECKPOINT="$CFG_RESUME_CHECKPOINT"
            if [[ ! -f "$CHECKPOINT" ]]; then
                _launcher_die "configured resume checkpoint not found: $CHECKPOINT"
                return 1
            fi
        elif [[ -f "${large_data_output_dir}/checkpoints/latest.pt" ]]; then
            CHECKPOINT="${large_data_output_dir}/checkpoints/latest.pt"
        elif [[ -f "${large_data_output_dir}/checkpoints/best.pt" ]]; then
            CHECKPOINT="${large_data_output_dir}/checkpoints/best.pt"
        else
            _launcher_die \
                "resume requested but no checkpoint found under ${large_data_output_dir}/checkpoints"
            return 1
        fi
        export CHECKPOINT
    fi

    local -a command_args=()
    if [[ "$CFG_DISTRIBUTED" == "1" ]]; then
        command_args=(
            torchrun
            "--nnodes=$nnodes"
            "--node-rank=$node_rank"
            "--master-addr=$master_addr"
            "--master-port=$master_port"
            "--nproc-per-node=$nproc_per_node"
            -m "$CFG_MODULE"
        )
    else
        command_args=(
            python -m "$CFG_MODULE"
        )
    fi

    # Task-scoped context: every task owns one output directory, and the
    # experiment root is its parent. Artifacts produced by one task are
    # referenced by later tasks through {experiment_root}/<sibling_task>/.
    local experiment_root
    experiment_root="$(dirname "$large_data_output_dir")"

    export LAUNCH_EXPERIMENT_ROOT="$experiment_root"
    export LAUNCH_DATA_EXPERIMENT_ROOT="$(dirname "$data_output_dir")"
    export LAUNCH_WORKSPACE_EXPERIMENT_ROOT="$(dirname "$workspace_output_dir")"
    export LAUNCH_RESOLVED_CONFIG="${LIGHT_DATA_OUTPUT_DIR}/configs/${TASK}.resolved.yaml"
    export LAUNCH_RENDERED_ENV="${LIGHT_DATA_OUTPUT_DIR}/configs/${TASK}.env"

    # Render args against the resolved context. --no-write first: --dry-run must
    # not create anything, and for real runs the output tree is not created until
    # the existing-output policy has been applied further below.
    local -a config_args=()
    mapfile -d '' -t config_args < <(
        python "$LAUNCHER_DIR/render_task.py" --config "$CONFIG" --task "$TASK"
    ) || return 1

    command_args+=("${config_args[@]}")
    command_args+=("${passthrough_args[@]}")

    local command_text
    command_text="$(_launcher_shell_join "${command_args[@]}")"

    if [[ "$dry_run" == "1" ]]; then
        cat <<EOF
project:               $CFG_PROJECT
dataset:               $CFG_DATASET
experiment:            $CFG_EXPERIMENT
stage:                 $CFG_STAGE
task:                  $TASK
config:                $CONFIG
resolved_config:       $LAUNCH_RESOLVED_CONFIG
module:                $CFG_MODULE
distributed:           $CFG_DISTRIBUTED
nnodes:                $nnodes
node_rank:             $node_rank
nproc_per_node:        $nproc_per_node
master_addr:           $master_addr
master_port:           $master_port
output_mode:           $output_mode
existing_output:       $existing_output
background:            $background
experiment_root:       $experiment_root
large_data_output_dir: $large_data_output_dir
light_data_output_dir: $light_data_output_dir
light_mirror_dir:      ${light_mirror_dir:-<none>}
resume:                $resume
checkpoint:            ${CHECKPOINT:-<none>}
command:               $command_text
EOF
        return 0
    fi

    if [[ "$CFG_DISTRIBUTED" == "1" ]]; then
        command -v torchrun >/dev/null 2>&1 || {
            _launcher_die "torchrun is not available in PATH"
            return 1
        }
        _launcher_validate_gpu_count "$nproc_per_node" || return 1
    fi

    # Validate Python module before mutating outputs.
    python - "$CFG_MODULE" <<'PY' >/dev/null || return 1
import importlib.util
import sys

module = sys.argv[1]
if importlib.util.find_spec(module) is None:
    raise SystemExit(f"python module not found: {module}")
PY

    # Resume preserves the current task output. A fresh launch applies the
    # configured existing_output policy.
    if [[ "$resume" == "0" ]]; then
        local timestamp
        timestamp="$(date '+%Y%m%d_%H%M%S')"

        case "$output_mode" in
            workspace)
                _launcher_handle_existing_output \
                    "$workspace_output_dir" "$workspace_output_root" \
                    "$existing_output" "$archive_tag" "$timestamp" || return 1
                ;;
            data)
                _launcher_handle_existing_output \
                    "$data_output_dir" "$data_output_root" \
                    "$existing_output" "$archive_tag" "$timestamp" || return 1
                ;;
            both)
                _launcher_handle_existing_output \
                    "$data_output_dir" "$data_output_root" \
                    "$existing_output" "$archive_tag" "$timestamp" || return 1

                if [[ "$workspace_output_dir" != "$data_output_dir" ]]; then
                    _launcher_handle_existing_output \
                        "$workspace_output_dir" "$workspace_output_root" \
                        "$existing_output" "$archive_tag" "$timestamp" || return 1
                fi
                ;;
        esac
    fi

    mkdir -p \
        "${LARGE_DATA_OUTPUT_DIR}/checkpoints" \
        "${LIGHT_DATA_OUTPUT_DIR}/configs" \
        "${LIGHT_DATA_OUTPUT_DIR}/logs"

    if [[ -n "$light_mirror_dir" ]]; then
        mkdir -p "$light_mirror_dir"
    fi

    local config_snapshot="${LIGHT_DATA_OUTPUT_DIR}/configs/$(basename "$CONFIG")"
    cp -f -- "$CONFIG" "$config_snapshot"

    # Persist the task-resolved config now that the output tree exists and the
    # existing-output policy has already been applied.
    python "$LAUNCHER_DIR/render_task.py" --config "$CONFIG" --task "$TASK" --write \
        >/dev/null || return 1

    if [[ -f "$LAUNCH_RENDERED_ENV" ]]; then
        # shellcheck disable=SC1090
        source "$LAUNCH_RENDERED_ENV"
    fi

    local log_file="${LIGHT_DATA_OUTPUT_DIR}/logs/${CFG_LOG_NAME}"
    local pid_file="${LIGHT_DATA_OUTPUT_DIR}/logs/${CFG_PID_NAME}"
    local exitcode_file="${LIGHT_DATA_OUTPUT_DIR}/logs/${TASK}.exitcode"
    local status_file="${LIGHT_DATA_OUTPUT_DIR}/logs/${TASK}.status.json"
    local manifest_file="${LIGHT_DATA_OUTPUT_DIR}/manifest.json"
    local command_file="${LIGHT_DATA_OUTPUT_DIR}/command.txt"
    local environment_file="${LIGHT_DATA_OUTPUT_DIR}/environment.json"

    case "$CFG_LOG_NAME" in
        */*)
            _launcher_die "log_name must be a filename, not a path: $CFG_LOG_NAME"
            return 1
            ;;
    esac

    case "$CFG_PID_NAME" in
        */*)
            _launcher_die "pid_name must be a filename, not a path: $CFG_PID_NAME"
            return 1
            ;;
    esac

    export LAUNCH_LOG_FILE="$log_file"
    export LAUNCH_PID_FILE="$pid_file"
    export LAUNCH_EXITCODE_FILE="$exitcode_file"
    export LAUNCH_STATUS_FILE="$status_file"
    export LAUNCH_MANIFEST_FILE="$manifest_file"
    export LAUNCH_COMMAND_FILE="$command_file"
    export LAUNCH_ENVIRONMENT_FILE="$environment_file"

    printf '%s\n' "$command_text" > "$command_file"
    _launcher_write_manifest || return 1
    _launcher_write_environment || return 1

    if [[ "$CFG_DISTRIBUTED" == "0" ]]; then
        export RANK=0
        export LOCAL_RANK=0
        export WORLD_SIZE=1
    fi

    if [[ "$background" == "1" ]]; then
        export -f _launcher_write_status
        export -f _launcher_sync_light_outputs

        # The child waits until the parent has written its pid file. This avoids
        # a race where a very short task exits and removes the pid before the
        # parent writes it.
        nohup bash -c '
            while [[ ! -f "$LAUNCH_PID_FILE" ]]; do
                sleep 0.05
            done

            set +e
            "$@"
            rc=$?

            printf "%s\n" "$rc" > "$LAUNCH_EXITCODE_FILE"
            _launcher_write_status "finished" "$rc" "$$"
            _launcher_sync_light_outputs
            rm -f -- "$LAUNCH_PID_FILE"

            exit "$rc"
        ' _ "${command_args[@]}" >>"$log_file" 2>&1 &

        local bg_pid=$!
        printf '%s\n' "$bg_pid" > "$pid_file"
        _launcher_write_status "running" "" "$bg_pid" || return 1
        _launcher_sync_light_outputs || return 1

        # Confirm the process did not immediately fail.
        sleep 0.5
        if ! kill -0 "$bg_pid" 2>/dev/null; then
            local early_rc="1"
            if [[ -f "$exitcode_file" ]]; then
                early_rc="$(cat "$exitcode_file")"
            fi

            if [[ "$early_rc" == "0" ]]; then
                _launcher_info "task completed before startup check; exit_code=0"
                return 0
            fi

            _launcher_die \
                "background task exited during startup; exit_code=$early_rc; log=$log_file"
            return "$early_rc"
        fi

        cat <<EOF
[launcher] started
task:                  $TASK
pid:                   $bg_pid
log:                   $log_file
large_data_output_dir: $LARGE_DATA_OUTPUT_DIR
light_data_output_dir: $LIGHT_DATA_OUTPUT_DIR
EOF
        return 0
    fi

    _launcher_write_status "running" "" "$$" || return 1
    _launcher_sync_light_outputs || return 1

    local had_errexit="0"
    case "$-" in
        *e*) had_errexit="1" ;;
    esac

    set +e
    "${command_args[@]}" 2>&1 | tee -a "$log_file"
    local task_rc="${PIPESTATUS[0]}"
    if [[ "$had_errexit" == "1" ]]; then
        set -e
    fi

    printf '%s\n' "$task_rc" > "$exitcode_file"
    _launcher_write_status "finished" "$task_rc" "$$"
    _launcher_sync_light_outputs

    return "$task_rc"
}

_launcher_main "$@"
_launcher_rc=$?

# launch_from_config.sh is normally sourced, but supporting direct execution
# makes debugging easier.
return "$_launcher_rc" 2>/dev/null || exit "$_launcher_rc"
