"""Allow launcher bookkeeping while refusing to overwrite experiment artifacts."""
from pathlib import Path


LAUNCHER_ENTRIES = frozenset({
    "checkpoints", "configs", "logs", "command.txt", "environment.json",
    "manifest.json",
})


def ensure_no_artifacts(output: Path) -> None:
    if not output.exists():
        return
    unexpected = sorted(path.name for path in output.iterdir()
                        if path.name not in LAUNCHER_ENTRIES)
    if unexpected:
        raise FileExistsError(
            f"experiment artifacts already exist in {output}: {unexpected[:5]}; "
            "start a new run or archive the previous task output"
        )
