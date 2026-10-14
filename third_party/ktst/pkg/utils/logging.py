from functools import lru_cache
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

RUN_DIR = os.getcwd()
OUTPUT_DIR = "outputs"

logger = logging.getLogger(name=__name__)

CONFIG_FILE_NAME = "config.yaml"
CHECKPOINT_NAME = "checkpoint"
BEST_FILE_NAME = "best.json"
RESULTS_FILE_NAME = "results.json"


def save_snapshot_of_source_code(
    source_path: str = ".source", pkg: str = "pkg", file_name: str = "main.py"
) -> None:
    dst = Path(source_path).absolute()
    code_dir = Path(
        RUN_DIR
    ).absolute()  # this introduces a convention: run code from code directory
    shutil.copytree(src=code_dir / pkg, dst=dst / pkg)
    shutil.copy(src=code_dir / Path(file_name).name, dst=dst / Path(file_name).name)

    logger.info(f"Saving source code")


class Timer:
    def __init__(self):
        self.start = datetime.now()

    def get_duration_in_seconds(self) -> float:
        self.stop = datetime.now()
        duration = self.stop - self.start
        return duration.total_seconds()


def get_checkpoint_path_without_suffix(
    dir: str, iteration: Union[str, int] = "best"
) -> Path:
    if iteration == "best":
        with open(file=Path(dir) / BEST_FILE_NAME, mode="r") as file:
            best_dict: dict = json.load(fp=file)
        iteration = int(best_dict.get("iteration"))  # type: ignore
    checkpoint_path_without_suffix = Path(dir) / f"{CHECKPOINT_NAME}-{iteration:06}"
    return checkpoint_path_without_suffix


def delete_old_checkpoints(
    log_dir: Path = Path(), keep_iterations: Optional[list[int]] = None
) -> None:
    all_log_files = [path for path in log_dir.glob(pattern="*") if path.is_file()]
    checkpoint_files = [file for file in all_log_files if CHECKPOINT_NAME in str(file)]

    if keep_iterations is not None:
        checkpoint_iterations = [
            _extract_iteration(checkpoint) for checkpoint in checkpoint_files
        ]
        paths_of_checkpoints_to_be_removed: list[Path] = [
            checkpoint
            for (checkpoint, iteration) in zip(checkpoint_files, checkpoint_iterations)
            if iteration not in keep_iterations
        ]
    else:
        paths_of_checkpoints_to_be_removed = checkpoint_files

    for checkpoint in paths_of_checkpoints_to_be_removed:
        try:
            checkpoint.unlink()
        except:
            raise OSError("Error while deleting old checkpoint files")


def _extract_iteration(checkpoint_path: Path) -> int:
    import re

    match, *_ = re.findall(pattern=r"\d+", string=str(checkpoint_path.name))
    assert _ == []
    return int(match)


@lru_cache(None)
def logger_warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)
