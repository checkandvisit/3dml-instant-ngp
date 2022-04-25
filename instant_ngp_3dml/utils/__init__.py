# /usr/bin/python3
"""Constant Information."""
import os
from pathlib import Path
from typing import Final


def is_aws_job() -> bool:
    """Check if we are in aws job."""
    return "AWS_BATCH_JOB_ID" in os.environ


DIR_PATH: Final[str] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../")
HOME_DIR: Final[str] = str(Path.home())+"/"

DATA_DIR: Final[str] = os.path.join(
    HOME_DIR if is_aws_job() else DIR_PATH, "data/")
NERF_CONFIG: Final[str] = os.path.join(
    DIR_PATH, "configs", "nerf", "{config}.json")
