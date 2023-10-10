#!/usr/bin/python3
"""NeRF Utils Software."""
from typing import Callable
from typing import Dict

from utils_3dml.software import Cli

from instant_ngp_3dml.software.rendering import main as render
from instant_ngp_3dml.software.training import main as train

modules: Dict[str, Callable] = {
    "rendering": render,
    "training": train
}

if __name__ == "__main__":
    Cli("3DML Instant-NGP Software", modules)()
