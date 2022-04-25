# /usr/bin/python3
"""Geometry Utils Software."""
import glob
import os
import sys
from typing import Callable
from typing import Dict

import fire

from instant_ngp_3dml.compute_scene import compute_scene
from instant_ngp_3dml.rendering import render
from instant_ngp_3dml.training import train
from instant_ngp_3dml.utils import DIR_PATH

# Add pyngp to PYTHONPATH
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(
    os.path.join(DIR_PATH, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(
    os.path.join(DIR_PATH, "build*", "**/*.so"), recursive=True)]


modules: Dict[str, Callable] = {
    "rendering": render,
    "training": train,
    "compute_scene": compute_scene
}

HELP = """Geometry
positional arguments:
  """+str(list(modules.keys()))+"""     Software name
optional arguments:
  -h, --help            show this help message and exit
other:
    software_arguments  the arguments of software
"""

if __name__ == "__main__":
    argv = sys.argv

    if len(argv) > 1 and argv[1] in modules:
        fire.Fire(modules[argv[1]], command=argv[2:])
    else:
        print(HELP)
        sys.exit(1)
