#!/usr/bin/env python3
"""Density Extraction Script"""
import os
import shutil
import tempfile

import pyngp as ngp  # noqa

from instant_ngp_3dml.utils.log import logger
from instant_ngp_3dml.utils.profiler import profile


@profile
def density(snapshot_msgpack: str,
            output_image: str,
            resolution: int = 256,
            thresh: float = 2.5,
            density_range: float = 4.0,
            flip_y_and_z_axes: bool = False):
    """Extract Density from NeRF Volume"""
    # pylint: disable=too-many-arguments,too-many-statements,too-many-branches
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    logger.info(f"Loading snapshot {snapshot_msgpack}")
    assert snapshot_msgpack.endswith(".msgpack")
    testbed.load_snapshot(snapshot_msgpack)

    testbed.display_gui = False

    with tempfile.TemporaryDirectory() as dirpath:
        res3d = testbed.compute_and_save_png_slices(
            filename=os.path.join(dirpath, "density"),
            resolution=resolution,
            thresh=thresh,
            density_range=density_range,
            flip_y_and_z_axes=flip_y_and_z_axes)

        filename = os.path.join(dirpath, f"density.density_slices_{res3d[0]}x{res3d[1]}x{res3d[2]}.png")
        shutil.copyfile(filename, output_image)
