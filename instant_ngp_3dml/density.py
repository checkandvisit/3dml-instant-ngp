#!/usr/bin/env python3
"""Density Extraction Script"""
import os
from instant_ngp_3dml.utils.io import read_json
import numpy as np

import pyngp as ngp  # noqa

from instant_ngp_3dml.utils.log import logger
from instant_ngp_3dml.utils.profiler import profile


@profile
def density(snapshot_msgpack: str,
            nerf_config_file: str,
            output_folder: str,
            resolution: int = 256,
            thresh: float = 2.5,
            density_range: float = 4.0,
            flip_y_and_z_axes: bool = True):
    """Extract Density from NeRF Volume"""
    # pylint: disable=too-many-arguments,too-many-statements,too-many-branches
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    logger.info(f"Loading snapshot {snapshot_msgpack}")
    assert snapshot_msgpack.endswith(".msgpack")
    testbed.load_snapshot(snapshot_msgpack)
    testbed.display_gui = False

    logger.info(f"Loading render_aabb from {nerf_config_file}")
    assert nerf_config_file.endswith(".json")
    _render_aabb = read_json(nerf_config_file)["render_aabb"]
    render_aabb = ngp.BoundingBox(np.array(_render_aabb["p_min"]), np.array(_render_aabb["p_max"]))

    os.makedirs(output_folder, exist_ok=True)
    testbed.compute_and_save_png_slices(
        filename=os.path.join(output_folder, "density"),
        resolution=resolution,
        aabb=render_aabb,
        thresh=thresh,
        density_range=density_range,
        flip_y_and_z_axes=flip_y_and_z_axes)
