#!/usr/bin/env python3
"""Density Extraction Script"""
import math
import os

import numpy as np
import pyngp as ngp  # noqa

from instant_ngp_3dml.utils.io import read_json
from instant_ngp_3dml.utils.io import write_json
from instant_ngp_3dml.utils.log import logger
from instant_ngp_3dml.utils.profiler import profile


@profile
def main(snapshot_msgpack: str,
         nerf_config_json: str,
         out_density_folder: str,
         resolution: int = 256,
         thresh: float = 2.5,
         density_range: float = 4.0,
         cube_size: int = 4):
    """Extract Density from NeRF Volume.

        The NeRF volume is divided into cubes where each cube has two files:
        the 3d grid and a json with the grid bounding box.

        Args:
            snapshot_msgpack: Input NeRF Weight
            nerf_config_json: Input NeRF Configuration Json
            out_density_folder: Output folder with density images
            resolution: Grid resolution in voxel by meters
            thresh: Treshold value for density extraction
            density_range: Density Range to map density value in grayscale
            cube_size: Size of extracted cube
    """
    # pylint: disable=too-many-arguments,too-many-statements,too-many-branches
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    logger.info(f"Loading snapshot {snapshot_msgpack}")
    assert snapshot_msgpack.endswith(".msgpack")
    testbed.load_snapshot(snapshot_msgpack)
    testbed.display_gui = False

    logger.info(f"Loading render_aabb from {nerf_config_json}")
    assert nerf_config_json.endswith(".json")
    nerf_config = read_json(nerf_config_json)

    scale = nerf_config["scale"]
    _ngp_render_aabb = nerf_config["ngp_render_aabb"]  # In NGP coordinate system, not NERF's

    p_min = np.array(_ngp_render_aabb["p_min"])/(scale*cube_size)
    p_max = np.array(_ngp_render_aabb["p_max"])/(scale*cube_size)

    os.makedirs(out_density_folder, exist_ok=True)

    index = 0
    for x in range(math.floor(p_min[0]), math.ceil(p_max[0])):
        for y in range(math.floor(p_min[1]), math.ceil(p_max[1])):
            for z in range(math.floor(p_min[2]), math.ceil(p_max[2])):

                local_p_min = np.array([x, y, z])*scale*cube_size
                local_p_max = np.array([x+1, y+1, z+1])*scale*cube_size
                ngp_render_aabb = ngp.BoundingBox(local_p_min, local_p_max)

                res = testbed.compute_and_save_png_slices(
                    filename=os.path.join(out_density_folder, f"density_{index}"),
                    resolution=resolution*cube_size,
                    aabb=ngp_render_aabb,
                    thresh=thresh,
                    density_range=density_range,
                    flip_y_and_z_axes=False)

                density_data = {
                    "ngp_render_aabb": {
                        "p_min": local_p_min.tolist(),
                        "p_max": local_p_max.tolist()
                    },
                    "res": res.tolist()
                }

                write_json(os.path.join(out_density_folder, f"density_{index}.json"), density_data, pretty=True)

                index += 1
