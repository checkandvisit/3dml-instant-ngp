#!/usr/bin/env python3
"""Density Extraction Script"""
import os

import numpy as np
import pyngp as ngp  # noqa
from tqdm import tqdm

from instant_ngp_3dml.utils.io import read_json
from instant_ngp_3dml.utils.io import write_json
from instant_ngp_3dml.utils.log import logger
from instant_ngp_3dml.utils.profiler import profile


@profile
def main(snapshot_msgpack: str,
         nerf_config_json: str,
         out_density_folder: str,
         vox_by_m: float = 50.,
         thresh: float = 2.5,
         density_range: float = 4.0,
         cube_size_m: float = 4.0,
         as_binary: bool = False):
    """Extract Density from NeRF Volume.

        The NeRF volume is divided into cubes where each cube has two files:
        the 3d grid and a json with the grid bounding box.

        The result resolution may be slightly different to use the cache optimization.

        Args:
            snapshot_msgpack: Input NeRF Weight
            nerf_config_json: Input NeRF Configuration Json
            out_density_folder: Output folder with density images
            vox_by_m: Grid resolution in voxel by meters
            thresh: Treshold value for density extraction
            density_range: Density Range to map density value in grayscale
            cube_size_m: Size of extracted cube in meters
            as_binary: Export density as binary instead png slice
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
    global_ngp_aabb = nerf_config["ngp_render_aabb"]  # In NGP coordinate system, not NERF's

    logger.info("Split aabb in compute cube")
    lower_bound = np.floor(np.array(global_ngp_aabb["p_min"])/(scale*cube_size_m)).astype(int)
    upper_bound = np.ceil(np.array(global_ngp_aabb["p_max"])/(scale*cube_size_m)).astype(int)

    os.makedirs(out_density_folder, exist_ok=True)

    index = 0
    with tqdm(desc="Extract Density", total=np.prod(upper_bound-lower_bound), unit="bbox") as t:
        for x in range(lower_bound[0], upper_bound[0]):
            for y in range(lower_bound[1], upper_bound[1]):
                for z in range(lower_bound[2], upper_bound[2]):

                    local_p_min = np.array([x, y, z])*scale*cube_size_m
                    local_p_max = np.array([x+1, y+1, z+1])*scale*cube_size_m
                    local_ngp_aabb = ngp.BoundingBox(local_p_min, local_p_max)

                    if as_binary:
                        res = testbed.save_density(
                            filename=os.path.join(out_density_folder, f"density_{index}"),
                            resolution=int(vox_by_m*cube_size_m),
                            aabb=local_ngp_aabb)
                    else:
                        # TODO Upgrade: Don't save empty Density grid
                        res = testbed.compute_and_save_png_slices(
                            filename=os.path.join(out_density_folder, f"density_{index}"),
                            resolution=int(vox_by_m*cube_size_m),
                            aabb=local_ngp_aabb,
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
                    t.update(1)
