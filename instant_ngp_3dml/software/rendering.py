#!/usr/bin/python3
"""Rendering Script."""
import os
from typing import Dict
from typing import Final
from typing import Tuple

import imageio
import numpy as np
import pyngp as ngp  # noqa
from tqdm import tqdm
from utils_3dml.file.extensions import FileExt
from utils_3dml.monitoring.profiler import LogScopeTime
from utils_3dml.monitoring.profiler import profile
from utils_3dml.structure.nerf.nerf_predicted_images import NERF_RENDERING_FORMATS
from utils_3dml.structure.nerf.nerf_predicted_images import NerfPredictionPath
from utils_3dml.structure.nerf.nerf_transforms import NerfTransforms
from utils_3dml.utils.asserts import assert_in
from utils_3dml.utils.asserts import assert_isfile
from utils_3dml.utils.asserts import assert_len

from instant_ngp_3dml import logger
from instant_ngp_3dml.utils.tonemapper import linear_to_srgb
from instant_ngp_3dml.utils.tonemapper import tonemap

NGP_RENDER_MODES: Final[Dict[NerfPredictionPath, ngp.RenderMode]] = {
    NerfPredictionPath.IMAGE: ngp.RenderMode.Shade,
    NerfPredictionPath.DEPTH: ngp.RenderMode.Depth,
    # "confidence": ngp.RenderMode.Confidence
}


@profile
def __save_color(outname, image):
    image = np.copy(image)
    # Un-multiply alpha
    image[..., 0:3] = np.divide(
        image[..., 0:3],
        image[..., 3:4],
        out=np.zeros_like(image[..., 0:3]),
        where=image[..., 3:4] != 0)
    image[..., 0:3] = linear_to_srgb(image[..., 0:3])
    image = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # Some NeRF datasets lack the .png suffix in the dataset metadata
    if os.path.splitext(outname)[1] != ".png":
        outname = os.path.splitext(outname)[0] + ".png"

    os.makedirs(os.path.dirname(outname), exist_ok=True)
    imageio.imwrite(outname, image)


def get_testbed_and_spp(snapshot_msgpack: str, render_mode: NerfPredictionPath, spp: int) -> Tuple[ngp.Testbed, int]:
    """Init TestBed and Spp for Rendering."""
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    logger.info(f"Loading snapshot {snapshot_msgpack}")
    assert snapshot_msgpack.endswith(".msgpack")
    testbed.load_snapshot(snapshot_msgpack)

    testbed.shall_train = False
    testbed.display_gui = False

    testbed.dynamic_res = False
    testbed.fixed_res_factor = 1

    assert_in(render_mode, NGP_RENDER_MODES)
    testbed.render_mode = NGP_RENDER_MODES[render_mode]

    if render_mode == NerfPredictionPath.IMAGE:
        pass
    elif render_mode == NerfPredictionPath.DEPTH:
        logger.info("Set depth rendering params")
        testbed.tonemap_curve = ngp.TonemapCurve.Identity
        testbed.color_space = ngp.ColorSpace.Linear
        testbed.render_mode = ngp.RenderMode.Depth
        spp = 1
    # elif render_mode == "confidence":
    #     logger.info("Set confidence rendering params")
    #     testbed.tonemap_curve = ngp.TonemapCurve.Identity
    #     testbed.color_space = ngp.ColorSpace.Linear
    #     testbed.render_mode = ngp.RenderMode.Confidence
    #     spp = 1
    else:
        raise ValueError(f"Unhandled rendering mode: {render_mode.name}")

    return testbed, spp


@profile
def main(snapshot_msgpack: str,
         nerf_transform_json: str,
         out_rendering_folder: str,
         render_type: str,
         spp: int = 4,
         color_depth: bool = True):
    """Render NeRF Scene.

    Args:
        snapshot_msgpack: Input NeRF Weights
        nerf_transform_json: Input NeRF Transform Json
        out_rendering_folder: Output Folder with rendered images
        spp: Input number of samples per pixel
        render_type: Input renderer method (See utils_3dml.structure.nerf.nerf_dataset.NerfDatasetFormat)
        color_depth: Input tonemap the generated Depthmaps, if render_type=="depth"

    Raises:
        ValueError: if render_type doesn't exist

    Resources:
        cpu: normal
        ram: normal
        gpu: intensive
        network: none
    """
    assert_isfile(nerf_transform_json, ext=FileExt.JSON)
    logger.debug(f"Load rendering transforms from {nerf_transform_json}")
    nerf_transform = NerfTransforms.load(nerf_transform_json)  # Validate JSON Schema

    render_mode = NerfPredictionPath[render_type.upper()]
    assert_in(render_mode, NERF_RENDERING_FORMATS)

    testbed, spp = get_testbed_and_spp(snapshot_msgpack, render_mode, spp)

    # Use load_training_data to load each input camera and re-run them using set_camera_to_training_view
    testbed.load_training_data(nerf_transform_json)

    with LogScopeTime(f"NeRF {render_type.capitalize()} Rendering"):
        assert_len(nerf_transform.frames, testbed.nerf.training.dataset.n_images)
        for trainview, filepath in tqdm(enumerate(testbed.nerf.training.dataset.paths), desc="Rendering", unit="frame",
                                        total=testbed.nerf.training.dataset.n_images):
            testbed.set_camera_to_training_view(trainview)
            assert testbed.nerf.render_with_camera_distortion
            w, h = tuple(testbed.nerf.training.dataset.metadata[trainview].resolution)

            image = testbed.render(w, h, spp, True)
            outname = os.path.join(out_rendering_folder, os.path.basename(filepath))

            if render_mode == NerfPredictionPath.IMAGE:
                __save_color(outname, image)
            elif render_mode == NerfPredictionPath.DEPTH:
                # Force depth in numpy format
                outname = os.path.splitext(outname)[0] + ".npy"
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                raw_depth = image[..., 0]
                np.save(outname, raw_depth)

                if color_depth:
                    outname = os.path.splitext(outname)[0] + ".png"
                    os.makedirs(os.path.dirname(outname), exist_ok=True)
                    imageio.imwrite(outname, tonemap(raw_depth))
            # elif render_type == "confidence":
            #     __save_color(outname, image)
            else:
                raise ValueError(f"Invalid render mode '{render_mode}'. Should be in {NGP_RENDER_MODES.keys()}")
