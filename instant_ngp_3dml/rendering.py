#!/usr/bin/env python3
"""Rendering Script"""
import os
from typing import Dict

import imageio
import numpy as np
import pyngp as ngp  # noqa
from tqdm import tqdm

from instant_ngp_3dml.utils.io import read_json
from instant_ngp_3dml.utils.log import logger
from instant_ngp_3dml.utils.profiler import profile
from instant_ngp_3dml.utils.tonemapper import linear_to_srgb


RENDER_MODES: Dict[str, ngp.RenderMode] = {"depth": ngp.RenderMode.Depth,
                                           "color": ngp.RenderMode.Shade,
                                           "confidence": ngp.RenderMode.Confidence}
CAMERA_MODES: Dict[str, ngp.CameraMode] = {"perspective": ngp.CameraMode.Perspective,
                                           "orthographic": ngp.CameraMode.Orthographic,
                                           "environment": ngp.CameraMode.Environment}


@profile
def __save_color(outname, image):
    image = np.copy(image)
    # Unmultiply alpha
    image[..., 0:3] = np.divide(
        image[..., 0:3],
        image[..., 3:4],
        out=np.zeros_like(image[..., 0:3]),
        where=image[..., 3:4] != 0)
    image[..., 0:3] = linear_to_srgb(image[..., 0:3])
    image = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # Some NeRF datasets lack the .png suffix in the dataset metadata
    if os.path.splitext(outname)[1] != ".png":
        outname = outname + ".png"

    os.makedirs(os.path.dirname(outname), exist_ok=True)
    imageio.imwrite(outname, image)


def set_render_params(testbed: ngp.Testbed, render_mode: str, spp: int) -> int:
    """Set Testbed with render params for render mode

        Args:
            testbed: InstantNGP Manager
            render_mode: The mode of rendering (cf ngp.RenderMode)
            spp: The desired spp for rendering

        Returns:
            int: The spp optimized for the render_mode

        Raises:
            ValueError: Only RENDER_MODES is accepted."""

    if render_mode == "color":
        return spp

    if render_mode == "depth":
        logger.info("Set depth rendering params")
        testbed.tonemap_curve = ngp.TonemapCurve.Identity
        testbed.color_space = ngp.ColorSpace.Linear
        testbed.render_mode = ngp.RenderMode.Depth
        testbed.exposure = -4.0
        return 1

    if render_mode == "confidence":
        logger.info("Set confidence rendering params")
        testbed.tonemap_curve = ngp.TonemapCurve.Identity
        testbed.color_space = ngp.ColorSpace.Linear
        testbed.render_mode = ngp.RenderMode.Confidence
        testbed.exposure = 1.0
        return 1

    raise ValueError


@profile
def main(snapshot_msgpack: str,
         nerf_transform_json: str,
         out_rendering_folder: str,
         spp: int = 4,
         display: bool = False,
         num_max_images: int = -1,
         downscale_factor: float = 1.0,
         render_mode: str = "color",
         camera_mode: str = "perspective"):
    """Render NeRF Scene.

        Args:
            snapshot_msgpack: Input NeRF Weight
            nerf_transform_json: Input NeRF Transform Json
            out_rendering_folder: Output Folder with rendered images
            spp: Sample per pixel
            display: Display result directly in GUI
            num_max_images: Limit the number of rendered images
            downscale_factor: Downscale rendered frames
            render_mode: Renderer method
            camera_mode: Camera model

        Raises:
            ValueError: if CameraMode or RenderMode doesn't exist
    """
    # pylint: disable=too-many-arguments,too-many-statements,too-many-branches
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    logger.info(f"Loading snapshot {snapshot_msgpack}")
    assert snapshot_msgpack.endswith(".msgpack")
    testbed.load_snapshot(snapshot_msgpack)

    logger.debug(f"Load rendering transforms from {nerf_transform_json}")
    ref_transforms = read_json(nerf_transform_json)

    # Pick a sensible GUI resolution depending on arguments.
    sw = int(ref_transforms["w"] / downscale_factor)
    sh = int(ref_transforms["h"] / downscale_factor)
    while sw*sh > 1920*1080*4:
        sw = int(sw / 2)
        sh = int(sh / 2)
    if display:
        logger.debug("Use Onscreen Rendering")
        testbed.init_window(sw, sh, hidden=False)
    else:
        logger.debug("Use Offscreen Rendering")

    testbed.shall_train = False
    testbed.nerf.render_with_camera_distortion = True
    testbed.display_gui = False

    testbed.exposure = 1.0
    testbed.fov_axis = 0
    testbed.fov = np.rad2deg(ref_transforms["camera_angle_x"])
    testbed.dynamic_res = False
    testbed.fixed_res_factor = 1

    assert camera_mode.lower() in CAMERA_MODES,\
        f"Invalid camera mode '{camera_mode}'. Should be in {CAMERA_MODES.keys()}"
    testbed.camera_mode = CAMERA_MODES[camera_mode.lower()]

    assert render_mode.lower() in RENDER_MODES, \
        f"Invalid render mode '{render_mode}'. Should be in {RENDER_MODES.keys()}"
    testbed.render_mode = RENDER_MODES[render_mode.lower()]

    spp = set_render_params(testbed, render_mode, spp)

    nb_frames = len(ref_transforms["frames"])
    if num_max_images > 0:
        nb_frames = min(num_max_images, nb_frames)

    with tqdm(desc="Rendering", total=nb_frames, unit="frame") as t:
        for idx in range(nb_frames):
            f = ref_transforms["frames"][int(idx)]
            cam_matrix = f["transform_matrix"]
            testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1, :])
            outname = os.path.join(out_rendering_folder, os.path.basename(f["file_path"]))

            if display:
                testbed.reset_accumulation()
                for _ in range(spp):
                    testbed.frame()

                image = testbed.screenshot()
            else:
                image = testbed.render(sw, sh, spp, True)
                image[..., :3] *= 2**(-1*testbed.exposure)

            if render_mode == "color":
                __save_color(outname, image)
            elif render_mode == "depth":
                # Force depth in numpy format
                outname = os.path.splitext(outname)[0] + ".npy"
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                np.save(outname, image[..., 0])
            elif render_mode == "confidence":
                __save_color(outname, image)
            else:
                raise ValueError(f"Invalid render mode '{render_mode}'. Should be in {RENDER_MODES.keys()}")

            t.update(1)
