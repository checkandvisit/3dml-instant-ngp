#!/usr/bin/env python3
"""Rendering Script"""
import os

import imageio
import numpy as np
import pyngp as ngp  # noqa
from tqdm import tqdm

from instant_ngp_3dml.utils.io import read_json
from instant_ngp_3dml.utils.log import logger
from instant_ngp_3dml.utils.profiler import profile
from instant_ngp_3dml.utils.tonemapper import linear_to_srgb


@profile
def __save(outname, image):
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
    if not os.path.splitext(outname)[1]:
        outname = outname + ".png"

    os.makedirs(os.path.dirname(outname), exist_ok=True)
    imageio.imwrite(outname, image)


@profile
def render(load_snapshot: str,
           screenshot_transforms: str,
           screenshot_dir: str,
           width: int = 1920,
           height: int = 1080,
           depth: bool = False,
           screenshot_spp: int = 4,
           display: bool = False,
           num_max_images: int = -1):
    """Nerf renderer"""
    # pylint: disable=too-many-arguments

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    logger.info(f"Loading snapshot {load_snapshot}")
    testbed.load_snapshot(load_snapshot)

    logger.debug(f"Screenshot transforms from {screenshot_transforms}")
    ref_transforms = read_json(screenshot_transforms)

    # Pick a sensible GUI resolution depending on arguments.
    sw = width
    sh = height
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
    testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
    testbed.dynamic_res = False
    testbed.fixed_res_factor = 1

    if depth:
        logger.info("Set depth rendering params")
        testbed.tonemap_curve = ngp.TonemapCurve.Identity
        testbed.color_space = ngp.ColorSpace.Linear
        testbed.render_mode = ngp.RenderMode.Depth
        testbed.exposure = -4.0
        screenshot_spp = 1

    nb_frames = len(ref_transforms["frames"])
    if num_max_images > 0:
        nb_frames = min(num_max_images, nb_frames)
    with tqdm(desc="Rendering", total=nb_frames, unit="frame") as t:
        for idx in range(nb_frames):
            f = ref_transforms["frames"][int(idx)]
            cam_matrix = f["transform_matrix"]
            testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1, :])
            outname = os.path.join(
                screenshot_dir, os.path.basename(f["file_path"]))

            if display:
                testbed.reset_accumulation()
                for _ in range(screenshot_spp):
                    testbed.frame()

                image = testbed.screenshot()
            else:
                image = testbed.render(sw, sh, screenshot_spp, True)
                image[..., :3] *= 2**(-1*testbed.exposure)

            if depth:
                # Force depth in numpy format
                outname = os.path.splitext(outname)[0] + ".npy"
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                np.save(outname, image[..., 0])
                # No need to correct exposure on depth

            else:
                __save(outname, image)

            t.update(1)
