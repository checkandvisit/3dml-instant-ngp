# /usr/bin/python3
"""Compute NeRF Scene"""
import os
from time import process_time
from typing import Any
from typing import Dict
from typing import get_args
from typing import List
from typing import Literal

from instant_ngp_3dml.density import density
from instant_ngp_3dml.rendering import render
from instant_ngp_3dml.training import train
from instant_ngp_3dml.utils import DATA_DIR
from instant_ngp_3dml.utils import NERF_CONFIG
from instant_ngp_3dml.utils.io import write_json
from instant_ngp_3dml.utils.log import logger
from instant_ngp_3dml.utils.profiler import export_profiling_events
from instant_ngp_3dml.utils.profiler import profile
from instant_ngp_3dml.utils.tonemapper import tonemap_folder

DEFAULT_MAX_STEP = 20000
ENABLE_S3_UPLOAD = False
S3_URL_FORMAT = "s3://checkandvisit-3dml-dev/dataset_test/nerf/{scene_name}/"

SceneName = Literal["lego", "barbershop", "0223-1010", "0223-1118", "0223-1120"]


def img_folder_to_video(folder: str, output_mp4: str, fps: int):
    """Convert folder of images to video"""
    assert output_mp4.endswith(".mp4")
    cmd = f"ffmpeg -r {fps} -y"
    cmd += " -i "+os.path.join(folder, "frame_%04d.png")
    cmd += " -c:v libx264"
    cmd += f" -vf 'fps={fps},format=yuv420p'"
    cmd += f" {output_mp4}"
    os.system(cmd)


def merge_videos(videos_mp4: List[str], output_mp4: str, horizontal: bool = True):
    """Merge videos"""
    cmd = "ffmpeg -y "
    cmd += " ".join([f"-i {video}" for video in videos_mp4])
    cmd += " -filter_complex "
    cmd += "hstack" if horizontal else "vstack"
    cmd += f" {output_mp4}"
    os.system(cmd)


def get_snapshot_path(snapshot_dir: str, idx: int) -> str:
    """Get snapshot path"""
    return os.path.join(snapshot_dir, f"snap_{idx}.msgpack")


class SceneComputer:
    """Train and render a test scene with NERF."""

    def __init__(self, scene_name: SceneName, config: str):
        assert scene_name in set(get_args(SceneName)), \
            f"Unknown test scene name, should be in {set(get_args(SceneName))}"
        self.scene_dir = os.path.join(DATA_DIR, scene_name)
        if not os.path.isdir(self.scene_dir):
            logger.info("Download data from S3")
            self.download_scene(S3_URL_FORMAT.format(scene_name=scene_name))

        self.config_path = NERF_CONFIG.format(config=config)

        self.result_dir = os.path.join(self.scene_dir, config)
        self.snapshot_dir = os.path.join(self.result_dir, "snapshot")

        os.makedirs(self.result_dir, exist_ok=True)

        self.info: Dict[str, Any] = {}

    def download_scene(self, scene_url: str):
        """Download Scene"""

        os.makedirs(self.scene_dir, exist_ok=True)

        cmd = f"aws s3 sync {scene_url} {self.scene_dir}"
        os.system(cmd)

    @profile
    def train(self, n_step: int = 2000, max_step: int = DEFAULT_MAX_STEP):
        """Train Scene."""

        training_json = os.path.join(self.scene_dir, "training.json")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        start_time = process_time()
        for i in range(int(max_step/n_step)):
            logger.info(f"--- Run Step {i*n_step} to {(i + 1) * n_step} ---")
            snapshot_path = get_snapshot_path(self.snapshot_dir, (i+1)*n_step)
            if os.path.isfile(snapshot_path):
                logger.info("Snapshot already exists, continue...")
                continue
            prev_snapshot_path = get_snapshot_path(self.snapshot_dir, i*n_step) if i > 0 else ""
            train(scene=training_json,
                  save_snapshot=snapshot_path,
                  load_snapshot=prev_snapshot_path,
                  n_steps=(i+1) * n_step,
                  network=self.config_path)
        end_time = process_time()

        self.info["n_step"] = n_step
        self.info["max_step"] = max_step
        self.info["training_time"] = end_time-start_time

    @profile
    def render(self, render_mode: str = "color", camera_mode: str = "perspective", topview: bool = False,
               step_idx: int = DEFAULT_MAX_STEP, display: bool = False, num_max_images: int = -1) -> str:
        """Render Scene."""
        # pylint:disable=too-many-arguments
        transforms_json = os.path.join(self.scene_dir, "topview.json" if topview else "test.json")

        start_time = process_time()
        render_folder = render(snapshot_msgpack=get_snapshot_path(self.snapshot_dir, step_idx),
                               transforms_json=transforms_json,
                               output_dir=self.result_dir,
                               display=display,
                               num_max_images=num_max_images,
                               render_mode=render_mode,
                               camera_mode=camera_mode)
        end_time = process_time()
        self.info["render_time"] = end_time-start_time

        return render_folder

    @profile
    def extract_density(self, step_idx: int = DEFAULT_MAX_STEP) -> str:
        """Extract Density"""

        output_density = os.path.join(self.scene_dir, "density.png")

        start_time = process_time()
        density(snapshot_msgpack=get_snapshot_path(self.snapshot_dir, step_idx),
                output_image=output_density)
        end_time = process_time()
        self.info["density_time"] = end_time-start_time

        return output_density

    def save_info(self) -> str:
        """Save Info."""

        info_json = os.path.join(self.result_dir, "info.json")
        write_json(info_json, self.info)
        return info_json

    def upload_scene(self, scene_url: str):
        """Upload Scene."""

        cmd = f"aws s3 sync {self.scene_dir} {scene_url}"
        os.system(cmd)


def compute_scene(scene: SceneName,
                  config: str = "base",
                  display: bool = False,
                  output_video_fps: int = 2,
                  skip_color: bool = False,
                  skip_depth: bool = False,
                  skip_topview: bool = False,
                  skip_density: bool = False):
    """
    Train and render a scene with NERF

    Args:
        scene: Scene Name on S3 bucket
        config: NeRF Network Configuration
        display: Display result during rendering
    """
    # pylint:disable=too-many-arguments
    logger.info(f"Run NeRF Scene on {scene}")
    computer = SceneComputer(scene, config)

    logger.info("Train")
    computer.train()

    prefix_str = "GPU" if display else "CPU"
    if not skip_color:
        logger.info(f"Render Color on {prefix_str}")
        screenshot = computer.render("color", display=display)

    if not skip_topview:
        logger.info(f"Render Color Topview on {prefix_str}")
        _ = computer.render("color", "orthographic", topview=True, display=display)

    if not skip_depth:
        logger.info(f"Render Depth on {prefix_str}")
        depth_screenshot = computer.render("depth", display=display)
        logger.info("ToneMap DepthMap")
        color_depth = depth_screenshot+"_png"
        tonemap_folder(depth_screenshot, color_depth)

    if not skip_density:
        logger.info("Render Density")
        computer.extract_density()

    logger.info("Convert to video")
    color_video = os.path.join(computer.result_dir, "video.mp4")
    depth_video = os.path.join(computer.result_dir, "depth.mp4")
    if not skip_color:
        img_folder_to_video(screenshot, color_video, output_video_fps)
    if not skip_depth:
        img_folder_to_video(color_depth, depth_video, output_video_fps)
    if not skip_color and not skip_depth:
        result_video = os.path.join(computer.result_dir, "result.mp4")
        merge_videos([color_video, depth_video], result_video)

    logger.info("Save json result")
    computer.save_info()

    if ENABLE_S3_UPLOAD:
        logger.info("Upload result")
        computer.upload_scene(S3_URL_FORMAT.format(scene))

    export_profiling_events()
