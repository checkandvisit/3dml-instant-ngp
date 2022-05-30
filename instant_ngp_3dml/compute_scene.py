# /usr/bin/python3
"""Compute NeRF Scene"""
import os
from time import process_time
from typing import Any, Literal
from typing import Dict, get_args
from typing import Tuple

from instant_ngp_3dml.rendering import render
from instant_ngp_3dml.training import train
from instant_ngp_3dml.utils import DATA_DIR
from instant_ngp_3dml.utils import NERF_CONFIG
from instant_ngp_3dml.utils.io import read_json
from instant_ngp_3dml.utils.io import write_json
from instant_ngp_3dml.utils.log import logger
from instant_ngp_3dml.utils.profiler import export_profiling_events
from instant_ngp_3dml.utils.profiler import profile
from instant_ngp_3dml.utils.tonemapper import tonemap_folder

DEFAULT_MAX_STEP = 20000
DEBUG = False
ENABLE_S3_UPLOAD = False
S3_URL_FORMAT = "s3://checkandvisit-3dml-dev/dataset_test/nerf/{scene_name}/"

SceneName = Literal["lego", "barbershop", "0223-1010", "0223-1118", "0223-1120"]


class SceneComputer:
    """Train and render a test scene with NERF."""

    def __init__(self, scene_name: SceneName, config: str):
        assert scene_name in set(get_args(SceneName)), \
            f"Unknown test scene name, should be in {set(get_args(SceneName))}"
        self.scene_dir = os.path.join(DATA_DIR, scene_name)
        if not os.path.isdir(self.scene_dir):
            logger.info("Download data from S3")
            self.download_scene(S3_URL_FORMAT.format(scene_name))

        self.config_path = NERF_CONFIG.format(config=config)

        self.result_dir = os.path.join(self.scene_dir, config)
        self.snapshot_dir = os.path.join(self.result_dir, "snapshot")
        self.snapshot = os.path.join(self.snapshot_dir, "snap_"+"{idx}.msgpack")

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
            if os.path.isfile(self.snapshot.format(idx=(i+1)*n_step)):
                logger.info("Snapshot already exist, continue...")
                continue

            train(scene=training_json,
                  save_snapshot=self.snapshot.format(idx=(i+1)*n_step),
                  load_snapshot=self.snapshot.format(
                      idx=i*n_step) if i > 0 else "",
                  n_steps=(i+1) * n_step,
                  network=self.config_path)
        end_time = process_time()

        self.info["n_step"] = n_step
        self.info["max_step"] = max_step
        self.info["training_time"] = end_time-start_time

    @profile
    def render(self, depth: bool, step_idx: int = DEFAULT_MAX_STEP, display: bool = False) -> str:
        """Render Scene."""

        test_json = os.path.join(self.scene_dir, "test.json")
        test_data = read_json(test_json)

        screenshot = os.path.join(
            self.result_dir, "screenshot"+("_depth" if depth else "")+("_gl" if display else ""))
        os.makedirs(screenshot, exist_ok=True)

        width = int(test_data["w"])
        height = int(test_data["h"])

        start_time = process_time()
        render(load_snapshot=self.snapshot.format(idx=step_idx),
               screenshot_transforms=test_json,
               screenshot_dir=screenshot,
               width=width,
               height=height,
               depth=depth,
               display=display)
        end_time = process_time()
        self.info["render_time"] = end_time-start_time

        return screenshot

    def color_depth(self, depth_screenshot: str) -> str:
        """Color Depth."""

        color_depth = os.path.join(self.result_dir, "color_depth")
        tonemap_folder(depth_screenshot, color_depth)

        return color_depth

    def save_info(self) -> str:
        """Save Info."""

        info_json = os.path.join(self.result_dir, "info.json")
        write_json(info_json, self.info)
        return info_json

    def upload_scene(self, scene_url: str):
        """Upload Scene."""

        cmd = f"aws s3 sync {self.scene_dir} {scene_url}"
        os.system(cmd)

    def convert_to_video(self, color: str, depth: str) -> Tuple[str, str, str]:
        """Convert color and depth to video."""

        video = os.path.join(self.result_dir, "video.mp4")
        depth_video = os.path.join(self.result_dir, "depth.mp4")
        result_video = os.path.join(self.result_dir, "result.mp4")

        cmd = "ffmpeg -r 30 -y"
        cmd += " -i "+os.path.join(color, "frame_%04d.png")
        cmd += " -c:v libx264"
        cmd += " -vf fps=30"
        cmd += f" {video}"
        os.system(cmd)

        cmd = "ffmpeg -r 30 -y"
        cmd += " -i "+os.path.join(depth, "frame_%04d.png")
        cmd += " -c:v libx264"
        cmd += " -vf fps=30"
        cmd += f" {depth_video}"
        os.system(cmd)

        cmd = "ffmpeg -y"
        cmd += f" -i {video}"
        cmd += f" -i {depth_video}"
        cmd += " -filter_complex hstack"
        cmd += f" {result_video}"
        os.system(cmd)

        return video, depth_video, result_video


def compute_scene(scene: SceneName,
                  config: str = "base",
                  display: bool = False):
    """
    Train and render a scene with NERF

    Args:
        scene: Scene Name on S3 bucket
        config: NeRF Network Configuration
        display: Display result during rendering
    """

    logger.info(f"Run NeRF Scene on {scene}")
    computer = SceneComputer(scene,
                             config)

    logger.info("Train")
    computer.train()

    if display or DEBUG:
        logger.info("Render Result on CPU")
        screenshot = computer.render(False, display=True)

        logger.info("Render Depth on CPU")
        depth_screenshot = computer.render(True, display=True)

    if not display or DEBUG:
        logger.info("Render Result on GPU")
        screenshot = computer.render(False, display=False)

        logger.info("Render Depth on GPU")
        depth_screenshot = computer.render(True, display=False)

    logger.info("ToneMap DepthMap")
    color_depth = computer.color_depth(depth_screenshot)

    logger.info("Convert to video")
    _, _, _ = computer.convert_to_video(screenshot, color_depth)

    logger.info("Save json result")
    computer.save_info()

    if ENABLE_S3_UPLOAD:
        logger.info("Upload result")
        computer.upload_scene(S3_URL_FORMAT.format(scene))

    export_profiling_events()
