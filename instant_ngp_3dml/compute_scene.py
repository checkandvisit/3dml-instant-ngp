# /usr/bin/python3
"""Compute NeRF Scene"""
import os
from time import process_time
from typing import Any
from typing import Dict
from typing import get_args
from typing import List
from typing import Literal

from instant_ngp_3dml.density import main as density
from instant_ngp_3dml.rendering import main as render
from instant_ngp_3dml.training import main as train
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

SceneName = Literal["lego", "barbershop", "0223-1010", "0223-1118", "0223-1120", "kitchen_refined", "voiture"]


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
        self.scene_folder = os.path.join(DATA_DIR, scene_name)

        logger.info("Download data from S3")
        self.download_scene(S3_URL_FORMAT.format(scene_name=scene_name))

        self.nerf_config_json = NERF_CONFIG.format(config=config)

        self.result_folder = os.path.join(self.scene_folder, config)
        self.snapshot_folder = os.path.join(self.result_folder, "snapshot")

        os.makedirs(self.result_folder, exist_ok=True)

        self.info: Dict[str, Any] = {}

    def download_scene(self, scene_url: str):
        """Download Scene"""

        os.makedirs(self.scene_folder, exist_ok=True)

        cmd = f"aws s3 sync {scene_url} {self.scene_folder}"
        os.system(cmd)

    @profile
    def train(self, n_step: int = 2000, max_step: int = DEFAULT_MAX_STEP):
        """Train Scene."""

        transform_json = os.path.join(self.scene_folder, "training.json")
        if not os.path.isfile(transform_json):
            transform_json = os.path.join(self.scene_folder, "transform.json")

        os.makedirs(self.snapshot_folder, exist_ok=True)
        training_info_json = os.path.join(self.snapshot_folder, "training_info.json")

        start_time = process_time()
        for i in range(int(max_step/n_step)):
            logger.info(f"--- Run Step {i*n_step} to {(i + 1) * n_step} ---")
            snapshot_msgpack = get_snapshot_path(self.snapshot_folder, (i+1)*n_step)
            if os.path.isfile(snapshot_msgpack):
                logger.info("Snapshot already exists, continue...")
                continue
            prev_snapshot_path = get_snapshot_path(self.snapshot_folder, i*n_step) if i > 0 else ""
            train(nerf_transform_json=transform_json,
                  out_snapshot_msgpack=snapshot_msgpack,
                  snapshot_msgpack=prev_snapshot_path,
                  n_steps=(i+1) * n_step,
                  nerf_network_configuration_json=self.nerf_config_json,
                  out_training_info_json=training_info_json,
                  enable_depth_supervision=True)
        end_time = process_time()

        self.info["n_step"] = n_step
        self.info["max_step"] = max_step
        self.info["training_time"] = end_time-start_time

    @profile
    def render(self,
               render_mode: str = "color",
               camera_mode: str = "perspective",
               topview: bool = False,
               step_idx: int = DEFAULT_MAX_STEP,
               display: bool = False, num_max_images: int = -1) -> str:
        """Render Scene."""
        # pylint:disable=too-many-arguments
        transform_json = os.path.join(self.scene_folder, "topview.json" if topview else "test.json")
        if not os.path.isfile(transform_json):
            if topview:
                return ""
            transform_json = os.path.join(self.scene_folder, "transform.json")

        start_time = process_time()

        render_folder = os.path.join(self.result_folder, f"{render_mode}_{camera_mode}")
        render(snapshot_msgpack=get_snapshot_path(self.snapshot_folder, step_idx),
               nerf_transform_json=transform_json,
               out_rendering_folder=render_folder,
               display=display,
               num_max_images=num_max_images,
               render_mode=render_mode,
               camera_mode=camera_mode)
        end_time = process_time()
        self.info["render_time"] = end_time-start_time

        return render_folder

    @profile
    def extract_density(self,
                        nerf_config_json: str,
                        step_idx: int = DEFAULT_MAX_STEP) -> str:
        """Extract Density"""

        out_density_folder = os.path.join(self.scene_folder, "density")

        start_time = process_time()
        density(snapshot_msgpack=get_snapshot_path(self.snapshot_folder, step_idx),
                nerf_config_json=nerf_config_json,
                out_density_folder=out_density_folder,
                vox_by_m=50)
        end_time = process_time()
        self.info["density_time"] = end_time-start_time

        return out_density_folder

    def save_info(self) -> str:
        """Save Info."""

        info_json = os.path.join(self.result_folder, "info.json")
        write_json(info_json, self.info)
        return info_json

    def upload_scene(self, scene_url: str):
        """Upload Scene."""

        cmd = f"aws s3 sync {self.scene_folder} {scene_url}"
        os.system(cmd)


def compute_scene(scene: SceneName,
                  config: str = "base",
                  display: bool = False,
                  out_video_fps: int = 2,
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
        out_video_fps: Fps for output video
        skip_color: Skip color rendering part
        skip_depth: Skip depth rendering part
        skip_topview: Skip topview rendering part
        skip_density: Skip density extraction part
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
        nerf_config_file = os.path.join(computer.scene_folder, "config.json")
        if os.path.isfile(nerf_config_file):
            logger.info("Render Density")
            computer.extract_density(nerf_config_file)

    logger.info("Convert to video")
    color_video = os.path.join(computer.result_folder, "video.mp4")
    depth_video = os.path.join(computer.result_folder, "depth.mp4")
    if not skip_color:
        img_folder_to_video(screenshot, color_video, out_video_fps)
    if not skip_depth:
        img_folder_to_video(color_depth, depth_video, out_video_fps)
    if not skip_color and not skip_depth:
        result_video = os.path.join(computer.result_folder, "result.mp4")
        merge_videos([color_video, depth_video], result_video)

    logger.info("Save json result")
    computer.save_info()

    if ENABLE_S3_UPLOAD:
        logger.info("Upload result")
        computer.upload_scene(S3_URL_FORMAT.format(scene))

    export_profiling_events()
