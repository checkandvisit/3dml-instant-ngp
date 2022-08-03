#!/usr/bin/env python3
"""Training Script"""
import json
import os
import time

import pyngp as ngp  # noqa
from tqdm import tqdm

from instant_ngp_3dml.utils.log import logger


def train(scene: str = "",
          network: str = "",
          load_snapshot: str = "",
          save_snapshot: str = "",
          n_steps: int = -1,
          training_info: str = "",
          enable_depth_supervision: bool = False):
    """Train NeRF Scene"""
    # pylint: disable=too-many-arguments,no-member

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    if scene != "":
        testbed.load_training_data(scene)

    if load_snapshot != "":
        logger.info(f"Loading snapshot {load_snapshot}")
        testbed.load_snapshot(load_snapshot)
    else:
        assert network != "", "Snapshot or Network need to be defined"
        testbed.reload_network_from_file(network)

    testbed.shall_train = True

    testbed.nerf.render_with_camera_distortion = True

    if not enable_depth_supervision:
        testbed.nerf.training.depth_supervision_lambda = 0.0

    old_training_step = 0
    if n_steps < 0:
        n_steps = 100000

    step_info = []
    if n_steps > 0:
        begin_time = time.monotonic()
        tqdm_last_update = 0.0
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while testbed.frame():

                # What will happen when training is done?
                if testbed.training_step >= n_steps:
                    break

                # Update progress bar
                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()

                if enable_depth_supervision:
                    depth_supervision_lambda = max(1.0 - testbed.training_step / 2000, 0.2)
                    testbed.nerf.training.depth_supervision_lambda = depth_supervision_lambda

                now = time.monotonic()

                step_info.append({
                    "step": testbed.training_step,
                    "loss": testbed.loss,
                    "time": now,
                    "depth_supervision_lambda": depth_supervision_lambda})

                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss, depth=depth_supervision_lambda)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

        end_time = time.monotonic()

    if save_snapshot != "":
        logger.info(f"Saving snapshot {save_snapshot}")
        testbed.save_snapshot(save_snapshot, False)

    if training_info != "":
        logger.info(f"Save training info {training_info}")

        info = {
            "begin_time": begin_time,
            "end_time": end_time,
            "step_info": step_info,
            "n_steps": n_steps,
            "enable_depth_supervision": enable_depth_supervision
        }

        os.makedirs(os.path.dirname(training_info), exist_ok=True)
        with open(training_info, "w", encoding="utf-8") as _f:
            json.dump(info, _f, indent=4)
