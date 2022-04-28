#!/usr/bin/env python3
"""Training Script"""
import pyngp as ngp  # noqa
from tqdm import tqdm

from instant_ngp_3dml.utils.log import logger


def train(scene: str = "", network: str = "", load_snapshot: str = "", save_snapshot: str = "", n_steps: int = -1):
    """Train NeRF Scene"""

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

    old_training_step = 0
    if n_steps < 0:
        n_steps = 100000

    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while testbed.frame():

                # What will happen when training is done?
                if testbed.training_step >= n_steps:
                    break

                # Update progress bar
                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()

                t.update(testbed.training_step - old_training_step)
                t.set_postfix(loss=testbed.loss)
                old_training_step = testbed.training_step

    if save_snapshot != "":
        logger.info(f"Saving snapshot {save_snapshot}")
        testbed.save_snapshot(save_snapshot, False)
