"""
run_bridgev2_eval.py

Runs a model in a real-world Bridge V2 environment.

Usage:
    # OpenVLA:
    python experiments/robot/bridge/run_bridgev2_eval.py --model_family openvla --pretrained_checkpoint openvla/openvla-7b
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import draccus

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from experiments.robot.bridge.bridgev2_utils import (
    get_next_task_label,
    get_preprocessed_image,
    get_widowx_env,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)

import simplejpeg
from PIL import Image
import numpy as np
import requests
import pickle


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])

def _make_request(uri: str, element):
    response = requests.post(uri, data=pickle.dumps(element))
    if response.status_code != 200:
        raise Exception(response.text)

    action = pickle.loads(response.content)
    # print(f"model output: {action}")
    return action["actions"]


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = ""                # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    host_ip: str = "localhost"
    port: int = 5556

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, -0.09, 0.26])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = True                                      # Whether to use blocking control
    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 200                                         # Max number of timesteps per episode
    control_frequency: float = 2.5                                # WidowX control frequency

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = False                                     # Whether to save rollout data (images, actions, etc.)

    # fmt: on


@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = "bridge_orig"

    # Load model
    # model = get_model(cfg)
    model = None

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize the WidowX environment
    env = get_widowx_env(cfg, model)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user
        task_label = get_next_task_label(task_label)

        # Reset environment
        obs, _ = env.reset()

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []

        # Start episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if (curr_tstamp > last_tstamp + step_duration):
                    if t % 5 == 0:
                        print(f"t: {t}")
                        print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                        last_tstamp = time.time()

                        # Refresh the camera image and proprioceptive state
                        obs = refresh_obs(obs, env)

                        # Save full (not preprocessed) image for replay video
                        replay_images.append(obs["full_image"])

                        #################### OPENVLA ####################
                        # Get preprocessed image
                        # obs["full_image"] = get_preprocessed_image(obs, resize_size)

                        # # Query model to get action
                        # action = get_action(
                        #     cfg,
                        #     model,
                        #     obs,
                        #     task_label,
                        #     processor=processor,
                        # )
                        ##################################################

                        bridge_image = obs["full_image"]
                        bridge_state = obs["proprio"]
                        bridge_instruction = task_label
                        #obs_image = resize_with_pad(bridge_image, 256, 320)
                        # obs_image = resize_with_pad(resize_with_pad(bridge_image, 256, 256), 256, 320)
                        bridge_resized = Image.fromarray(bridge_image).resize((256, 256))
                        obs_image = resize_with_pad(np.array(bridge_resized), 224, 224)
                        # obs_image = resize_with_pad(np.array(bridge_resized), 256, 320)
                        
                        element = {
                            "observation/image_0": np.array(obs_image),  # bridge_image is 256x256,
                            "observation/image_0_mask": np.array(True),
                            # "observation/image_1": np.zeros((256, 320, 3), dtype=np.uint8),
                            "observation/image_1": np.zeros((224, 224, 3), dtype=np.uint8),
                            "observation/image_1_mask": np.array(False),
                            "observation/state": bridge_state[:-1],   # we zero out the state for this model
                            "raw_text": bridge_instruction,     # a simple string
                        }
                        # Image.fromarray(element["observation/image_0"]).save(f"/tmp/images/{int(time.time())}.jpeg")

                        # this returns action chunk [4, 7] of 4 eef velocity actions (6) + gripper delta (1)
                        actions = _make_request("http://0.0.0.0:8000/infer", element)
                        action = actions[0]
                    else:
                        action = actions[t % 5]
                    

                    # [If saving rollout data] Save preprocessed image, robot state, and action
                    if cfg.save_data:
                        rollout_images.append(obs["full_image"])
                        rollout_states.append(obs["proprio"])
                        rollout_actions.append(action)

                    # Execute action
                    print("action:", action)
                    obs, _, _, _, _ = env.step(action)
                    t += 1

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    raise e
                break

        # Save a replay video of the episode
        # save_rollout_video(replay_images, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Redo episode or continue
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()