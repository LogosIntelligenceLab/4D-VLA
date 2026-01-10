"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import torch

import imageio
import numpy as np
from PIL import Image
# import tensorflow as tf
import torchvision.transforms as transforms
try:
    from libero.libero import get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore
except ModuleNotFoundError:
    get_libero_path = None
    OffScreenRenderEnv = None
try:
    import robosuite.utils.camera_utils as CU  # type: ignore
except ModuleNotFoundError:
    CU = None
from scipy.spatial.transform import Rotation as R
try:
    from robosuite.utils.camera_utils import (  # type: ignore
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )
except ModuleNotFoundError:
    get_camera_extrinsic_matrix = None
    get_camera_intrinsic_matrix = None
import cv2

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def _require_libero_deps() -> None:
    if get_libero_path is None or OffScreenRenderEnv is None:
        raise ModuleNotFoundError(
            "Missing dependency: libero. Install LIBERO (and its deps) to run simulation eval."
        )
    if CU is None or get_camera_extrinsic_matrix is None or get_camera_intrinsic_matrix is None:
        raise ModuleNotFoundError(
            "Missing dependency: robosuite. Install it to run LIBERO simulation eval (e.g. `pip install robosuite`)."
        )


def get_libero_env(task, model_family, resolution=256, seed=0):
    """Initializes and returns the LIBERO environment, along with the task description."""
    _require_libero_deps()
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_depths": True}
    env = OffScreenRenderEnv(**env_args)
    print("seed: ", seed)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


# def resize_image(img, resize_size):
#     """
#     Takes numpy array corresponding to a single image and returns resized image as numpy array.

#     NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
#                     the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
#     """
#     assert isinstance(resize_size, tuple)
#     # Resize to image size expected by model
#     img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
#     img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
#     img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
#     img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
#     img = img.numpy()
#     return img

def resize_image(img, resize_size):
    """
    Takes a numpy array corresponding to a single image and returns resized image as a numpy array.
    
    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, 
                    we follow the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    
    # Convert numpy array to PIL image
    img = Image.fromarray(img)
    
    # Resize the image using Lanczos filter
    resize_transform = transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.LANCZOS)
    img = resize_transform(img)
    
    # Convert the image back to numpy array with uint8 type
    img = np.array(img, dtype=np.uint8)
    
    return img

def get_libero_image(obs, resize_size, use_2view=False, use_depth=False, env=None):
    """Extracts image from observations and preprocesses it."""
    _require_libero_deps()
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    # img = img[::-1, ::-1]
    img = resize_image(img, resize_size)

    if use_depth:
        agentview_depth = obs["agentview_depth"]
        agentview_depth = CU.get_real_depth_map(sim=env.sim, depth_map=agentview_depth)
        agentview_depth = agentview_depth[::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
        # agentview_depth = agentview_depth[::-1, ::-1]
        depth = cv2.resize(agentview_depth, resize_size, interpolation=cv2.INTER_LINEAR)
        depth_copy = depth.copy()
        depth = torch.tensor(depth_copy, dtype=torch.float).unsqueeze(0).squeeze(-1)

        camera_id = env.sim.model.camera_name2id('agentview')
        fovy = env.sim.model.cam_fovy[camera_id]
        width, height = resize_size
        fovy_rad = np.deg2rad(fovy)

        # 计算焦距 fy 和 fx
        fy = height / (2 * np.tan(fovy_rad / 2))
        fx = fy * (width / height)  # 由于宽高相等，fx = fy

        # 计算光心 (cx, cy)
        cx, cy = width / 2, height / 2

        K = np.array([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0,  0,  1, 0],
            [0,  0,  0, 1]
        ])

        intrinsics = torch.tensor(K, dtype=torch.float).unsqueeze(0)

        # camera_pos = env.sim.model.cam_pos[camera_id]
        # camera_quat = env.sim.model.cam_quat[camera_id]
        # pose_matrix = np.eye(4)
        # rotation_matrix = R.from_quat(camera_quat).as_matrix()
        # pose_matrix[:3, :3] = rotation_matrix
        # pose_matrix[:3, 3] = camera_pos
        camera_agentview_extrinsic = get_camera_extrinsic_matrix(env.sim, 'agentview')


        pose = torch.tensor(camera_agentview_extrinsic, dtype=torch.float).unsqueeze(0)


    if use_2view:
        wrist = obs['robot0_eye_in_hand_image']
        wrist = wrist[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
        wrist = resize_image(wrist, resize_size)
    
    return img, wrist if use_2view else None, depth if use_depth else None, intrinsics if use_depth else None, pose if use_depth else None

# def get_libero_image(obs, resize_size, use_2view=True, use_depth=False, env=None):
#     """Extracts image from observations and preprocesses it."""
#     assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
#     if isinstance(resize_size, int):
#         resize_size = (resize_size, resize_size)
#     img = obs["agentview_image"]
#     img = img[::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
#     # img = img[::-1, ::-1]
#     img = resize_image(img, resize_size)

#     if use_depth:
#         agentview_depth = obs["agentview_depth"]
#         agentview_depth = CU.get_real_depth_map(sim=env.sim, depth_map=agentview_depth)
#         agentview_depth = agentview_depth[::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
#         # agentview_depth = agentview_depth[::-1, ::-1]
#         depth = cv2.resize(agentview_depth, resize_size, interpolation=cv2.INTER_LINEAR)
#         depth_copy = depth.copy()
#         depth = torch.tensor(depth_copy, dtype=torch.float).unsqueeze(0).squeeze(-1)

#         camera_id = env.sim.model.camera_name2id('agentview')
#         fovy = env.sim.model.cam_fovy[camera_id]
#         width, height = resize_size
#         fovy_rad = np.deg2rad(fovy)

#         # 计算焦距 fy 和 fx
#         fy = height / (2 * np.tan(fovy_rad / 2))
#         fx = fy * (width / height)  # 由于宽高相等，fx = fy

#         # 计算光心 (cx, cy)
#         cx, cy = width / 2, height / 2

#         K = np.array([
#             [fx, 0, cx, 0],
#             [0, fy, cy, 0],
#             [0,  0,  1, 0],
#             [0,  0,  0, 1]
#         ])

#         intrinsics = torch.tensor(K, dtype=torch.float).unsqueeze(0)

#         # camera_pos = env.sim.model.cam_pos[camera_id]
#         # camera_quat = env.sim.model.cam_quat[camera_id]
#         # pose_matrix = np.eye(4)
#         # rotation_matrix = R.from_quat(camera_quat).as_matrix()
#         # pose_matrix[:3, :3] = rotation_matrix
#         # pose_matrix[:3, 3] = camera_pos
#         camera_agentview_extrinsic = get_camera_extrinsic_matrix(env.sim, 'agentview')


#         pose = torch.tensor(camera_agentview_extrinsic, dtype=torch.float).unsqueeze(0)


#     wrist = obs['robot0_eye_in_hand_image']
#     wrist = wrist[::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
#     wrist = resize_image(wrist, resize_size)

#     robot0_eye_in_hand_depth = obs["robot0_eye_in_hand_depth"]
#     robot0_eye_in_hand_depth = CU.get_real_depth_map(sim=env.sim, depth_map=robot0_eye_in_hand_depth)
#     robot0_eye_in_hand_depth = robot0_eye_in_hand_depth[::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
#     # agentview_depth = agentview_depth[::-1, ::-1]
#     depth_wrist = cv2.resize(robot0_eye_in_hand_depth, resize_size, interpolation=cv2.INTER_LINEAR)
#     depth_wrist_copy = depth_wrist.copy()
#     depth_wrist = torch.tensor(depth_wrist_copy, dtype=torch.float).unsqueeze(0).squeeze(-1)
    
#     return img, wrist, depth, depth_wrist, intrinsics if use_depth else None, pose if use_depth else None


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, steps=0, exp_name="sample"):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}_{exp_name}/"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--steps={steps}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

# def save_rollout_video_depth(rollout_depth, idx, success, task_description, log_file=None, steps=0, exp_name="sample"):
#     """Saves an MP4 replay of an episode."""
#     rollout_dir = f"./rollouts/{DATE}_{exp_name}_depth/"
#     os.makedirs(rollout_dir, exist_ok=True)
#     processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
#     mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--steps={steps}--task={processed_task_description}.mp4"
#     video_writer = imageio.get_writer(mp4_path, fps=30)
#     for img in rollout_images:
#         video_writer.append_data(img)
#     video_writer.close()
#     print(f"Saved rollout MP4 at path {mp4_path}")
#     if log_file is not None:
#         log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
#     return mp4_path

def save_rollout_video_depth(rollout_depth, idx, success, task_description, log_file=None, steps=0, exp_name="sample"):
    """Saves an MP4 replay of an episode using depth frames, normalizing depth values to 0-255 and applying a colormap."""
    rollout_dir = f"./rollouts/{DATE}_{exp_name}_depth/"
    os.makedirs(rollout_dir, exist_ok=True)

    # Process the task description for a safe filename
    processed_task_description = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )

    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--steps={steps}--task={processed_task_description}.mp4"

    # Flatten all frames to determine global min and max for normalization
    depth_min = np.min(rollout_depth)
    depth_max = np.max(rollout_depth)

    if depth_max > depth_min:  # Avoid division by zero
        normalized_rollout_depth = (rollout_depth - depth_min) / (depth_max - depth_min) * 255.0
        normalized_rollout_depth = np.clip(normalized_rollout_depth, 0, 200) / 200.0 * 255.0
    else:
        normalized_rollout_depth = np.zeros_like(rollout_depth)  # If all values are the same

    # Convert normalized depths to uint8
    normalized_rollout_depth = normalized_rollout_depth.astype(np.uint8).squeeze(1)

    # Create a video writer and write all frames with a colormap
    video_writer = imageio.get_writer(mp4_path, fps=30)

    for depth_frame in normalized_rollout_depth:
        # Apply a colormap (e.g., COLORMAP_JET for purple-yellow)
        colored_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_PLASMA)
        video_writer.append_data(colored_frame[:, :, ::-1])  # Convert BGR to RGB

    video_writer.close()

    print(f"Saved rollout MP4 at path {mp4_path}")

    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")

    return mp4_path

def save_rollout_video_proprio(rollout_proprio, idx, success, task_description, log_file=None, steps=0, exp_name="sample"):
    """Saves an MP4 replay visualizing the trajectory of points in 3D space with fading color and limited frame display."""
    rollout_dir = f"./rollouts/{DATE}_{exp_name}_proprio/"
    os.makedirs(rollout_dir, exist_ok=True)

    # Process the task description for a safe filename
    processed_task_description = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )

    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--steps={steps}--task={processed_task_description}.mp4"

    # Setup the figure and 3D axis for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Trajectory Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set limits dynamically based on the data
    rollout_proprio = np.array(rollout_proprio)
    ax.set_xlim(np.min(rollout_proprio[:, 0]), np.max(rollout_proprio[:, 0]))
    ax.set_ylim(np.min(rollout_proprio[:, 1]), np.max(rollout_proprio[:, 1]))
    ax.set_zlim(np.min(rollout_proprio[:, 2]), np.max(rollout_proprio[:, 2]))

    # Setup the video writer
    metadata = dict(title="Trajectory Visualization", artist="Matplotlib", comment="Trajectory video")
    writer = FFMpegWriter(fps=30, metadata=metadata)

    with writer.saving(fig, mp4_path, dpi=200):
        for i in range(1, len(rollout_proprio)):
            ax.cla()  # Clear the axis for the new frame
            ax.set_title("Trajectory Visualization")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(np.min(rollout_proprio[:, 0]), np.max(rollout_proprio[:, 0]))
            ax.set_ylim(np.min(rollout_proprio[:, 1]), np.max(rollout_proprio[:, 1]))
            ax.set_zlim(np.min(rollout_proprio[:, 2]), np.max(rollout_proprio[:, 2]))
            ax.view_init(elev=30, azim=0)

            start_frame = max(0, i - 40)  # Only show the last 5 frames
            for j in range(start_frame, i):
                alpha = (j - start_frame + 1) / 40  # Gradually fade color
                ax.plot(
                    rollout_proprio[j:j+2, 0],
                    rollout_proprio[j:j+2, 1],
                    rollout_proprio[j:j+2, 2],
                    color=(0.5, 0, 0.5, alpha),  # Blue with fading alpha
                    linewidth=6
                )

            writer.grab_frame()

    plt.close(fig)

    print(f"Saved rollout MP4 at path {mp4_path}")

    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")

    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
