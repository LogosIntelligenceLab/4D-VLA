"""
prepare_libero_memory_bank_mb.py

Generates a Libero RGBD multi-frame jsonl (and writes per-frame images/depth PNGs) using a simple
memory bank selection strategy (MB).

This script is part of the "generate ft data" stage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _add_internvl_chat_to_sys_path() -> Path:
    # This script lives at: internvl_chat/tools/prepare_libero_memory_bank_mb.py
    # We want `internvl_chat/` on sys.path so `import internvl...` works no matter the CWD.
    internvl_chat_root = Path(__file__).resolve().parents[1]
    internvl_chat_root_str = str(internvl_chat_root)
    if internvl_chat_root_str not in sys.path:
        sys.path.insert(0, internvl_chat_root_str)
    return internvl_chat_root


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_resolution(values: list[str]) -> Tuple[int, int]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("resolution must be 2 integers: H W")
    h, w = int(values[0]), int(values[1])
    if h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError("resolution values must be > 0")
    return h, w


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_episode_features(feature_extractor, image_episode: np.ndarray, batch_size: int) -> np.ndarray:
    """Extract ResNet features for an episode in chunks to avoid OOM."""
    if batch_size <= 0:
        raise ValueError(f"Invalid feature batch size: {batch_size}")
    features = []
    for start in range(0, len(image_episode), batch_size):
        features.append(feature_extractor.extract_image_feature(image_episode[start:start + batch_size]))
    return np.concatenate(features, axis=0) if features else np.empty((0,), dtype=np.float32)


def _maybe_import_cv2():
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2


def _save_depth_png(path: Path, depth: np.ndarray) -> None:
    """
    Saves a single-channel depth image to PNG.

    Prefer OpenCV if installed; fallback to PIL for uint16-like arrays.
    """
    cv2 = _maybe_import_cv2()
    if cv2 is not None:
        ok = cv2.imwrite(str(path), depth)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed: {path}")
        return

    depth = np.asarray(depth)
    if depth.dtype != np.uint16:
        # Best-effort: keep integer depth without changing units.
        if np.issubdtype(depth.dtype, np.floating):
            depth = np.clip(depth, 0, 65535).astype(np.uint16)
        else:
            depth = depth.astype(np.uint16, copy=False)
    Image.fromarray(depth, mode="I;16").save(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Libero RGBD multi-frame jsonl + saved frames using memory bank (MB).",
    )
    parser.add_argument("--data-root-dir", type=str, default=None, help="Root dir containing RLDS datasets.")
    parser.add_argument("--data-mix", type=str, default="libero_object_no_noops_3d", help="OXE dataset name.")
    parser.add_argument("--resize-resolution", nargs=2, default=["448", "448"], help="Resize resolution: H W.")
    parser.add_argument("--shuffle-buffer-size", type=int, default=0)
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--no-train", dest="train", action="store_false")
    parser.add_argument("--image-aug", action="store_true", default=False)

    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--bank-size", type=int, default=5)
    parser.add_argument("--feature-batch-size", type=int, default=64, help="ResNet feature extraction batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-episodes", type=int, default=-1, help="Process at most N episodes (-1 = all).")
    parser.add_argument("--max-frames", type=int, default=-1, help="Process at most N frames per episode (-1 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Only validate dataset access, do not write outputs.")

    args = parser.parse_args(argv)

    internvl_chat_root = _add_internvl_chat_to_sys_path()
    repo_root = internvl_chat_root.parent

    data_root_dir = Path(args.data_root_dir) if args.data_root_dir else (repo_root / "data")
    resize_h, resize_w = _parse_resolution(list(args.resize_resolution))
    if args.window_size <= 0 or args.bank_size <= 0:
        raise ValueError("--window-size and --bank-size must be > 0")

    if not data_root_dir.exists():
        raise FileNotFoundError(f"--data-root-dir not found: {data_root_dir}")
    dataset_dir_candidate = data_root_dir / args.data_mix
    if not dataset_dir_candidate.exists():
        print(
            f"[WARN] dataset directory does not exist: {dataset_dir_candidate}\n"
            "Continuing anyway (some TFDS/RLDS layouts do not use a simple `<root>/<dataset_name>` folder).",
            file=sys.stderr,
        )

    device = _resolve_device(args.device)
    set_seed(args.seed)

    # Delay heavy imports until after basic argument validation.
    try:
        from internvl.vla.datasets import EpisodicRLDSDataset
    except ModuleNotFoundError as e:
        missing = e.name or ""
        known_missing_prefixes = (
            "prismatic",
            "tensorflow",
            "rlds",
        )
        known_missing_exact = {
            "dlimp",
        }
        if missing in known_missing_exact or any(missing.startswith(p) for p in known_missing_prefixes):
            print(
                "[ERROR] Missing dependency required by RLDS/OXE dataset pipeline.\n"
                f"Missing module: {missing}\n"
                "Fix: install the missing deps in your conda/pip env (the RLDS stack is not vendored in this repo).\n"
                f"Original error: {e}",
                file=sys.stderr,
            )
            return 2
        raise

    from internvl.model.internvl_chat.memory_bank_exp2 import SimpleMemoryBank

    dataset = EpisodicRLDSDataset(
        data_root_dir,
        args.data_mix,
        resize_resolution=(resize_h, resize_w),
        shuffle_buffer_size=args.shuffle_buffer_size,
        train=args.train,
        image_aug=args.image_aug,
        load_depth=True,
        load_camera_info=True,
    )

    if args.dry_run:
        print(f"[OK] dataset_length={dataset.dataset_length} mix={args.data_mix} data_root_dir={data_root_dir}")
        return 0

    memory_bank = SimpleMemoryBank(window_size=args.window_size, bank_size=args.bank_size)
    memory_bank.feature_extractor.resnet50.to(device)

    output_image_dir = data_root_dir / args.data_mix / f"images_v2_mb_exp2_ws{args.window_size}_bs{args.bank_size}_rgbd"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    zero_rgb_path = output_image_dir / "zero.jpg"
    zero_depth_path = output_image_dir / "zero_depth.png"
    if not zero_rgb_path.exists():
        Image.fromarray(np.zeros((resize_h, resize_w, 3), dtype=np.uint8)).save(zero_rgb_path)
    if not zero_depth_path.exists():
        _save_depth_png(zero_depth_path, np.zeros((resize_h, resize_w), dtype=np.uint16))

    json_entries = []
    total_frames = 0
    dataset_length = int(dataset.dataset_length)
    max_episodes = dataset_length if args.max_episodes < 0 else min(args.max_episodes, dataset_length)

    for ep_idx, rlds_batch in enumerate(
        tqdm(dataset.dataset.as_numpy_iterator(), total=dataset_length, desc="Processing episodes")
    ):
        if ep_idx >= max_episodes:
            break

        dataset_name = rlds_batch["dataset_name"][0].decode()
        lang = rlds_batch["task"]["language_instruction"][0].decode().lower()

        action_ep = rlds_batch["action"]
        if action_ep.ndim == 3:
            action_ep = action_ep[:, 0]
        img_ep = rlds_batch["observation"]["image_primary"]
        if img_ep.ndim == 5:
            img_ep = img_ep[:, 0]
        img_wrist_ep = rlds_batch["observation"]["image_wrist"]
        if img_wrist_ep.ndim == 5:
            img_wrist_ep = img_wrist_ep[:, 0]

        img_ep_d = rlds_batch["depth"]["depth_primary"]
        if img_ep_d.ndim == 4:
            img_ep_d = img_ep_d[:, 0]
        img_wrist_ep_d = rlds_batch["depth"]["depth_wrist"]
        if img_wrist_ep_d.ndim == 4:
            img_wrist_ep_d = img_wrist_ep_d[:, 0]

        camera_info_agentview_extrinsic = rlds_batch["camera_info"]["camera_info_agentview_extrinsic"]
        camera_info_robothand_extrinsic = rlds_batch["camera_info"]["camera_info_robothand_extrinsic"]
        camera_info_fovy = rlds_batch["camera_info"]["camera_info_fovy"]

        # Features for the whole episode (chunked).
        img_feat_ep = _extract_episode_features(memory_bank.feature_extractor, img_ep, args.feature_batch_size)

        memory_bank.clear()

        num_frames = len(action_ep)
        if args.max_frames >= 0:
            num_frames = min(num_frames, args.max_frames)

        for fid in range(num_frames):
            total_frames += 1

            image_path = f"ep{ep_idx}-fid{fid}.jpg"
            image_path_wrist = f"ep{ep_idx}-fid{fid}_wrist.jpg"
            image_depth_path = f"ep{ep_idx}-fid{fid}_depth.png"
            image_depth_wrist_path = f"ep{ep_idx}-fid{fid}_depth_wrist.png"

            Image.fromarray(img_ep[fid]).save(output_image_dir / image_path)
            Image.fromarray(img_wrist_ep[fid]).save(output_image_dir / image_path_wrist)
            _save_depth_png(output_image_dir / image_depth_path, img_ep_d[fid])
            _save_depth_png(output_image_dir / image_depth_wrist_path, img_wrist_ep_d[fid])

            # Update memory bank with the current frame feature.
            memory_bank.add(img_feat_ep[fid], fid, from_img=False)

            fids = list(memory_bank.get_fid())
            if fid not in fids:
                # Ensure current frame is always present.
                fids = fids[-(args.bank_size - 1):] + [fid]
            history_fids = [x for x in fids if x != fid]
            # Left-pad with zeros if we do not have enough history yet.
            while len(history_fids) < args.bank_size - 1:
                history_fids.insert(0, -1)
            history_fids = history_fids[-(args.bank_size - 1):]

            rounded_action = [round(float(x), 5) for x in action_ep[fid]]
            entry = {
                "id": total_frames,
                "image": [],
                "image_depth": [],
                "width_list": [],
                "height_list": [],
                "width_depth_list": [],
                "height_depth_list": [],
                "conversations": [],
                "proprio": [],
                "action": rounded_action,
                "camera_info": {
                    "agentview_extrinsic": np.round(camera_info_agentview_extrinsic[fid], 4).tolist(),
                    "robothand_extrinsic": np.round(camera_info_robothand_extrinsic[fid], 4).tolist(),
                    "fovy": camera_info_fovy[fid].tolist(),
                },
            }

            # History frames (selected by memory bank) + current frame.
            for f in history_fids:
                if f < 0:
                    entry["image"].append(zero_rgb_path.name)
                    entry["image_depth"].append(zero_depth_path.name)
                    entry["width_list"].append(resize_w)
                    entry["height_list"].append(resize_h)
                    entry["width_depth_list"].append(resize_w)
                    entry["height_depth_list"].append(resize_h)
                else:
                    entry["image"].append(f"ep{ep_idx}-fid{f}.jpg")
                    entry["image_depth"].append(f"ep{ep_idx}-fid{f}_depth.png")
                    entry["width_list"].append(resize_w)
                    entry["height_list"].append(resize_h)
                    # Depth shape may differ; record actual dimensions.
                    d_h, d_w = int(img_ep_d[f].shape[0]), int(img_ep_d[f].shape[1])
                    entry["width_depth_list"].append(d_w)
                    entry["height_depth_list"].append(d_h)

            entry["image"].append(f"ep{ep_idx}-fid{fid}.jpg")
            entry["image_depth"].append(f"ep{ep_idx}-fid{fid}_depth.png")
            entry["width_list"].append(resize_w)
            entry["height_list"].append(resize_h)
            d_h, d_w = int(img_ep_d[fid].shape[0]), int(img_ep_d[fid].shape[1])
            entry["width_depth_list"].append(d_w)
            entry["height_depth_list"].append(d_h)

            num_history_images = args.bank_size - 1
            q = {
                "from": "human",
                "value": (
                    "History:\n"
                    + "<image>\n" * num_history_images
                    + "Current:\n<image>\n"
                    + f"What action should the robot take to {lang}"
                ),
            }
            a = {"from": "gpt", "value": "<state_pred>"}
            entry["conversations"].append(q)
            entry["conversations"].append(a)

            json_entries.append(entry)

    len_k = len(json_entries) // 1000
    jsonl_name = f"{args.data_mix}_{len_k}k_v2_mb_exp2_ws{args.window_size}_bs{args.bank_size}_rgbd.jsonl"
    jsonl_path = data_root_dir / jsonl_name

    with open(jsonl_path, "w") as f:
        for entry in json_entries:
            json.dump(entry, f)
            f.write("\n")

    print(f"[OK] processed_frames={total_frames} saved_jsonl={jsonl_path} images_dir={output_image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
