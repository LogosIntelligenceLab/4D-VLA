#!/usr/bin/env python3
"""
Validate a dataset meta JSON used by InternVL training scripts.

Usage:
  # from repo root:
  python internvl_chat/tools/validate_meta.py internvl_chat/shell/exp_data/droid_pretrain_mb_20ws_5bs_simple.json

  # from internvl_chat/:
  python tools/validate_meta.py shell/exp_data/droid_pretrain_mb_20ws_5bs_simple.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_KEYS = {"root", "annotation", "data_augment", "repeat_time", "length"}


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def _infer_internvl_chat_root(meta_path: Path) -> Path | None:
    """
    Most meta files in this repo use paths relative to `internvl_chat/` (not relative to the meta file itself).
    Make validation robust regardless of current working directory.
    """
    # 1) Try to find a parent dir that looks like internvl_chat root.
    for p in [meta_path.parent, *meta_path.parents]:
        if (p / "internvl").exists() and (p / "shell").exists():
            return p
    # 2) Try CWD/internvl_chat if called from repo root.
    cwd_candidate = Path.cwd() / "internvl_chat"
    if (cwd_candidate / "internvl").exists() and (cwd_candidate / "shell").exists():
        return cwd_candidate
    return None


def _resolve_path(p: Path, base: Path | None) -> Path:
    if p.is_absolute():
        return p
    if p.exists():
        return p
    if base is not None:
        alt = (base / p).resolve()
        return alt
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=Path, help="Path to meta JSON (e.g. shell/exp_data/*.json)")
    parser.add_argument(
        "--check-length",
        action="store_true",
        help="Also count lines in each annotation jsonl and compare with the 'length' field (can be slow).",
    )
    args = parser.parse_args()

    meta_path: Path = args.meta_path
    if not meta_path.exists():
        raise SystemExit(f"[ERROR] meta file not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict) or not meta:
        raise SystemExit("[ERROR] meta JSON must be a non-empty dict")

    internvl_chat_root = _infer_internvl_chat_root(meta_path)
    errors: list[str] = []
    for name, spec in meta.items():
        if not isinstance(spec, dict):
            errors.append(f"{name}: spec must be a dict")
            continue

        missing = REQUIRED_KEYS - set(spec.keys())
        if missing:
            errors.append(f"{name}: missing keys: {sorted(missing)}")
            continue

        root = _resolve_path(Path(spec["root"]), internvl_chat_root)
        ann = _resolve_path(Path(spec["annotation"]), internvl_chat_root)
        if not ann.name.endswith(".jsonl"):
            errors.append(f"{name}: annotation must end with .jsonl, got: {ann}")

        if not root.exists():
            errors.append(f"{name}: root not found: {root}")
        if not ann.exists():
            errors.append(f"{name}: annotation not found: {ann}")

        if args.check_length and ann.exists():
            try:
                n = _count_lines(ann)
                expected = int(spec["length"])
                if n != expected:
                    errors.append(f"{name}: length mismatch (meta={expected}, jsonl_lines={n})")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: failed to count lines: {e}")

    if errors:
        print("[FAIL] meta validation errors:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(2)

    print(f"[OK] meta file looks consistent: {meta_path}")
    print(f"Datasets: {len(meta)}")


if __name__ == "__main__":
    main()
