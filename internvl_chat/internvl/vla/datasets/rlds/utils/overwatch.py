"""
overwatch.py

Minimal replacement for OpenVLA's `prismatic.overwatch.initialize_overwatch`.

The RLDS pipeline only needs a logger-like object with `.info/.warning/.error` methods.
"""

from __future__ import annotations

import logging


def initialize_overwatch(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Keep setup lightweight and avoid configuring root logging multiple times.
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger

