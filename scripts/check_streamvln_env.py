#!/usr/bin/env python3
"""Smoke-check key Python imports for the StreamVLN environment."""

from __future__ import annotations

import importlib


MODULES = [
    "torch",
    "torchvision",
    "torch_scatter",
    "transformers",
    "accelerate",
    "deepspeed",
    "habitat",
    "habitat_sim",
    "quaternion",
    "depth_camera_filtering",
    "safetensors",
    "peft",
    "decord",
    "cv2",
    "PIL",
    "omegaconf",
    "clip",
    "open_clip",
    "bitsandbytes",
]


def main() -> int:
    missing = 0
    for name in MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - smoke check should report all failures.
            missing += 1
            print(f"MISS {name}: {type(exc).__name__}: {exc}")
            continue

        version = getattr(module, "__version__", "ok")
        print(f"OK {name}: {version}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
