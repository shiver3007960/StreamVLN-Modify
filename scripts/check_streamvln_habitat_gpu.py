#!/usr/bin/env python3
"""Smoke-check Habitat-Sim GPU context and one R2R reset."""

from __future__ import annotations

import os

import habitat
from habitat import Env
from habitat_baselines.config.default import get_config as get_habitat_config


def main() -> int:
    config = get_habitat_config("config/vln_r2r.yaml")
    gpu_device_id = int(os.environ.get("STREAMVLN_HABITAT_GPU_DEVICE_ID", "0"))

    with habitat.config.read_write(config):
        config.habitat.dataset.split = "val_seen"
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_device_id

    env = Env(config=config)
    try:
        observations = env.reset()
        print(f"episodes={len(env.episodes)}")
        print(f"gpu_device_id={gpu_device_id}")
        print(f"rgb_shape={observations['rgb'].shape}")
        print(f"depth_shape={observations['depth'].shape}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
