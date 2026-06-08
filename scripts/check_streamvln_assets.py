#!/usr/bin/env python3
import gzip
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ok(label, detail=""):
    print(f"[OK] {label}{': ' + detail if detail else ''}")


def miss(label, detail=""):
    print(f"[MISSING] {label}{': ' + detail if detail else ''}")


def warn(label, detail=""):
    print(f"[WARN] {label}{': ' + detail if detail else ''}")


def check_path(label, rel, kind="any"):
    p = ROOT / rel
    exists = p.exists()
    if kind == "dir":
        exists = p.is_dir()
    elif kind == "file":
        exists = p.is_file()
    (ok if exists else miss)(label, str(p))
    return p if exists else None


def count_dirs(path):
    try:
        return sum(1 for p in path.iterdir() if p.is_dir())
    except Exception:
        return 0


def check_gz_episodes(label, rel):
    p = check_path(label, rel, "file")
    if not p:
        return
    try:
        with gzip.open(p, "rt") as f:
            data = json.load(f)
        episodes = data.get("episodes", data if isinstance(data, list) else [])
        ok(label + " episodes", str(len(episodes)))
    except Exception as exc:
        warn(label + " read", repr(exc))


def check_annotations(label, rel, require_images=True):
    base = ROOT / rel
    ann = base / "annotations.json"
    images = base / "images"
    if not ann.is_file():
        miss(label + " annotations", str(ann))
        return
    try:
        data = json.load(open(ann))
        first = data[0] if isinstance(data, list) and data else None
        detail = f"{len(data)} samples"
        if first:
            detail += f", first={first.get('video')}"
        ok(label + " annotations", detail)
        if require_images:
            if not images.is_dir():
                miss(label + " images", str(images))
            elif first and first.get("video"):
                sample = base / first["video"]
                (ok if sample.is_dir() else miss)(label + " sample video", str(sample))
    except Exception as exc:
        warn(label + " annotations read", repr(exc))


def check_checkpoint(rel):
    base = check_path("checkpoint", rel, "dir")
    if not base:
        return
    idx = base / "model.safetensors.index.json"
    if not idx.is_file():
        miss("checkpoint index", str(idx))
        return
    data = json.load(open(idx))
    shards = sorted(set(data.get("weight_map", {}).values()))
    missing = [s for s in shards if not (base / s).is_file()]
    if missing:
        miss("checkpoint shards", ", ".join(missing))
    else:
        ok("checkpoint shards", f"{len(shards)} shards")
    cfg = base / "config.json"
    if cfg.is_file():
        c = json.load(open(cfg))
        ok("checkpoint config", f"{c.get('architectures')} vision={c.get('mm_vision_tower')}")


def main():
    mp3d = check_path("MP3D scenes", "data/scene_datasets/mp3d", "dir")
    if mp3d:
        n = count_dirs(mp3d)
        (ok if n >= 90 else warn)("MP3D scene count", str(n))
    hm3d = check_path("HM3D scenes", "data/scene_datasets/hm3d", "dir")
    if hm3d:
        n = count_dirs(hm3d)
        (ok if n >= 800 else warn)("HM3D scene count", str(n))

    check_gz_episodes("R2R train", "data/datasets/r2r/train/train.json.gz")
    check_gz_episodes("R2R val_seen", "data/datasets/r2r/val_seen/val_seen.json.gz")
    check_gz_episodes("R2R val_unseen", "data/datasets/r2r/val_unseen/val_unseen.json.gz")
    check_gz_episodes("RxR train guide", "data/datasets/rxr/train/train_guide_en.json.gz")
    check_gz_episodes("RxR val_seen guide", "data/datasets/rxr/val_seen/val_seen_guide.json.gz")
    check_gz_episodes("RxR val_unseen guide", "data/datasets/rxr/val_unseen/val_unseen_guide.json.gz")
    check_gz_episodes("EnvDrop episodes", "data/datasets/envdrop/envdrop.json.gz")
    check_gz_episodes("ScaleVLN subset", "data/datasets/scalevln/scalevln_subset_150k.json.gz")

    for name in ["R2R", "RxR", "EnvDrop", "ScaleVLN"]:
        check_annotations(f"trajectory {name}", f"data/trajectory_data/{name}")
    for name in ["R2R", "RxR", "EnvDrop"]:
        check_annotations(f"dagger {name}", f"data/dagger_data/{name}")

    check_checkpoint("checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3")


if __name__ == "__main__":
    main()
