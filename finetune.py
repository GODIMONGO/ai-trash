#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune (continue training) a YOLO model using parameters from a YAML config file.

Usage (PowerShell):
    python finetune.py --config configs/finetune.yaml

Notes:
- Edit configs/finetune.yaml to change all hyperparameters and paths.
- Ensure dependencies are installed: pip install -r requirements.txt
"""
from __future__ import annotations
import datetime
from pathlib import Path

try:
    import yaml  # PyYAML
except Exception as e:
    raise SystemExit("PyYAML is required. Install with 'pip install pyyaml' or 'pip install -r requirements.txt'") from e

try:
    from ultralytics import YOLO
except Exception as e:
    err = str(e)
    hint = (
        "Try: pip install --no-cache-dir ultralytics opencv-python-headless\n"
        "If you see 'libGL.so.1' errors on Linux, install system libs: sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0"
    )
    raise SystemExit(
        f"Failed to import Ultralytics: {err}\n{hint}"
    ) from e


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit("Invalid config format: expected a YAML mapping (dict)")
    return cfg


def main() -> None:
    # Zero-argument execution: use default config path
    cfg = load_config("configs/finetune.yaml")
    weights_path = cfg.pop("weights", None)
    if not weights_path:
        raise SystemExit("'weights' must be specified in the finetune config YAML")

    project = Path(cfg.get("project", "runs/finetune"))
    project.mkdir(parents=True, exist_ok=True)
    if cfg.get("name") in (None, "", "null"):
        cfg["name"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.setdefault("exist_ok", True)

    # Resolve device automatically if set to 'auto'
    try:
        import torch
        device = str(cfg.get("device", "auto"))
        if device == "auto":
            cfg["device"] = "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    model = YOLO(weights_path)
    results = model.train(**cfg)

    print("\nFine-tuning complete.")
    save_dir = getattr(results, "save_dir", None)
    print(f"Results saved to: {save_dir if save_dir else project / cfg['name']}")


if __name__ == "__main__":
    main()
