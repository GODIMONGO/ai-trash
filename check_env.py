#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick environment check: verifies key packages and prints versions and basic system info.
Run:
  python check_env.py
"""
from __future__ import annotations
import os
import shutil
import sys

print("Python:", sys.version)

# Disk space
try:
    total, used, free = shutil.disk_usage(".")
    print(f"Disk: total={total//(1024**3)}GB used={used//(1024**3)}GB free={free//(1024**3)}GB")
except Exception as e:
    print("Disk check error:", e)

# Imports and versions
mods = {}
for name in ["yaml", "ultralytics", "torch", "cv2", "numpy", "PIL", "psutil"]:
    try:
        m = __import__(name if name != "PIL" else "PIL")
        ver = getattr(m, "__version__", getattr(getattr(m, "__about__", object), "__version__", "unknown"))
        mods[name] = ver
    except Exception as e:
        mods[name] = f"ERROR: {e}"

print("\nPackages:")
for k, v in mods.items():
    print(f"- {k}: {v}")

# Torch details
try:
    import torch
    print("\nTorch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
except Exception as e:
    print("Torch info error:", e)
