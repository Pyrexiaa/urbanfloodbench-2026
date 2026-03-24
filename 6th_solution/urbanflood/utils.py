# #####
# utils.py
# #####

# urbanflood/utils.py
from __future__ import annotations

import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    return x


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_config(config: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(config):
        data = asdict(config)
    elif isinstance(config, dict):
        data = config
    else:
        data = {"value": str(config)}
    lines = [f"{k}: {v}" for k, v in sorted(data.items(), key=lambda kv: kv[0])]
    p.write_text("\n".join(lines) + "\n")


def env_default_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)

