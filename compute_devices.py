from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def enumerate_inference_devices() -> list[tuple[str, str]]:
    """Return PyTorch-visible inference devices as (value, label) pairs."""
    devices = [("cpu", "CPU")]
    try:
        if not torch.cuda.is_available():
            return devices
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            devices.append((f"cuda:{index}", f"GPU {index}: {name}"))
    except Exception:
        logger.exception("Unable to enumerate CUDA inference devices")
    return devices
