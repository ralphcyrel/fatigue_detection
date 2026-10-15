"""
Per-driver neutral head pose, stored on this unit (2026-10-15).

Head-pose alerts are judged as a deviation from the driver's own neutral pose
(``modules.head_pose``). The neutral pose is the median pitch / yaw / roll over
enrollment's calibration window. It depends on how the camera is mounted, so
it belongs to this unit rather than to the driver's backend record, and the
backend API has no field for it: it is kept in ``config.POSE_NEUTRAL_FILE``
as ``{"<driver_id>": {"pitch", "yaw", "roll", "frames", "captured_at",
"device_id", "calibration"}}``.

A driver with no entry falls back to the median of the pre-drive settling
period (or, in monitoring, of the first seconds of face time); see
``modules.pipeline.MetricsPipeline.set_pose_neutral``.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)

# Fewer pose estimates than this cannot give a meaningful median.
MIN_NEUTRAL_FRAMES: int = 5


def median_pose(poses: Iterable[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """
    Median pitch / yaw / roll of ``poses`` (dicts from ``HeadPoseEstimator``).

    Returns:
        ``{"pitch", "yaw", "roll", "frames"}``, or ``None`` with fewer than
        :data:`MIN_NEUTRAL_FRAMES` poses.
    """
    rows = [(float(p["pitch"]), float(p["yaw"]), float(p["roll"])) for p in poses
            if p is not None]
    if len(rows) < MIN_NEUTRAL_FRAMES:
        return None
    arr = np.array(rows)
    pitch, yaw, roll = (float(v) for v in np.median(arr, axis=0))
    return {"pitch": pitch, "yaw": yaw, "roll": roll, "frames": len(rows)}


def _path(path: Optional[Path]) -> Path:
    # Read at call time, so tests (and a re-pointed config) take effect.
    return Path(path) if path is not None else Path(config.POSE_NEUTRAL_FILE)


def _read_all(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.error("Neutral-pose store %s unreadable (%s) - treated as empty", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def load_neutral(driver_id: Optional[int],
                 path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    The stored neutral pose for ``driver_id``, or ``None`` if there is none
    (or the entry is malformed).
    """
    if driver_id is None:
        return None
    entry = _read_all(_path(path)).get(str(driver_id))
    if not isinstance(entry, dict):
        return None
    try:
        for key in ("pitch", "yaw", "roll"):
            value = float(entry[key])
            if not np.isfinite(value):
                raise ValueError(key)
            entry[key] = value
    except (KeyError, TypeError, ValueError):
        logger.warning("Neutral pose for driver %s is malformed - ignored: %r", driver_id, entry)
        return None
    return entry


def save_neutral(driver_id: int, neutral: Dict[str, Any], calibration: str = "",
                 path: Optional[Path] = None) -> bool:
    """
    Store ``neutral`` (``median_pose`` output) for ``driver_id``, replacing any
    earlier entry. Written atomically. Returns ``False`` on an I/O error.
    """
    target = _path(path)
    data = _read_all(target)
    data[str(driver_id)] = {
        "pitch": round(float(neutral["pitch"]), 3),
        "yaw": round(float(neutral["yaw"]), 3),
        "roll": round(float(neutral["roll"]), 3),
        "frames": int(neutral.get("frames", 0)),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "device_id": config.DEVICE_ID,
        "calibration": calibration,
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        logger.error("Could not store the neutral pose for driver %s in %s: %s",
                     driver_id, target, exc)
        return False
    logger.info("Neutral pose for driver %s stored in %s: pitch %.1f yaw %.1f roll %.1f "
                "(%d frames)", driver_id, target, neutral["pitch"], neutral["yaw"],
                neutral["roll"], int(neutral.get("frames", 0)))
    return True
