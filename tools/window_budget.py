"""Retain a finite window's original clock across repeated setup invocations."""
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from common import read_json


def remaining_seconds(receipt, hours=8, epoch=None):
    started = receipt["started_epoch"]
    if isinstance(started, bool) or not isinstance(started, (int, float)) or not math.isfinite(started):
        raise ValueError("Invalid retained window start")
    current = time.time() if epoch is None else epoch
    if started > current + 1:
        raise ValueError("Retained window start is in the future")
    if not 0 < hours <= 8:
        raise ValueError("This window is bounded by eight hours")
    return max(0.0, hours * 3600 - max(0.0, current - started))


def ensure_window(root, epoch=None):
    path = Path(root) / "setup-receipt.json"
    if not path.exists():
        receipt = {"window_id": uuid.uuid4().hex,
                   "started_epoch": time.time() if epoch is None else epoch, "status": "started"}
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            pass
    receipt = read_json(path)
    remaining_seconds(receipt, epoch=epoch)
    # Legacy timestamps keep their original identity rather than resetting the budget.
    identity = receipt.get("window_id") or hashlib.sha256(str(receipt["started_epoch"]).encode()).hexdigest()[:32]
    if len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Invalid retained window identity")
    return {**receipt, "window_id": identity}


def claim_attempt(root, window, run_id):
    marker = Path(root) / "runs/windows" / window["window_id"] / "attempt.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "window_id": window["window_id"],
                   "started_epoch": window["started_epoch"]}, stream)
        stream.flush()
        os.fsync(stream.fileno())
