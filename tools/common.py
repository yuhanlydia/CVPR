"""Small provenance helpers. No scientific metric is implemented here."""
import datetime as dt
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def run_plan_compat(native, root, plan, *, authorizer, process_fds=(), lease_factory=None):
    """Call installed Research Autopilot runners across small API revisions."""
    kwargs = {"authorizer": authorizer}
    parameters = inspect.signature(native.run_plan).parameters
    if "process_fds" in parameters:
        kwargs["process_fds"] = tuple(process_fds)
    if lease_factory is not None and "lease_factory" in parameters:
        kwargs["lease_factory"] = lease_factory
    return native.run_plan(root, plan, **kwargs)
