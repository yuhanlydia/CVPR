"""Verify consumed checkpoint bytes against the pinned public HF revision."""
import hashlib
from pathlib import Path

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
MODEL_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"


def content_hash(path, algorithm="sha256", git_blob=False):
    path = Path(path)
    hasher = hashlib.new(algorithm)
    if git_blob:
        hasher.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoint(root):
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION,
                             files_metadata=True, timeout=30)
    if info.sha != MODEL_REVISION:
        raise ValueError("Public model revision did not resolve to the pinned commit")
    files = {entry.rfilename: entry for entry in info.siblings}
    required = ["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth",
                "config.json", "diffusion_pytorch_model.safetensors"]
    tokenizer = sorted(name for name in files if name.startswith("google/umt5-xxl/"))
    if not tokenizer:
        raise ValueError("Pinned model has no expected UMT5 tokenizer files")
    receipts = []
    for name in required + tokenizer:
        item = files.get(name)
        path = Path(root) / name
        if item is None or not path.is_file():
            raise FileNotFoundError(name)
        if item.lfs is not None:
            expected = item.lfs.sha256
            algorithm, git_blob = "sha256", False
        else:
            expected = item.blob_id
            algorithm, git_blob = "sha1", True
        observed = content_hash(path, algorithm, git_blob)
        if not expected or observed != expected:
            raise ValueError(f"Checkpoint content differs from pinned source: {name}")
        receipts.append({"path": name, "bytes": path.stat().st_size,
                         "algorithm": "git-blob-sha1" if git_blob else algorithm,
                         "digest": observed})
    return {"model_id": MODEL_ID, "verified_revision": info.sha,
            "files": receipts, "ok": True}
