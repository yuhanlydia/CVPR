"""Acquire original public model/benchmark assets; no output upload to HF.

New immutable-local branch: generated_unexecuted; Local acceptance pending.

Revisions are resolved once per attempt, pinned in every subsequent dataset load,
and returned in the receipt. Only referenced original image files are extracted.
"""
import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from common import digest, now, read_json, write_json


def referenced_images(dataset):
    result = set()
    for row in dataset:
        for key in ("qry_img_path", "tgt_img_path"):
            values = row.get(key) or []
            if isinstance(values, str):
                values = [values]
            for value in values:
                if value:
                    p = PurePosixPath(value)
                    if p.is_absolute() or ".." in p.parts:
                        raise ValueError("Unsafe native image path")
                    result.add(p.as_posix())
    return result


def extract_referenced(archive, wanted, target):
    target = Path(target)
    missing = set(wanted)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            parts = PurePosixPath(item.filename).parts
            if ".." in parts or PurePosixPath(item.filename).is_absolute():
                raise ValueError("Unsafe archive path")
            matches = ["/".join(parts[i:]) for i in range(len(parts))
                       if "/".join(parts[i:]) in wanted]
            if len(matches) > 1:
                raise ValueError("Ambiguous native image mapping")
            if not matches:
                continue
            relative = matches[0]
            if relative not in missing:
                raise ValueError("Duplicate original image in archive")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
            missing.remove(relative)
    if missing:
        raise ValueError(f"Archive does not cover {len(missing)} required native images")


def prepare(config_path, out, image_root=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    cfg, out = read_json(config_path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revisions = {
        "model": api.model_info(cfg["model_id"]).sha,
        "metadata": api.dataset_info(cfg["metadata_id"]).sha,
        "images": api.dataset_info(cfg["image_repository"]).sha,
    }
    write_json(out / "resolved-revisions.json", {"resolved_at": now(), **revisions})
    model_path = snapshot_download(cfg["model_id"], revision=revisions["model"],
                                  ignore_patterns=["*.md", "*.msgpack", "*.h5", "*.onnx"])
    wanted, fingerprints, native_sizes = set(), {}, {}
    for name in cfg["core_tasks"] + cfg["reserve_tasks"]:
        ds = load_dataset(cfg["metadata_id"], name, split="test", revision=revisions["metadata"])
        wanted.update(referenced_images(ds))
        fingerprints[name] = ds._fingerprint
        native_sizes[name] = len(ds)
    archive_sha = None
    if image_root is None:
        archive = hf_hub_download(cfg["image_repository"], cfg["image_archive"],
                                  repo_type="dataset", revision=revisions["images"])
        archive_sha = digest(archive)
        image_root = out / "image-tasks"
        extract_referenced(archive, wanted, image_root)
    image_root = Path(image_root).resolve()
    images = {}
    for relative in sorted(wanted):
        image = image_root / relative
        if not image.is_file():
            raise FileNotFoundError(f"Required original image missing: {relative}")
        images[relative] = digest(image)
    weights = {p.name: digest(p) for p in Path(model_path).glob("*.safetensors")}
    if not weights:
        raise ValueError("No safetensors model weights acquired")
    manifest = {"acquired_at": now(), "model_id": cfg["model_id"],
                "model_path": str(Path(model_path).resolve()), "model_revision": revisions["model"],
                "model_weight_hashes": weights, "metadata_id": cfg["metadata_id"],
                "metadata_revision": revisions["metadata"], "dataset_fingerprints": fingerprints,
                "native_task_sizes": native_sizes, "image_repository": cfg["image_repository"],
                "image_revision": revisions["images"], "archive_sha256": archive_sha,
                "image_source": "declared_local_files" if archive_sha is None else "official_archive",
                "image_root": str(image_root), "image_hashes": images}
    write_json(out / "assets.json", manifest)
    return manifest



def prepare_locked(config_path, out, asset_lock, asset_root, asset_receipt, seconds):
    """generated_unexecuted: offline bridge from immutable files to native qualification."""
    import time
    from datasets import load_dataset
    from acquire_native_assets import load_lock, verify_assets, asset_dir, under
    cfg, out = read_json(config_path), Path(out)
    started = time.monotonic()
    if seconds is None or not 0 < seconds < float("inf"):
        raise ValueError("Locked preparation requires a finite --seconds budget")
    def deadline():
        if time.monotonic() - started >= seconds:
            raise TimeoutError("Locked asset verification deadline exhausted")
    lock, root = load_lock(asset_lock), Path(asset_root).resolve()
    by_id = {x["asset_id"]: x for x in lock["assets"]}
    expected_ids = {"model": cfg["model_id"], "test": cfg["metadata_id"],
                    "eval-images": cfg["image_repository"]}
    if lock["source_version"] != cfg["upstream_commit"]:
        raise ValueError("Native source version differs from locked acquisition scope")
    if any(by_id[k]["repo_id"] != v for k, v in expected_ids.items()):
        raise ValueError("Config repository differs from the immutable asset lock")
    tasks = cfg["core_tasks"] + cfg["reserve_tasks"]
    if len(tasks) != 3 or set(tasks) != {"ScienceQA", "ChartQA", "MSCOCO_i2t"}:
        raise ValueError("Locked bridge supports exactly the three retained full native tasks")
    receipt = read_json(asset_receipt)
    if (receipt.get("schema") != "cvpr-native-assets-local-receipt-v1"
            or receipt.get("action") != "extract" or receipt.get("status") != "completed"
            or receipt.get("lock_sha256") != digest(asset_lock)
            or receipt.get("code_sha256") != digest(Path(__file__).with_name("acquire_native_assets.py"))):
        raise ValueError("Matching completed Local extraction receipt is required")
    expected_root = under(root, "test-images").resolve()
    if Path(receipt["image_roots"]["test"]).resolve() != expected_root:
        raise ValueError("Extraction path changed; retain the old receipt and rebind actual Local paths")
    verified = verify_assets(lock, root, {"model", "test", "eval-images"}, deadline)
    metadata_root, model_path = asset_dir(root, by_id["test"]), asset_dir(root, by_id["model"])
    wanted, fingerprints, native_sizes, task_files = set(), {}, {}, {}
    for name in tasks:
        entries = [e for e in by_id["test"]["files"]
                   if e["path"].startswith(name + "/") and e["path"].endswith(".parquet")]
        task_files[name] = [entry["path"] for entry in entries]
        ds = load_dataset("parquet", data_files={"test": [
            str(under(metadata_root, entry["path"])) for entry in entries]}, split="test")
        wanted.update(referenced_images(ds))
        fingerprints[name], native_sizes[name] = ds._fingerprint, len(ds)
        if native_sizes[name] != receipt["native_inventory"]["test/" + name]["released_rows"]:
            raise ValueError("Released native denominator differs from extraction receipt")
    images = {}
    if wanted != set(receipt["images"]["test"]):
        raise ValueError("Original native image coverage changed")
    for relative in sorted(wanted):
        deadline()
        image = under(expected_root, relative)
        actual = digest(image)
        if actual != receipt["images"]["test"][relative]["sha256"]:
            raise ValueError("Original extracted image changed: " + relative)
        images[relative] = actual
    weights = {e["path"]: e["sha256"] for e in by_id["model"]["files"]
               if e["path"].endswith(".safetensors")}
    if not weights or any(value is None for value in weights.values()):
        raise ValueError("Complete SHA-bound safetensors inventory is required")
    manifest = {
        "acquired_at": now(), "acquisition_mode": "immutable_local_verified",
        "model_id": cfg["model_id"], "model_path": str(model_path.resolve()),
        "model_revision": by_id["model"]["revision"], "model_weight_hashes": weights,
        "model_file_inventory": verified["model"],
        "metadata_id": cfg["metadata_id"], "metadata_path": str(metadata_root.resolve()),
        "metadata_revision": by_id["test"]["revision"], "metadata_task_files": task_files,
        "dataset_fingerprints": fingerprints, "native_task_sizes": native_sizes,
        "image_repository": cfg["image_repository"],
        "image_revision": by_id["eval-images"]["revision"],
        "archive_sha256": by_id["eval-images"]["files"][0]["sha256"],
        "image_source": "locked_official_archive_and_local_extraction",
        "image_root": str(expected_root), "image_hashes": images,
        "asset_lock_ref": {"path": str(Path(asset_lock).resolve()), "sha256": digest(asset_lock)},
        "extraction_receipt_ref": {"path": str(Path(asset_receipt).resolve()),
                                   "sha256": digest(asset_receipt)},
        "scientific_verdict": "NONE",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "resolved-revisions.json", {
        "resolved_at": now(), "resolution": "immutable_asset_lock_no_head_lookup",
        "model": by_id["model"]["revision"], "metadata": by_id["test"]["revision"],
        "images": by_id["eval-images"]["revision"]})
    write_json(out / "assets.json", manifest)
    return manifest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image-root")
    p.add_argument("--asset-lock")
    p.add_argument("--asset-root")
    p.add_argument("--asset-receipt")
    p.add_argument("--seconds", type=float)
    a = p.parse_args()
    locked = (a.asset_lock, a.asset_root, a.asset_receipt)
    if any(locked):
        if not all(locked) or a.image_root:
            p.error("Use all three --asset-* arguments without --image-root")
        prepare_locked(a.config, a.out, *locked, a.seconds)
    else:
        prepare(a.config, a.out, a.image_root)
