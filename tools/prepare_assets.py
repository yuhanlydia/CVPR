"""Acquire original public model/benchmark assets; no output upload to HF.

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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image-root")
    a = p.parse_args()
    prepare(a.config, a.out, a.image_root)
