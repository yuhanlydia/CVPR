#!/usr/bin/env python3
"""generated_unexecuted: immutable native MMEB asset acquisition for Local.

No model inference, fitting, scoring, SSH, driver, clock reset or retry loop.
Run each invocation as a foreground CPU job inside the existing run_harness.
The outer finite deadline must also bound an SDK/network call that blocks.
"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import sys
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

TASKS = ("ScienceQA", "ChartQA", "MSCOCO_i2t")
TRAIN_TASKS = ("ScienceQA", "A-OKVQA")
ASSET_IDS = ("model", "test", "eval-images", "train")


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_hash(path, deadline=None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            if deadline:
                deadline()
            value.update(chunk)
    return value.hexdigest()


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Invalid native relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or ":" in p.parts[0] or p.as_posix() != value:
        raise ValueError("Unsafe/noncanonical native relative path: " + value)
    return p


def under(root, value):
    path = Path(root) / relative(value)
    path.resolve().relative_to(Path(root).resolve())
    if path.is_symlink():
        raise ValueError("Asset path is a symlink: " + str(path))
    return path


def load_lock(path):
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if lock.get("schema") != "cvpr-native-asset-lock-v1":
        raise ValueError("Unsupported asset lock")
    entries = lock["assets"]
    if len(entries) != 4 or {a["asset_id"] for a in entries} != set(ASSET_IDS):
        raise ValueError("Asset lock must contain exactly four distinct native asset groups")
    for asset in entries:
        revision = asset["revision"]
        if len(revision) != 40 or any(x not in "0123456789abcdef" for x in revision):
            raise ValueError("An immutable full revision is required")
        if asset["repo_type"] not in ("model", "dataset") or asset.get("private") or asset.get("gated"):
            raise ValueError("This acquisition path is for the recorded public ungated inputs")
        names = set()
        for entry in asset["files"]:
            relative(entry["path"])
            if entry["path"] in names or type(entry["size"]) is not int or entry["size"] < 0:
                raise ValueError("Duplicate/invalid locked file")
            names.add(entry["path"])
            digest = entry["sha256"] or entry["git_blob_sha1"]
            length = 64 if entry["sha256"] else 40
            if len(digest) != length or any(x not in "0123456789abcdef" for x in digest):
                raise ValueError("Invalid locked file digest")
        if sum(e["size"] for e in asset["files"]) != asset["download_bytes"]:
            raise ValueError("Locked byte total differs from file inventory")
    return lock


def verify_file(path, entry, deadline):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != entry["size"]:
        raise ValueError("Missing/size-mismatched locked file: " + str(path))
    sha = hashlib.sha256()
    git = hashlib.sha1(("blob " + str(entry["size"]) + "\0").encode("ascii"))
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            deadline()
            sha.update(chunk)
            git.update(chunk)
    actual_sha = sha.hexdigest()
    if entry["sha256"]:
        good = actual_sha == entry["sha256"]
    else:
        good = git.hexdigest() == entry["git_blob_sha1"]
    if not good:
        raise ValueError("Locked checksum mismatch (file retained): " + str(path))
    return {"path": entry["path"], "bytes": entry["size"], "sha256": actual_sha,
            "identity_basis": "hub_lfs_sha256" if entry["sha256"] else "hub_git_blob_sha1",
            "git_blob_sha1": None if entry["sha256"] else git.hexdigest()}


def asset_dir(root, asset):
    return under(root, "hf/" + asset["asset_id"])


def verify_assets(lock, root, ids, deadline):
    """Reusable offline verification; raises on any changed or missing required file."""
    records = {}
    for asset in lock["assets"]:
        if asset["asset_id"] in ids:
            records[asset["asset_id"]] = [
                verify_file(under(asset_dir(root, asset), e["path"]), e, deadline)
                for e in asset["files"]]
    return records


def lists(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else value


def required_images(lock, root, deadline, splits=("test", "train")):
    # Read every physical released row. No performance/sample filtering.
    import pyarrow.parquet as pq
    wanted = {"test": set(), "train": set()}
    inventory = {}
    by_id = {a["asset_id"]: a for a in lock["assets"]}
    for split, tasks, fields in (
        ("test", TASKS, ("qry_img_path",)),
        ("train", TRAIN_TASKS, ("qry_image_path", "pos_image_path", "neg_image_path")),
    ):
        if split not in splits:
            continue
        base = asset_dir(root, by_id[split])
        for task in tasks:
            entries = [e for e in by_id[split]["files"]
                       if e["path"].startswith(task + "/") and e["path"].endswith(".parquet")]
            if not entries:
                raise ValueError("Missing locked native task shards: " + task)
            rows, task_images, candidate_counts = 0, set(), []
            for entry in entries:
                table = pq.ParquetFile(under(base, entry["path"]))
                required = ("qry_img_path", "tgt_text") if split == "test" else ("qry_image_path",)
                if any(key not in table.schema_arrow.names for key in required):
                    raise ValueError("Released image-field layout differs: " + task)
                for batch in table.iter_batches(batch_size=512):
                    deadline()
                    for row in batch.to_pylist():
                        rows += 1
                        for key in fields:
                            values = lists(row.get(key))
                            if not isinstance(values, list):
                                raise ValueError("Native image field must be string/list: " + key)
                            for value in values:
                                if value:
                                    task_images.add(relative(value).as_posix())
                        if split == "test":
                            text = lists(row.get("tgt_text"))
                            if not isinstance(text, list) or not text or not all(isinstance(t, str) for t in text):
                                raise ValueError("Original target text/image candidates are incomplete: " + task)
                            candidate_counts.append(len(text))
            if rows <= 0:
                raise ValueError("Empty released task: " + task)
            wanted[split].update(task_images)
            inventory[split + "/" + task] = {
                "released_rows": rows, "image_paths": len(task_images),
                "all_shards": [e["path"] for e in entries],
                "native_candidate_counts_sha256": hashlib.sha256(
                    json.dumps(candidate_counts, separators=(",", ":")).encode()).hexdigest()
                    if split == "test" else None,
                "coverage": "Every locked released row; no fitting/eval row sampling",
            }
    return wanted, inventory


def extract_images(archives, wanted, target, deadline, event):
    # Structural prefix/suffix mapping only. Never match by arbitrary basename.
    target.mkdir(parents=True, exist_ok=True)
    mapping, aliases = {}, {}
    for name in sorted(wanted):
        for alias in (name, name.removeprefix("images/")):
            if alias in aliases and aliases[alias] != name:
                raise ValueError("Ambiguous native metadata image alias")
            aliases[alias] = name
    with contextlib.ExitStack() as stack:
        for archive in archives:
            z = stack.enter_context(zipfile.ZipFile(archive))
            for entry in z.infolist():
                deadline()
                if entry.is_dir():
                    continue
                parts = relative(entry.filename).parts
                if stat.S_ISLNK((entry.external_attr >> 16) & 0xFFFF):
                    raise ValueError("Symlink archive member rejected")
                matches = {aliases[suffix] for i in range(len(parts))
                           if (suffix := "/".join(parts[i:])) in aliases}
                if len(matches) > 1:
                    raise ValueError("Ambiguous ZIP-to-native-path mapping")
                if matches:
                    name = next(iter(matches))
                    if name in mapping:
                        raise ValueError("Duplicate native path across original ZIP members: " + name)
                    mapping[name] = (z, entry, str(archive))
        missing = wanted - mapping.keys()
        if missing:
            raise ValueError("Original archives omit native paths: " + repr(sorted(missing)[:20]))
        needed = sum(entry.file_size for name, (_, entry, _) in mapping.items()
                     if not under(target, name).exists())
        if shutil.disk_usage(target).free < needed + 1024 ** 3:
            raise ValueError("Insufficient free disk for declared extraction plus 1 GiB margin")
        hashes = {}
        for name in sorted(mapping):
            deadline()
            z, entry, archive = mapping[name]
            dest = under(target, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            existing = dest.exists()
            temp = dest.with_name(dest.name + ".partial-" + uuid.uuid4().hex)
            h = hashlib.sha256()
            count = 0
            # Reading to EOF also invokes ZipFile's CRC check.
            with z.open(entry) as inp:
                output = contextlib.nullcontext(None) if existing else temp.open("xb")
                with output as stream:
                    while chunk := inp.read(4 * 1024 * 1024):
                        deadline()
                        h.update(chunk)
                        count += len(chunk)
                        if stream:
                            stream.write(chunk)
            if count != entry.file_size:
                raise ValueError("ZIP decompressed-size mismatch")
            value = h.hexdigest()
            if existing:
                if file_hash(dest, deadline) != value:
                    raise ValueError("Existing native image differs; retained: " + str(dest))
            else:
                # No overwrite if an unexpected external writer creates the destination.
                os.link(temp, dest)
                temp.unlink()
            hashes[name] = {"sha256": value, "bytes": count, "archive": archive,
                            "member": entry.filename, "crc32": entry.CRC, "reused": existing}
            event("image_verified", path=name, bytes=count, reused=existing)
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True, help="A new immutable per-invocation receipt directory")
    p.add_argument("--action", required=True, choices=("catalog", "download", "verify", "extract"))
    p.add_argument("--asset", action="append", choices=ASSET_IDS)
    p.add_argument("--split", action="append", choices=("test", "train"),
                   help="Extraction dependency scope; default both, independently processed")
    p.add_argument("--seconds", type=float, required=True, help="Within the existing remaining harness budget")
    a = p.parse_args()
    if not 0 < a.seconds < float("inf"):
        p.error("--seconds must be finite and positive")
    start = time.monotonic()
    def deadline():
        if time.monotonic() - start >= a.seconds:
            raise TimeoutError("Declared inner deadline exhausted; outer harness must bound blocking calls")
    out, root = Path(a.out).resolve(), Path(a.root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "cvpr-native-assets-local-receipt-v1", "started_at": stamp(),
               "action": a.action, "elapsed_seconds": 0, "status": "running",
               "role": "local_asset_maintenance", "scientific_verdict": "NONE",
               "software_acceptance": "not_asserted", "assets": {}, "failures": []}
    def save():
        temp = out / "receipt.json.tmp"
        temp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temp.replace(out / "receipt.json")
    def event(kind, **fields):
        record = {"at": stamp(), "elapsed_seconds": time.monotonic() - start,
                  "kind": kind, **fields}
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        receipt["elapsed_seconds"] = record["elapsed_seconds"]
        save()
    try:
        lock = load_lock(a.lock)
        receipt["lock_sha256"] = file_hash(a.lock, deadline)
        receipt["code_sha256"] = file_hash(__file__, deadline)
        receipt["root"] = str(root)
        ids = set(a.asset or ASSET_IDS)
        if a.action == "extract" and a.asset:
            raise ValueError("Use --split for extraction; each split verifies its exact dependencies")
        if a.split and a.action != "extract":
            raise ValueError("--split applies only to extraction")
        receipt["selected_assets"] = sorted(ids)
        if a.action == "catalog":
            receipt["catalog"] = [{"asset_id": x["asset_id"], "revision": x["revision"],
                                   "files": len(x["files"]), "download_bytes": x["download_bytes"]}
                                  for x in lock["assets"] if x["asset_id"] in ids]
            receipt["status"] = "catalog_only"
            event("catalog", downloaded=False, total_raw_bytes=sum(
                x["download_bytes"] for x in lock["assets"] if x["asset_id"] in ids),
                  uncompressed_size="Resolve from verified original ZIP central directory in Local")
            return 0
        root.mkdir(parents=True, exist_ok=True)
        with under(root, "acquisition.lock").open("a+") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.action == "download":
                if importlib.metadata.version("huggingface-hub") != lock["sdk_version"]:
                    raise ValueError("Acquisition SDK version differs from the source-reviewed lock")
                from huggingface_hub import hf_hub_download
                for asset in lock["assets"]:
                    if asset["asset_id"] not in ids:
                        continue
                    folder = asset_dir(root, asset)
                    folder.mkdir(parents=True, exist_ok=True)
                    records, failures = [], []
                    for entry in asset["files"]:
                        try:
                            deadline()
                            path = under(folder, entry["path"])
                            if path.exists():
                                record = verify_file(path, entry, deadline)
                                event("locked_file_reused", asset=asset["asset_id"], path=entry["path"])
                            else:
                                if shutil.disk_usage(root).free < entry["size"] + 1024 ** 3:
                                    raise ValueError("Insufficient disk for next file plus 1 GiB margin")
                                event("download_started", asset=asset["asset_id"], path=entry["path"],
                                      revision=asset["revision"])
                                hf_hub_download(repo_id=asset["repo_id"], filename=entry["path"],
                                                repo_type=asset["repo_type"], revision=asset["revision"],
                                                local_dir=str(folder))
                                deadline()
                                record = verify_file(path, entry, deadline)
                                event("download_verified", asset=asset["asset_id"], **record)
                            records.append(record)
                        except Exception as error:
                            failure = {"asset": asset["asset_id"], "path": entry["path"],
                                       "error_type": type(error).__name__, "message": str(error)}
                            failures.append(failure)
                            receipt["failures"].append(failure)
                            event("file_failed", **failure)
                            if isinstance(error, TimeoutError):
                                raise
                            # No project retry; continue independent files/groups within the same budget.
                    receipt["assets"][asset["asset_id"]] = {
                        "revision": asset["revision"], "files": records,
                        "status": "failed" if failures else "verified"}
                    save()
            elif a.action == "verify":
                for asset_id in sorted(ids):
                    try:
                        records = verify_assets(lock, root, {asset_id}, deadline)[asset_id]
                        receipt["assets"][asset_id] = {"status": "verified", "files": records}
                        event("asset_verified", asset=asset_id, files=len(records))
                    except Exception as error:
                        receipt["failures"].append({"asset": asset_id,
                            "error_type": type(error).__name__, "message": str(error)})
                        event("asset_failed", asset=asset_id, error_type=type(error).__name__,
                              message=str(error))
                        if isinstance(error, TimeoutError):
                            raise
            else:
                by_id = {x["asset_id"]: x for x in lock["assets"]}
                receipt["native_inventory"], receipt["images"], receipt["image_roots"] = {}, {}, {}
                for split in sorted(set(a.split or ("test", "train"))):
                    try:
                        deps = {"test", "eval-images"} if split == "test" else {"train"}
                        receipt["assets"].update(verify_assets(lock, root, deps, deadline))
                        event("extraction_inputs_verified", split=split)
                        wanted, inventory = required_images(lock, root, deadline, (split,))
                        asset_id = "eval-images" if split == "test" else "train"
                        base = asset_dir(root, by_id[asset_id])
                        archives = [under(base, x["path"]) for x in by_id[asset_id]["files"]
                                    if x["path"].endswith(".zip")]
                        receipt["images"][split] = extract_images(
                            archives, wanted[split], under(root, split + "-images"), deadline, event)
                        receipt["native_inventory"].update(inventory)
                        receipt["image_roots"][split] = str(under(root, split + "-images"))
                        event("split_extraction_completed", split=split, inventory=inventory)
                    except Exception as error:
                        receipt["failures"].append({"split": split,
                            "error_type": type(error).__name__, "message": str(error)})
                        event("split_extraction_failed", split=split,
                              error_type=type(error).__name__, message=str(error))
                        if isinstance(error, TimeoutError):
                            raise
                receipt["native_semantics"] = "Original labels/candidates untouched; no score reported"
        receipt["status"] = "failed" if receipt["failures"] else "completed"
        return 2 if receipt["failures"] else 0
    except Exception as error:
        receipt["failures"].append({"error_type": type(error).__name__, "message": str(error)})
        receipt["status"] = "failed"
        event("invocation_failed", error_type=type(error).__name__, message=str(error))
        return 2
    finally:
        receipt["finished_at"] = stamp()
        receipt["elapsed_seconds"] = time.monotonic() - start
        save()


if __name__ == "__main__":
    sys.exit(main())
