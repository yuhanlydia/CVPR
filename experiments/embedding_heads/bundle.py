"""Identity checks; training and evaluation have separate loading interfaces."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .heads import Training, Blocked, unit

SCORER_PATH = "src/evaluation/mmeb_v2/utils/eval_utils/metrics.py"
SCORER_BLOB = "097365b58831711d0b04eef08f9d4232f6486d6c"
MODEL_PATH = "src/evaluation/mmeb_v2/models.py"
MODEL_BLOB = "f00799a084e8772c4617d9efcbe62efd7a0d2643"
UPSTREAM = "393e2978d27852b0d0230d6994f37f9c15bed73c"

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]

def referenced(base, ref):
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Bundle references must be relative and contained")
    path = (Path(base)/relative).resolve()
    path.relative_to(Path(base).resolve())
    if not path.is_file() or sha256(path) != ref["sha256"]:
        raise ValueError(f"Missing/changed actual cache: {relative}")
    return path

def all_refs(manifest):
    refs = [manifest["projection_ref"], manifest["train"]["features_ref"],
            manifest["train"]["rows_ref"], manifest["leakage_audit_ref"]]
    for task in manifest["evaluation"]:
        refs += [task[k] for k in ("features_ref", "metadata_ref", "original_predictions_ref", "original_scores_ref")]
    refs += manifest.get("provenance_refs", [])
    if manifest.get("interval_calibration"):
        refs += [manifest["interval_calibration"]["radius_ref"], manifest["interval_calibration"]["evidence_ref"]]
    return refs

def validate_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["schema"] != "r003.frozen-embedding-bundle.v1" or value["upstream_commit"] != UPSTREAM:
        raise ValueError("Unqualified cache schema or upstream identity")
    if value["model_id"] != "Qwen/Qwen3-VL-Embedding-2B":
        raise ValueError("This frozen batch supports only the declared 2B carrier")
    train = value["train"]
    if train["dataset_id"] != "TIGER-Lab/MMEB-train" or train["split"] != "original":
        raise ValueError("Only declared original public training data may fit these heads")
    if len(train["revision"]) != 40 or value["teacher"]["kind"] != "frozen_full_dimension_same_2B":
        raise ValueError("Unpinned training or undeclared teacher")
    for ref in all_refs(value):
        referenced(path.parent, ref)
    if not value["evaluation"] or len({t["task"] for t in value["evaluation"]}) != len(value["evaluation"]):
        raise ValueError("Missing/duplicate original evaluation tasks")
    leakage = json.loads(referenced(path.parent, value["leakage_audit_ref"]).read_text())
    if (set(leakage["train_image_sha256"]) & set(leakage["eval_image_sha256"])
            or set(leakage["train_query_ids"]) & set(leakage["eval_query_ids"])):
        raise ValueError("Training/evaluation input overlap")
    return value

def load_training(path, manifest, temperature):
    if manifest["teacher"]["temperature"] != temperature:
        raise ValueError("Teacher/student logit units changed")
    base = Path(path).parent
    rows = jsonl(referenced(base, manifest["train"]["rows_ref"]))
    with np.load(referenced(base, manifest["train"]["features_ref"]), allow_pickle=False) as data:
        values = {name: data[name].copy() for name in ("q", "c", "positive", "allowed", "teacher")}
    if len(rows) != len(values["q"]) or any(row["split"] != "original" for row in rows):
        raise ValueError("Training row identity/role mismatch")
    radius = None
    spec = manifest.get("interval_calibration")
    if spec:
        evidence = json.loads(referenced(base, spec["evidence_ref"]).read_text())
        leakage = json.loads(referenced(base, manifest["leakage_audit_ref"]).read_text())
        excluded = set(row["query_id"] for row in rows) | set(leakage["eval_query_ids"])
        if (evidence["units"] != "teacher_logit" or evidence["model_revision"] != manifest["model_revision"]
                or not evidence["query_ids"] or excluded & set(evidence["query_ids"])
                or evidence["split"] in {"test", "confirmation"}):
            raise Blocked("INTERVAL_CALIBRATION_ROLE_OR_IDENTITY_INVALID")
        with np.load(referenced(base, spec["radius_ref"]), allow_pickle=False) as data:
            radius = data["radius"].copy()
    return Training(**values, groups=np.array([row["group"] for row in rows]), radius=radius).validate()

def load_projection(path, manifest):
    with np.load(referenced(Path(path).parent, manifest["projection_ref"]), allow_pickle=False) as data:
        return data["mean"].copy(), data["basis"].copy()

def project(x, projection):
    mean, basis = projection
    return unit((np.asarray(x, dtype=np.float64)-mean)@basis)

def check_original_source(root):
    for relative, expected in ((SCORER_PATH, SCORER_BLOB), (MODEL_PATH, MODEL_BLOB)):
        data = (Path(root)/relative).read_bytes()
        actual = hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        if actual != expected:
            raise ValueError(f"Official source changed: {relative}")
