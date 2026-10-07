"""Versioned inventory binding; original 43-arm config remains unchanged."""
import hashlib
import json
from pathlib import Path
from .audit_controls import registry

TASKS = ["ScienceQA", "ChartQA", "MSCOCO_i2t"]


def load_inventory(config_path, extension_path=None):
    raw = Path(config_path).read_bytes()
    config = json.loads(raw)
    arms = registry(config)
    if extension_path is None:
        return config, arms
    extension = json.loads(Path(extension_path).read_bytes())
    git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if (extension["schema"] != "cvpr.existing-method-extensions.v1"
            or extension["version"] != 1 or extension["status"] != "generated_unexecuted"
            or extension["native_tasks"] != TASKS
            or extension["parent_config_git_blob"] != git_blob
            or extension["dispatch_ready"] is not False):
        raise ValueError("Extension/base source identity or three-task design changed")
    additional = extension["arms"]
    ids = [a["id"] for a in additional]
    if len(ids) != 29 or len(set(ids)) != 29 or set(ids) & set(arms):
        raise ValueError("Reviewed 29-arm extension inventory changed")
    grouped = [a["id"] for g in extension["groups"] for a in g["arms"]]
    if sorted(ids) != sorted(grouped):
        raise ValueError("Extension group/arm inventory mismatch")
    kinds = {"query_balanced_all_pairs", "teacher_hard_negative", "uniform_native_negative",
             "teacher_label_mixture", "source_restricted_head", "nested_head_subset",
             "balanced_query_candidate_second_moment", "positive_weighted_second_moment",
             "source_svd_protection", "random_orthogonal_protection"}
    for a in additional:
        if not a["id"].startswith("B_") or a["kind"] not in kinds:
            raise ValueError("Unknown extension construction")
        if a.get("parent") not in {None, "I09", "I13", "A_CE_LBFGS"}:
            raise ValueError("Unreviewed subset parent")
        if a["kind"] == "teacher_label_mixture" and a["alpha"] not in {.25, .5, .75}:
            raise ValueError("Unreviewed teacher mixture")
        if a["kind"] == "nested_head_subset" and a["unique_queries_per_group"] not in {64, 128, 256}:
            raise ValueError("Unreviewed head supervision count")
        if a["kind"] == "source_restricted_head" and a["group"] not in {"ScienceQA", "A-OKVQA"}:
            raise ValueError("Unreviewed training source")
        if a["kind"] in {"uniform_native_negative", "nested_head_subset", "random_orthogonal_protection"} and a["seed"] not in {42, 104729, 314159}:
            raise ValueError("Unreviewed sampling seed")
        if a["kind"] == "source_svd_protection" and a["protected_rank"] not in {2, 4, 16}:
            raise ValueError("Unreviewed protected rank")
        if a["kind"] == "random_orthogonal_protection" and (a["protected_rank"] != 8 or a["seed"] != 42):
            raise ValueError("Unreviewed capacity control")
        if a.get("pair_chunk_size", 256) != 256:
            raise ValueError("Unreviewed streaming objective chunk")
    merged = dict(config)
    merged["arms"] = config["arms"] + additional
    merged["primary_contrasts"] = config["primary_contrasts"] + extension["primary_contrasts"]
    merged["secondary_contrasts"] = config["secondary_contrasts"] + extension["secondary_contrasts"]
    merged["_extension"] = extension
    all_ids = set(arms) | set(ids)
    contrast_ids = []
    for c in merged["primary_contrasts"] + merged["secondary_contrasts"]:
        if not set(c["weights"]) <= all_ids or abs(sum(c["weights"].values())) > 1e-12:
            raise ValueError("Contrast requires unregistered arms or nonzero total weight")
        contrast_ids.append(c["id"])
    if len(contrast_ids) != len(set(contrast_ids)):
        raise ValueError("Duplicate contrast identity")
    return merged, {a["id"]: a for a in merged["arms"]}
