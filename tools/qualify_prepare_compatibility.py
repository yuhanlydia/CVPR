"""Local source-equivalence/cache acceptance only, never a method experiment."""
import argparse
import copy
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import read_json, write_json
from experiments.embedding_heads.bundle import (
    validate_manifest, preparation_receipt, preparation_code_binding, sha256,
    _preparation_code_binding, PREPARE_EOF_LF_COMPATIBILITY)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    # No receipt rewriting. validate_manifest hashes all actual retained input refs.
    manifest = validate_manifest(args.bundle)
    receipt = preparation_receipt(args.bundle)
    before = sha256(receipt)
    producer = read_json(receipt)
    binding = preparation_code_binding(args.bundle)
    source = Path(__file__).resolve().parent / "prepare_candidate_bundle.py"
    data = source.read_bytes()
    current = hashlib.sha256(data).hexdigest()
    if current != PREPARE_EOF_LF_COMPATIBILITY["reviewed_current_sha256"]:
        raise ValueError("Compatibility acceptance needs the reviewed prepare source revision")
    checks = []
    def case(name, refs, should_accept):
        probe = copy.deepcopy(producer)
        probe["code_refs"] = [r for r in probe["code_refs"] if r["path"] != "tools/prepare_candidate_bundle.py"] + refs
        try:
            result = _preparation_code_binding(probe)
            accepted, detail = True, result
        except ValueError as error:
            accepted, detail = False, str(error)
        checks.append({"property": name, "passed": accepted == should_accept, "detail": detail,
                       "scope": "in-memory source-binding software case; not an execution receipt"})
    def ref(digest):
        return {"path": "tools/prepare_candidate_bundle.py", "sha256": digest}
    case("exact_source_accepted", [ref(current)], True)
    case("only_reviewed_eof_blank_line_accepted",
         [ref(PREPARE_EOF_LF_COMPATIBILITY["reviewed_legacy_sha256"])], True)
    case("semantic_source_change_rejected", [ref(hashlib.sha256(data + b"raise RuntimeError()\n").hexdigest())], False)
    case("unreviewed_second_blank_line_rejected", [ref(hashlib.sha256(data + b"\n").hexdigest())], False)
    case("missing_source_ref_rejected", [], False)
    case("duplicate_source_refs_rejected", [ref(current), ref(current)], False)
    checks.append({"property": "actual_old_receipt_unchanged", "passed": sha256(receipt) == before})
    success = all(c["passed"] for c in checks)
    write_json(args.out, {"schema": "cvpr.prepare-compatibility-acceptance.v1",
        "status": "PROPERTY_CHECKS_PASS" if success else "PROPERTY_CHECKS_FAIL",
        "actual_bundle_sha256": sha256(args.bundle), "actual_receipt_sha256": before,
        "preparation_code_binding": binding, "checks": checks,
        "cache_ref_integrity": "validated actual manifest refs",
        "native_model_scoring_qualification": "SEPARATE_REQUIRED",
        "method_experiments": 0, "scientific_verdict": "NONE", "gate_advanced": False,
        "budget_reset": False, "model_id": manifest["model_id"]})
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
