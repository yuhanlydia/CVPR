"""Checked compatibility edits to an isolated, pinned upstream checkout.

The dataset parser, candidate lists, pooling and RankingMetrics remain upstream.
FP16/SDPA is a declared protocol variation, not an exact leaderboard reproduction.
"""
import hashlib
from pathlib import Path

UPSTREAM_COMMIT = "393e2978d27852b0d0230d6994f37f9c15bed73c"
EXPECTED_BLOBS = {
    "src/evaluation/mmeb_v2/eval_embedding.py": "fe25a1563d0adbf57b25b707dc0b4ead4ec0eff1",
    "src/models/qwen3_vl_embedding.py": "a2d4a73349c4648c5d66633eccd57e02aea1279f",
    "src/evaluation/mmeb_v2/data/datasets/__init__.py": "fa8296ecc5e64cd81573312f5eb2876394988325",
}


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def transform(path, text):
    if path.endswith("datasets/__init__.py"):
        # This window evaluates image tasks only; do not import unrelated video/visdoc dependencies.
        return ("from .image_qa_dataset import load_image_qa_dataset\n"
                "from .image_i2t_eval import load_image_i2t_dataset\n"
                "from .image_t2i_eval import load_image_t2i_dataset\n")
    if path.endswith("eval_embedding.py"):
        if text.count("torch.bfloat16") != 4 or text.count("'flash_attention_2'") != 2:
            raise ValueError("Upstream precision/attention edit locations changed")
        if text.count("local_embeds.append(reps)") != 1:
            raise ValueError("Upstream embedding retention changed")
        text = text.replace("torch.bfloat16", "torch.float16")
        text = text.replace("'flash_attention_2'", "'sdpa'")
        # One GPU only: release each encoded batch from VRAM immediately.
        text = text.replace("local_embeds.append(reps)", "local_embeds.append(reps.cpu().float())")
        return text
    if path.endswith("qwen3_vl_embedding.py"):
        marker = '            logger.error(f"Error in processing vision info: {e}")'
        if text.count(marker) != 1:
            raise ValueError("Upstream vision error handling changed")
        return text.replace(marker, marker + '\n            raise RuntimeError("Invalid native vision input; no NULL substitution allowed") from e')
    raise ValueError("No declared edit for this path")


def adapt(root):
    changes = []
    for relative, expected in EXPECTED_BLOBS.items():
        path = Path(root) / relative
        raw = path.read_bytes()
        if expected is None or blob_sha(raw) != expected:
            raise ValueError(f"Unrecognized upstream blob: {relative}")
        new = transform(relative, raw.decode("utf-8")).encode("utf-8")
        path.write_bytes(new)
        changes.append({"path": relative, "original_blob": expected,
                        "adapted_sha256": hashlib.sha256(new).hexdigest()})
    return changes
