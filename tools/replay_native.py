"""Replay the unchanged upstream scorer on real persisted native predictions."""
import argparse
import json
import sys
from pathlib import Path
from common import digest, read_json, write_json, now


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.upstream).resolve()))
    from src.evaluation.mmeb_v2.utils.eval_utils.metrics import RankingMetrics
    predictions = [json.loads(line) for line in Path(a.predictions).read_text(encoding="utf-8").splitlines()]
    result = RankingMetrics(["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]).evaluate(predictions)
    recorded = read_json(a.scores)
    if recorded["num_pred"] != len(predictions) or recorded["num_data"] != len(predictions):
        raise ValueError("Native denominator mismatch")
    for key, value in result.items():
        if abs(value - recorded[key]) > 1e-12:
            raise ValueError(f"Native scorer replay mismatch: {key}")
    write_json(a.out, {"replayed_at": now(), "status": "matched", "denominator": len(predictions),
                       "prediction_sha256": digest(a.predictions), "metrics": result,
                       "scorer_sha256": digest(Path(a.upstream) / "src/evaluation/mmeb_v2/utils/eval_utils/metrics.py")})


if __name__ == "__main__":
    main()
