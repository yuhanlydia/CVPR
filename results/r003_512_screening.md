# R003 512-Row Screening Results

This is a developmental screening run on the frozen Qwen3-VL-Embedding-2B setup.

- Training rows: 512 per task for ScienceQA and A-OKVQA; 1,019 unique rows after deterministic deduplication.
- Evaluation: 1,000 examples each from ScienceQA, ChartQA, and MSCOCO_i2t.
- Runtime: 22m54s on an NVIDIA GeForce RTX 2080 Ti.
- Outcome: 17/18 methods completed; I01 was blocked because independent interval calibration was unavailable.
- Replay checks: 51/51 completed-task replays matched; train/eval image and query intersections were zero.
- Scientific status: developmental only; Gate A was not advanced.

| Method | ScienceQA hit@1 | ChartQA hit@1 | MSCOCO_i2t hit@1 | Mean |
|---|---:|---:|---:|---:|
| C_ERM | 0.421 | 0.394 | 0.491 | 0.435 |
| C_KD | 0.400 | 0.339 | 0.371 | 0.370 |
| C_PROJECTED | 0.419 | 0.360 | 0.495 | 0.425 |
| C_RBF | 0.419 | 0.360 | 0.495 | 0.425 |
| C_SINGLE | 0.421 | 0.393 | 0.493 | 0.436 |
| C_WHITEN | 0.434 | 0.388 | 0.495 | 0.439 |
| I01 | blocked | blocked | blocked | blocked |
| I02 | 0.378 | 0.336 | 0.365 | 0.360 |
| I03 | 0.153 | 0.126 | 0.222 | 0.167 |
| I05 | 0.421 | 0.395 | 0.492 | 0.436 |
| I09 | 0.455 | 0.426 | 0.472 | 0.451 |
| I10 | 0.202 | 0.168 | 0.117 | 0.162 |
| I11 | 0.421 | 0.394 | 0.491 | 0.435 |
| I12 | 0.026 | 0.003 | 0.021 | 0.017 |
| I13 | 0.438 | 0.391 | 0.495 | 0.441 |
| I14 | 0.399 | 0.351 | 0.381 | 0.377 |
| I16 | 0.191 | 0.136 | 0.345 | 0.224 |
| I17 | 0.382 | 0.377 | 0.486 | 0.415 |

The strongest developmental candidate is I09 by mean hit@1 (0.451), followed by I13 (0.441). I05 is stable and comparable to the controls, but this run is not sufficient for a scientific claim.
