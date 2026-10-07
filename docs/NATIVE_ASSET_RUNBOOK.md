# Native acquisition / Local acceptance

Status: **generated_unexecuted**. This is acquisition maintenance for the retained r003 inputs
and an offline bridge to the official native parsers. It is not an accepted top-15/G01 scientific
queue. Web inspected source/metadata and used only the existing read-only evidence checker;
no new project code, software tests, model, SSH/GPU or benchmark was executed.

Read [Local entry](../LOCAL_AGENT_RUNBOOK.md), [round handoff](../rounds/r003/WEB_HANDOFF.md),
[math review](../research/reviews/R003_REVIEW_2026-10-07.md) and
[skill delta](../research/AUTOPILOT_DELTA_2026-10-07_99d8ac0.md) at the exact delivered commit.

## Download datasets and models

[The complete immutable lock](../configs/native-assets.lock.json) records every path, size,
Git blob identity and upstream LFS SHA256. Official HF metadata was inspected on 2026-10-07.
Weights, ZIPs and Parquet content were not downloaded on Web. All four sources were public,
ungated and non-private. Local reuses existing access; any later real access block remains pending.
No HF output upload or authentication secret is required by this card.

| Input / consumers | Immutable source revision | Complete required files / bytes | Destination / loader |
|---|---|---|---|
| Same frozen 2B model for native baseline, retained 12 heads and 6 controls | Qwen/Qwen3-VL-Embedding-2B @ 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda | All 18 files; 4,271,068,726 bytes; one complete safetensors file, no weight shard index at this revision | ASSET_ROOT/hf/model; local MMEBEmbeddingModel.load |
| Three complete native tests | ziyjiang/MMEB_Test_Instruct @ bb7e445a62555136b6fa83f17f02f0cc9d5b8962 | README plus ScienceQA, ChartQA, MSCOCO_i2t complete single test shards; 4,189,094 bytes | ASSET_ROOT/hf/test; exact locked physical test Parquet |
| Original eval pictures | TIGER-Lab/MMEB-eval @ 2f069730be515ea60778413777816b53e2d2a697 | Full images.zip; 7,128,863,486 bytes | ASSET_ROOT/hf/eval-images → ASSET_ROOT/test-images |
| Original fitting/PCA/teacher train inputs | TIGER-Lab/MMEB-train @ 0c3f4b828d347c4e8508339f99530f6c820061fd | Six files: README, unzip source, both task train shards and both task ZIPs; 1,040,177,030 bytes | ASSET_ROOT/hf/train; ASSET_ROOT/train-images |

Model files: .gitattributes, 1_Pooling/config.json, README.md, added_tokens.json,
chat_template.jinja, config.json, config_sentence_transformers.json, merges.txt,
model.safetensors, modules.json, preprocessor_config.json, scripts/qwen3_vl_embedding.py,
sentence_bert_config.json, special_tokens_map.json, tokenizer.json, tokenizer_config.json,
video_preprocessor_config.json, vocab.json. All processor/tokenizer/config components are retained.

Test paths: ScienceQA/test-00000-of-00001.parquet, ChartQA/test-00000-of-00001.parquet,
MSCOCO_i2t/test-00000-of-00001.parquet.
Train paths: ScienceQA/train-00000-of-00001.parquet, A-OKVQA/train-00000-of-00001.parquet,
images_zip/ScienceQA.zip, images_zip/A-OKVQA.zip, README.md, unzip_file.py.
All physical train rows are acquired; historical first-128 eligible rows/task and rank-32 PCA
remain developmental settings, not complete CVPR validation.

Raw acquisition totals **12,444,298,336 bytes**. Referenced uncompressed pictures, environment,
partial/cache overhead, feature/output storage and available capacity are unmeasured.
Extraction derives exact required sizes from verified ZIP directories and checks an additional
1 GiB operational margin. Existing 35 GiB guidance is not measured sufficiency. Actual network
time/RAM/VRAM/complete native costs must be charged to the remaining cumulative authorization.

Stronger controls remain science obligations: Qwen learned MRL prefix, optimized CE/listwise KD,
outside-log SupCon, correct metric-distance scoring, online group-DRO, Procrustes and fitted
shrinkage. Reading an author source does not implement/qualify its control. Future rerankers,
video models, judges and additional data are not silently declared covered.

## Real host / single harness / budgets

Local runs on the user's controller and restores the existing SSH alias, delivery checkout,
actual remote project/runtime/Conda roots, GPU UUID/driver/capacity, live attempts and original
remaining budgets. Web restored no SSH/GPU path or telemetry. Do not use a Web scratch path remotely.

The following are **inner foreground command cards for admitted jobs**, cwd the actual pinned
remote checkout. Local binds real absolute variables: CVPR_PROJECT_DIR, CVPR_CONDA_PREFIX,
CVPR_ASSET_ROOT, CVPR_RECEIPT_ROOT, RESEARCH_AUTOPILOT_ROOT, CVPR_STAGED_UPSTREAM,
CVPR_TEST_EXTRACT_RECEIPT, CVPR_BASELINE_OUT, CVPR_SETUP_RECEIPT_DIR and CVPR_JOB_SECONDS.
Use shell unset-variable checking in accepted command wrappers. Unknown bindings keep launch pending.

Use one remote run_harness owner, whose native runner executes each command. CPU setup/acquisition
requests gpu_count=0 and actual CPU/RAM reservations plus exclusive cache/prefix/output keys.
GPU native qualification uses one admitted UUID with an unknown peak treated as exclusive.
No actual outer/native plan is frozen here: Local must bind existing code/config/environment hashes,
real resources, stable IDs, finite remaining bounds, dependencies and output obligations first.
No guessed plan path, future receipt, fake digest or automatic cross-window dispatcher is supplied.

Historical bounds remain 28,800 cumulative seconds, 18 method/control attempts, zero retries,
one preparation attempt, 7,200-second preparation and 600-second trial caps. The new skill treats
eight-hour reporting as soft and allows future Local cycles up to 24 hours only within actual
remaining authorization. It does not change old active digests/watchdogs/markers or reset budgets.
Inspect both runner layers before accepting any pending operational plan.

## Ordered command cards

1. Controller/source staging: fetch literal main into an isolated compatible checkout, read
AGENTS/runbook/handoff at the delivered SHA, preserve active work. Current complete source check,
a read-only controller action:

~~~bash
python3 tools/check_autopilot_source.py --skill-root "$RESEARCH_AUTOPILOT_ROOT" \
  --lock configs/autopilot-source-99d8ac0.json
~~~

The complete author source is Yunbo-max/Research_Autopilot @
99d8ac079682e91f61ab59bfd3e760eccb8b5726; follow its docs/INSTALL_LOCAL.md with all four sibling
skills. For old pinned attempts use configs/autopilot-source.json. Skill source is not vendored
into CVPR. Source bytes matching is separate from installation/hooks/runtime acceptance.

Official baseline/scorer source acquisition, on controller, only when the clean checkout is absent:

~~~bash
git clone --no-checkout https://github.com/QwenLM/Qwen3-VL-Embedding.git sources/Qwen3-VL-Embedding
git -C sources/Qwen3-VL-Embedding checkout --detach 393e2978d27852b0d0230d6994f37f9c15bed73c
git -C sources/Qwen3-VL-Embedding rev-parse HEAD
~~~

Stage a clean per-attempt copy using the existing authorized transport/path map.
tools/source_adapter.py modifies declared pinned source files in that staged copy; do not reuse
an adapted copy as clean source. Current FP16/SDPA compatibility is unqualified on the actual host.

2. Native Conda, dependency-ordered admitted CPU jobs. Reuse a compatible inactive prefix.
If creation is needed:

~~~bash
conda env create --prefix "$CVPR_CONDA_PREFIX" --file configs/environment-native.yml
bash tools/setup_native_env.sh "$CVPR_CONDA_PREFIX" "$CVPR_SETUP_RECEIPT_DIR"
~~~

Python 3.11 is a requested family, not a solved transitive lock. Existing Torch 2.8.0/
torchvision .23.0 cu126 and model dependencies are retained; requirements-assets.txt adds explicit
HF Hub .36.0 and PyArrow 22.0.0 (official 2025-10-24 release).
Setup exports project/source hashes, pip install reports, conda-explicit, pip-freeze and failure
exit code. Local must inspect dependency resolution and driver compatibility. Install exit 0 is
not software/native acceptance. The old .venv bootstrap remains historical.

3. Planned Local software checks in admitted CPU jobs:

~~~bash
conda run --no-capture-output --prefix "$CVPR_CONDA_PREFIX" python -m py_compile \
  tools/acquire_native_assets.py tools/prepare_assets.py tools/qualify.py
conda run --no-capture-output --prefix "$CVPR_CONDA_PREFIX" python -m unittest \
  discover -s tests -p 'test_candidate*.py' -v
~~~

Web did not execute them. These checks do not cover real downloads, ZIP schema/path mapping,
model or native scorer. Local additionally accepts the real immutable metadata/loader coverage
and official native replay. No handmade scientific eval fixtures replace these checks.

4. Catalog → download → independent verification → per-split extraction. Each command is one
admitted CPU job with real finite CVPR_JOB_SECONDS and a new immutable receipt directory:

~~~bash
conda run --no-capture-output --prefix "$CVPR_CONDA_PREFIX" python tools/acquire_native_assets.py \
  --lock configs/native-assets.lock.json --root "$CVPR_ASSET_ROOT" \
  --out "$CVPR_RECEIPT_ROOT/catalog" --action catalog --seconds "$CVPR_JOB_SECONDS"

conda run --no-capture-output --prefix "$CVPR_CONDA_PREFIX" python tools/acquire_native_assets.py \
  --lock configs/native-assets.lock.json --root "$CVPR_ASSET_ROOT" \
  --out "$CVPR_RECEIPT_ROOT/download" --action download --seconds "$CVPR_JOB_SECONDS"

conda run --no-capture-output --prefix "$CVPR_CONDA_PREFIX" python tools/acquire_native_assets.py \
  --lock configs/native-assets.lock.json --root "$CVPR_ASSET_ROOT" \
  --out "$CVPR_RECEIPT_ROOT/verify" --action verify --seconds "$CVPR_JOB_SECONDS"

conda run --no-capture-output --prefix "$CVPR_CONDA_PREFIX" python tools/acquire_native_assets.py \
  --lock configs/native-assets.lock.json --root "$CVPR_ASSET_ROOT" \
  --out "$CVPR_RECEIPT_ROOT/extract-test" --action extract --split test --seconds "$CVPR_JOB_SECONDS"

conda run --no-capture-output --prefix "$CVPR_CONDA_PREFIX" python tools/acquire_native_assets.py \
  --lock configs/native-assets.lock.json --root "$CVPR_ASSET_ROOT" \
  --out "$CVPR_RECEIPT_ROOT/extract-train" --action extract --split train --seconds "$CVPR_JOB_SECONDS"
~~~

The inspected SDK fetches each literal filename/full revision/type/local_dir, never latest HEAD.
Its .cache/huggingface partial/metadata files stay under each group; .36.0 resumes transport when
possible. No project retry loop. Blocking SDK calls require the outer finite timeout; hashing/
ZIP work checks the inner deadline. Corrupt existing files are retained and failed. Independent
files/groups continue within the same remaining budget. An entire deadlocked/access-blocked task
is not automatically rerun. Direct HF CLI acquisition is unnecessary because this SDK card
enforces the complete manifest and official SHA identities.

Extraction verifies exactly the split dependencies: test plus eval-images, or train.
A missing model does not block image extraction. Every physical released row is read, original
target text order and first-target labels retained. Structural ZIP mapping rejects ambiguous
duplicates/traversal/symlinks, reads to original CRC completion and records actual image SHA256.
Interrupted partial files/evidence are retained; no generated pictures or sampled eval rows.

5. Offline baseline bridge, once current Local software/inputs and actual harness are accepted.
Inspected exact CLI, **currently not dispatch-ready**:

~~~bash
conda run --no-capture-output --prefix "$CVPR_CONDA_PREFIX" python tools/qualify.py \
  --config configs/batch.json --upstream "$CVPR_STAGED_UPSTREAM" \
  --asset-lock configs/native-assets.lock.json --asset-root "$CVPR_ASSET_ROOT" \
  --asset-receipt "$CVPR_TEST_EXTRACT_RECEIPT" \
  --out "$CVPR_BASELINE_OUT" --seconds "$CVPR_JOB_SECONDS"
~~~

CVPR_TEST_EXTRACT_RECEIPT is the actual completed extract-test/receipt.json.
prepare_assets.py prepare_locked verifies model/test/archive files and extraction hashes,
records processor inventory/revisions and reads exact locked Parquet paths.
qualify.py intercepts the three official parser loads to those paths, retaining complete rows/
labels/candidates and RankingMetrics replay. It does no new method training or gate advancement.
ScienceQA/ChartQA are current core tasks and COCO is an admission reserve: **all three are required
for complete baseline/cache/comparison**. Budget-skipped/failed COCO stays pending, not full completion.

6. Training/historical queue: actual interfaces remain prepare_candidate_bundle.py,
run_candidates.py, run_method.py and collect_candidates.py; historical commands stay in
[LOCAL_AGENT_HANDOFF](LOCAL_AGENT_HANDOFF.md). New train roots map to ASSET_ROOT/hf/train and
ASSET_ROOT/train-images via --train-metadata-root/--train-image-root. Keep the immutable revision,
physical-train/original explanation, first-128 policy and train/eval overlap exclusion. Changed
producer bytes require child cache requalification. Do not use the old raw 12+6 queue as current
top-15 acceptance. Pool/selection/Parent/Natural Gate0/IPCG/G01 remain incomplete.

7. Local outer-plan review/freeze/launch: current author CLI shapes, with actual Local plan values:

~~~bash
python3 "$RESEARCH_AUTOPILOT_ROOT/scripts/run_harness.py" "$CVPR_DRAFT_HARNESS_PLAN" \
  --root "$CVPR_PROJECT_DIR" --freeze
python3 "$RESEARCH_AUTOPILOT_ROOT/scripts/run_harness.py" "$CVPR_FROZEN_HARNESS_PLAN" \
  --root "$CVPR_PROJECT_DIR" --execute --approved-plan-digest "$CVPR_APPROVED_HARNESS_DIGEST"
~~~

Retain actual freeze output before execution. This card creates no plan/authority or new attempts.
Do not add --stop-after-report for routine continuous execution. Future candidate experiment-design/
dispatch evidence checks bind the genuinely selected/current batch and all scientific gates.

## Native coverage

| Benchmark | Pinned official adapter | Required native semantics |
|---|---|---|
| ScienceQA | image_qa_dataset.py data_prepare/load_image_qa_dataset | All released test rows, original text candidates, first-target label, full local rankings |
| ChartQA | Same official image_qa adapter | Full original test/labels/candidates and parser/scorer parity |
| MSCOCO_i2t | image_i2t_eval.py data_prepare/load_image_i2t_dataset | Complete released local candidate lists; do not substitute global COCO caption retrieval |

image.yaml configures eval_type local. Official metrics.py RankingMetrics blob is
097365b58831711d0b04eef08f9d4232f6486d6c @ Qwen revision
393e2978d27852b0d0230d6994f37f9c15bed73c. replay_native.py invokes that unchanged scorer.
Primary metric hit@1 and all native ranking metrics remain intact. Actual denominators come
from immutable Parquet on Local and must equal parser/native prediction counts; none are invented.

## Logs / faults / complete return

Every acquisition action writes OUT/receipt.json and OUT/events.jsonl plus stdout JSON.
Actual outer/native receipts own stdout/stderr, process/host context, attempt status and budgets;
resolve paths from those receipts. A maintenance exit 0 confers no scientific verdict.
Qualification outputs assets/assets.json, resolved-revisions.json, native-samples/TASK.jsonl,
baselines/TASK/native-config.yaml, native query/target caches, TASK_info.jsonl/TASK_pred.jsonl/
TASK_score.json, replay.json, events.jsonl and summary.json. Keep failed/skipped tasks too.

| Fault | Actual source/evidence | Supported recovery / recheck |
|---|---|---|
| SDK/access/HTML/LFS pointer/shard/checksum | acquire_native_assets.py load_lock/verify_file; lock; receipt/events/stderr | Restore same source/access/version; retain corrupt/partial inputs; bounded accepted recovery and verification |
| Schema/image path/empty split/ZIP mapping | required_images/extract_images; real Parquet/ZIP member names/native_inventory | Inspect pinned parser/unzip source; repair supported structural mapping and verify every original image |
| Duplicate/symlink/traversal/existing-image mismatch | extract_images and retained partial evidence | Stop affected split, investigate actual source/path; independent ready split can proceed |
| Native import/Conda/CUDA | setup reports, requirements and actual host/attempt stderr | Reuse inactive compatible native prefix; review/version cause and repeat affected acceptance |
| Receipt/path/producer mismatch | prepare_locked; extraction refs; prepare_candidate_bundle baseline_evidence | Restore identity/path map; requalify child outputs; never certify old cache using changed producer |
| Denominator/candidate/scorer mismatch | qualify.py pinned_dataset/instantiate; raw native samples; official replay | Qualify full original rows/labels/candidates and live scorer; preserve bad outputs, no replacement score |
| Timeout/OOM/unknown activity | Actual harness context/attempt; window_budget/carryover audit | Reconcile same host/IDs/remaining budget; no clock reset, marker deletion or duplicate queue |

Return actual execution/main and skill SHAs, dirty patch, path map, source/Conda manifests,
all acquisition/extraction/software/native receipts, raw predictions/replay, every failure/
pending task, cumulative costs, current scientific prerequisites and E04 status. Retrieve through
existing authorized transport and preserve hash/path provenance. No HF output or private source
upload. Existing collect_candidates.py is historical collection, not a new top-15 E04 packet.

A coherent result packet goes to yuhanlydia/CVPR literal main through one integration writer
with exact readback. Web reads the full packet before a new direction or scientific verdict.

Primary sources: official HF tree at each full revision above; pinned
https://github.com/huggingface/huggingface_hub/blob/v0.36.0/src/huggingface_hub/file_download.py;
https://github.com/apache/arrow/releases/tag/apache-arrow-22.0.0.
