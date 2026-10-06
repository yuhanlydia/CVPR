# Round 002 pinned sources

Wan2.1: Wan-Video/Wan2.1 @ 9737cba9c1c3c4d04b33fcad41c111989865d315

TeaCache: ali-vilab/TeaCache @ 7c10efc4702c6b619f47805f7abe4a7a08085aa0

VBench: Vchitect/VBench @ fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490

Wan checkpoint: Wan-AI/Wan2.1-T2V-1.3B @ 37ec512624d61f7aa208f7ea8140a131f93afc9a

## Direct collision found in review, 2026-10-06

[RA-CFGCache](https://arxiv.org/html/2609.36433v1), submitted 2026-09-29.
Checked Sec. 3.4, Appendix B.3/C.2 and
[author implementation](https://github.com/yiming-l21/RA-CFGCache/tree/e17d6a74ac2f7ca68d9ee9b8af8bfd9bc3886d8a).
It combines CFG branch errors and calibrates timestep-dependent propagation from
isolated reuse perturbations, including Wan2.1-T2V-1.3B. The present question cannot
serve as the novel contribution. This is a literature collision, distinct from an
empirical KILL: no local model result has been obtained.

Its calibration/controller code was read for the audit; no GPL code was copied here.

Closest-work collision snapshot: TeaCache, Error-Optimized Cache, ProCache, SenCache, SODA, and SpectralCache already cover local error, sensitivity, non-uniform scheduling, or cumulative error. Therefore Round 002 makes no new-method claim. It only tests whether downstream error amplification contains useful information beyond local cache-error quantities.
