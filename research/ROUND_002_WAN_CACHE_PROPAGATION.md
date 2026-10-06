# Round 002 — Wan cache-error propagation

## Scientific decision

Question: does the final consequence of a cache approximation depend strongly on where the same-sized local approximation is injected into the denoising trajectory?

If yes, local cache-error heuristics omit a downstream amplification term. If no, a propagation-aware method is unnecessary and this route should be killed.

## Mathematical discovery card

Let one denoising update be z_(k+1) = F_k(z_k, f_k(z_k)), where f_k is the expensive DiT computation. Reusing a cached computation changes it to f_tilde_k = f_k + epsilon_k.

For a small perturbation:

delta z_(k+1) ≈ A_k delta z_k + B_k epsilon_k,

with A_k = dF_k/dz_k and B_k = dF_k/df_k.

If only step k is perturbed and later steps are recomputed normally, then to first order:

delta z_T ≈ Phi_(T,k+1) B_k epsilon_k,

where Phi_(T,k+1) is the product of subsequent local Jacobians.

A local cache trigger estimates something related to ||epsilon_k||. The terminal consequence depends on both local error and downstream amplification ||Phi_(T,k+1) B_k||.

Round 002 estimates empirical amplification:

a_hat_k = D(v_tilde_k, v_ref) / (e_k + eps),

where e_k is the counterfactual local denoiser-output error from reusing the preceding residual at step k and D is decoded-video deviation under the same prompt and seed.

This is a mechanism diagnostic, not a theorem about perceptual quality.

## Frozen experimental design

- Model: Wan2.1-T2V-1.3B.
- Inputs: deterministic prospective subset of released VBench prompts; no handcrafted evaluation prompts.
- Reference and perturbation use identical prompt, seed, solver, denoising steps, CFG, frame count and resolution.
- Default window: 3 VBench prompts, 50 denoising steps, five single-step interventions near 10/30/50/70/90 percent of the trajectory.
- Source revisions are recorded in research/SOURCES_ROUND_002.md.

## Decisive controls

1. Exact reference with no cache substitution.
2. Force cache reuse at exactly one diffusion step, then resume full computation.
3. TeaCache-style local proxy: relative change in timestep-conditioned embedding.
4. Stronger local oracle: actual counterfactual denoiser-output error on the reference trajectory.

The decisive test is whether terminal damage varies beyond what controls 3 and 4 predict.

## Metrics

- terminal decoded-video MSE, MAE and PSNR relative to exact reference;
- temporal-gradient MSE relative to reference;
- local timestep-embedding relative L1;
- local residual relative L1;
- local counterfactual denoiser-output relative L1;
- empirical amplification a_hat_k;
- Pearson/Spearman relation between local proxies and terminal deviation.

These mechanism diagnostics do not replace VBench quality scoring. Any later cache policy must use native VBench scoring and compute-matched published/simple cache baselines.

## Pre-registered decision

KILL if valid runs show that local denoiser error already predicts terminal deviation nearly monotonically and empirical amplification is approximately constant.

CONTINUE only if amplification varies materially across timesteps/prompts and interventions with comparable local error can cause meaningfully different terminal damage.

A CONTINUE decision authorizes method design only; it is not a paper claim.

## Resource envelope

User-reported hardware: 1 × RTX 2080 Ti with 22 GB VRAM.

The runner has a 7.5-hour wall-time boundary by default. Completed evidence is retained and unfinished frozen work may carry forward unchanged.
