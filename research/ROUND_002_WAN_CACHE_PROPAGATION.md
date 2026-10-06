# Round 002 — Wan cache-error propagation

## Scientific question — decision contract pending

Question: does the final consequence of a cache approximation depend strongly on where the same-sized local approximation is injected into the denoising trajectory?

This candidate mechanism requires native scoring and matched perturbation controls.
The current executable is engineering qualification; its outputs alone cannot
authorize KILL/CONTINUE or establish novelty.

## Mathematical discovery card

Let one denoising update be z_(k+1) = F_k(z_k, f_k(z_k)), where f_k is the expensive DiT computation. Reusing a cached computation changes it to f_tilde_k = f_k + epsilon_k.

For UniPC, z must include the scheduler's multistep history, not just the current
latent. Terminal video error also includes the decoder Jacobian.

For a small perturbation:

delta z_(k+1) ≈ A_k delta z_k + B_k epsilon_k,

with A_k = dF_k/dz_k and B_k = dF_k/df_k.

If only step k is perturbed and later steps are recomputed normally, then to first order:

delta z_T ≈ Phi_(T,k+1) B_k epsilon_k,

where Phi_(T,k+1) is the product of subsequent local Jacobians.

A local cache trigger estimates something related to ||epsilon_k||. The terminal consequence depends on both local error and downstream amplification ||Phi_(T,k+1) B_k||.

The review corrects the degree of the empirical error ratio:

a_hat_k = RMS(v_tilde_k - v_ref) / RMS(epsilon_guided_k),

where epsilon_guided = epsilon_uncond + CFG*(epsilon_cond-epsilon_uncond).
The denominator is the absolute counterfactual guided denoiser-output error.
Zero-denominator cases are reported but excluded from the ratio.

The previous MSE/relative-L1 ratio scaled linearly with perturbation amplitude
even for a constant linear operator: MSE(c*J*epsilon)/relL1(c*epsilon) is
proportional to c. It therefore could not establish variable propagation gain.
The matched RMS ratio is invariant under c in the linear regime, but remains
direction-specific and includes decoder/solver effects. Different cache-error
directions across timesteps are a confound; norm matching alone does not isolate time.

This is a mechanism diagnostic, not a theorem about perceptual quality.

## Frozen experimental design

- Model: Wan2.1-T2V-1.3B.
- Inputs: deterministic prospective subset of released VBench prompts; no handcrafted evaluation prompts.
- Reference and perturbation use identical prompt, seed, solver, denoising steps, CFG, frame count and resolution.
- Inherited prospective broader design: 3 VBench prompts, 50 denoising steps, five single-step interventions near 10/30/50/70/90 percent of the trajectory.
- First engineering qualification: one original prompt, 50 steps, one intervention at step 5, at most two hours. Unknown host throughput/peak memory is measured before admitting the broader design.
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
- direction-specific absolute-RMS gain a_hat_k;
- Pearson/Spearman relation between local proxies and terminal deviation.

These quantities are descriptive engineering diagnostics. No scientific gate may
use them as a substitute for the native VBench scorer. A scientific protocol must
freeze official prompt/video identities, seeds, native scorer version and denominator,
scorer replay, compute-matched published/simple controls and resource/uncertainty receipts.

## Candidate decision — inactive until preregistration is complete

KILL if valid runs show that local denoiser error already predicts terminal deviation nearly monotonically and empirical amplification is approximately constant.

CONTINUE only if amplification varies materially across timesteps/prompts and interventions with comparable local error can cause meaningfully different terminal damage.

The phrases "nearly monotonically", "approximately constant" and "materially"
have no numerical thresholds in the inherited card. Freeze thresholds, uncertainty,
prompt/seed replication, amplitude-matched and direction controls before collecting
scientific decision evidence. Current outputs cannot choose either verdict.

The closest-work list is a collision warning, not a completed originality audit.
Literature blocker: RA-CFGCache; see SOURCES_ROUND_002.md. Keep the current
question as reproduction/baseline. These descriptive outputs cannot authorize a new policy.
Current Parent/Natural/implementation qualification and native-benchmark Gate A
are prerequisites before a new cache policy is implemented.

## Resource envelope

User-reported hardware: 1 × RTX 2080 Ti with 22 GB VRAM.

The installed native runner hard-bounds one engineering attempt at up to 7.5 hours,
including preflight, hashing and loading. Setup retains the same original eight-hour
clock. Unique attempt directories, shared GPU lock and zero retries retain failures
and partial evidence. No automatic second window is launched.
