# Followup Analysis: Generative Modeling via Drifting

**Date**: 2026-05-18
**Seed paper**: [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) (arxiv 2602.04770)
**Authors**: Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He
**Venue / Year**: arXiv preprint, 2026
**Total citations (Semantic Scholar)**: 29
**Follow-ups analyzed**: 29 of 29 (full coverage — every citing paper had an abstract; abstract-only tagging via 4 parallel subagents)

## TL;DR

- **The field jumped on this paper hard and fast.** 29 follow-ups in ~3 months, dominated by **Extensions** (13/29 = 45%) — drifting has already been ported to speech enhancement, MRI/CT, robot policy, autoregressive vision, wireless decoding, motion planning, RL, trajectory planning, and Boltzmann sampling. This is not "interesting baseline" reception; it's "paradigm adoption."
- **A parallel theory wave is settling the open questions Deng et al. left open.** 8/29 follow-ups are **Theoretical** and most agree on one finding: the drift field is *score matching on a kernel-smoothed distribution* (Tweedie's formula for the Gaussian kernel), with three independent groups giving converging analyses (Lai et al., Cao et al., Turan & Ovsjanikov). Two further papers (Franz et al., Lee) characterize when the field is non-conservative and when identifiability holds.
- **There's one credible "Improves" claim already.** Yang et al. (Representation Fréchet Loss, 2 cits) report **0.72 FID** on ImageNet-256 one-step vs Drifting's 1.54 — half the FID at the same NFE budget. Two other improvers tackle bias (ABC) and lookahead, but on smaller benchmarks (CIFAR-10).
- **Notably absent**: no `Criticizes` tag. The follow-ups either build on it, theoretically validate it, or beat it on its own benchmark — nobody has reported it failing yet. That's a strong early-cycle signal.

## Cluster summary

| Tag | Count | Top follow-ups (by their own citation count) |
|-----|-------|----------------------------------------------|
| Extends      | 13 | Sinkhorn-Drifting (8), Receding-Horizon Control (1), Drifting-to-Boltzmann molecular (1) |
| Theoretical  | 8  | Unified View Drifting/Score (7), Gradient Flow Drifting (5), Drifting is Secretly Score Matching (3) |
| Improves     | 3  | Representation Fréchet Loss (2), Analytical Bias Correction (0), Lookahead Drifting (0) |
| Applies      | 1  | MRI-to-CT synthesis (1) |
| Benchmarks   | 1  | ELT Elastic Looped Transformers (2) |
| Mentions     | 3  | Consistency-Regularised GF (0), V-Co (0), ABounD (0) |
| Criticizes   | 0  | — |

## Extensions

Papers that build a new variant, new modality, or new application of the Drifting Models paradigm — ranked by their own citation count, then by recency.

### Sinkhorn-Drifting Generative Models — `cit:8` `yr:2026` `arxiv:2603.12366`
**Authors**: Ping He, Om Khangaonkar, Hamed Pirsiavash
**Digest**: Replaces the one-sided normalized Gibbs kernel in Drifting Models with two-sided Sinkhorn-scaled entropic OT couplings, closing the identifiability gap and improving stability at low kernel temperatures while keeping one-step inference.
> "We establish a theoretical link between the recently proposed \"drifting\" generative dynamics and gradient flows induced by the Sinkhorn divergence."

### Receding-Horizon Control via Drifting Models — `cit:1` `yr:2026` `arxiv:2604.04528`
**Authors**: Daniele Foffano, Alessio Russo, Alexandre Proutiere
**Digest**: Proposes Drifting MPC, an offline trajectory-optimization framework that adapts drifting generative models to conditional trajectory generation under unknown dynamics, biasing the learned distribution toward optimal plans while retaining one-step inference.
> "we propose Drifting MPC, an offline trajectory optimization framework that combines drifting generative models with receding-horizon planning under unknown dynamics"

### Drifting to Boltzmann: Million-Fold Acceleration in Boltzmann Sampling — `cit:1` `yr:2026` `arxiv:2603.05527`
**Authors**: Pipi Hu
**Digest**: Extends Drifting Models to molecular conformation generation by deriving a "Drifting Score Identity" linking the drifting field to score functions and injecting molecular force labels to correct training-distribution bias toward the true Boltzmann distribution.
> "We introduce Drifting Models to molecular conformation generation for the first time, establishing a theoretical bridge via the Drifting Score Identity"

### SymDrift: One-Shot Generative Modeling under Symmetries — `cit:0` `yr:2026` `arxiv:2605.06140`
**Authors**: S. Darouich, Vinh Tong, Lluís Pastor-Pérez
**Digest**: Identifies a symmetry-specific failure of vanilla drifting models and proposes SymDrift, a symmetry-aware drifting-field variant (symmetrized drift + G-invariant embedding) for one-shot equivariant generation of molecular conformers and transition states.
> "drifting models face a symmetry-specific challenge ... we propose SymDrift, a framework that makes the drifting field itself symmetry-aware"

### DriftDecode: One-Step Wireless Image Decoding via Drifting-Inspired Detail Recovery — `cit:0` `yr:2026` `arxiv:2605.02325`
**Authors**: Jingwen Fu, Ming Xiao, Mikael Skoglund
**Digest**: Repurposes the drifting-field mechanism as a perceptual-feature-space texture loss for a one-step U-Net wireless image decoder, trading the generative paradigm for a recovery-oriented variant under AWGN and Rayleigh channels.
> "The loss reformulates the drifting-field mechanism from generative drifting models in perceptual feature space"

### Speech Enhancement Based on Drifting Models (DriftSE) — `cit:0` `yr:2026` `arxiv:2604.24199`
**Authors**: Liang Xu, Diego Caviedes-Nozal, B. Kleijn
**Digest**: Applies the Drifting Models paradigm to speech enhancement, formulating denoising as an equilibrium problem with a learned Drifting Field for one-step inference, with two formulations (direct mapping and stochastic conditional generation).
> "We propose Speech Enhancement based on Drifting Models (DriftSE) ... This evolution is driven by a Drifting Field, a learned correction vector that guides samples toward the high-density regions of the clean distribution"

### MISTY: High-Throughput Motion Planning via Mixer-based Single-step Drifting — `cit:0` `yr:2026` `arxiv:2604.21489`
**Authors**: Yining Xing, Zehong Ke, Yiqian Tu
**Digest**: Adapts the drifting paradigm to autonomous-driving motion planning by introducing a latent-space drifting loss with attractive/repulsive forces, enabling single-step trajectory generation on nuPlan at 99+ FPS.
> "we introduce a latent-space drifting loss that shifts the complex distribution evolution entirely to the training phase. By formulating explicit attractive and repulsive forces, this mechanism empowers the model to synthesize novel, proactive maneuvers"

### Generative Drifting for Conditional Medical Image Generation (GDM) — `cit:0` `yr:2026` `arxiv:2604.19736`
**Authors**: Zirong Li, Siyuan Mei, Weiwen Wu
**Digest**: Extends Drifting Models to 3D conditional medical imaging (MRI-to-CT, sparse-view CT) via an attractive-repulsive drift with multi-level feature banks from a foundation encoder, retaining one-step inference.
> "GDM extends drifting to 3D medical imaging through an attractive-repulsive drift that minimizes the discrepancy between the generator pushforward and the target distribution."

### Positive-Only Drifting Policy Optimization (PODPO) — `cit:0` `yr:2026` `arxiv:2604.16519`
**Authors**: Qi Zhang
**Digest**: Adapts the drifting model to online RL: PODPO performs advantage-weighted local contrastive drifting using only positive-advantage samples, offering a likelihood-free, gradient-clipping-free generative policy update.
> "By leveraging the drifting model, PODPO performs policy updates via advantage-weighted local contrastive drifting."

### Drift-Based Policy Optimization (DBP/DBPO) — `cit:0` `yr:2026` `arxiv:2604.03540`
**Authors**: Yuxuan Gao, Yedong Shen, Shiqi Zhang
**Digest**: Builds a native one-step generative policy for robot manipulation using fixed-point drifting objectives (DBP) and an online RL extension (DBPO) with a stochastic interface, achieving up to 100× faster inference than multi-step diffusion policies.
> "we introduce the Drift-Based Policy (DBP), which leverages fixed-point drifting objectives to internalize iterative refinement into the model parameters, yielding a one-step generative backbone by design"

### Drift-AR: Single-Step Visual Autoregressive Generation via Anti-Symmetric Drifting — `cit:0` `yr:2026` `arxiv:2603.28049`
**Authors**: Zhengwei Zou, Xiaoxiao Ma, Mingde Yao
**Digest**: Adapts drifting to the visual-decoder stage of AR-diffusion hybrids by treating prediction entropy as the variance of an anti-symmetric drifting field, enabling single-step (1-NFE) decoding without iterative denoising or distillation.
> "we reinterpret entropy as the *physical variance* of the initial state for an anti-symmetric drifting field ... enabling single-step (1-NFE) decoding"

### One-Step Sampler for Boltzmann Distributions via Drifting — `cit:0` `yr:2026` `arxiv:2603.17579`
**Authors**: Wenhan Cao, Keyu Yan, Lin Zhao
**Digest**: Extends Drifting Models from data-driven generation to amortized one-step sampling of Boltzmann distributions specified by an energy function, deriving a target-side drift from a smoothed energy plus a sampler-side Gaussian mean-shift score.
> "the results support drifting as an effective way to amortize iterative sampling from Boltzmann distributions into a single forward pass at test time"

### Amortizing Trajectory Diffusion with Keyed Drift Fields — `cit:0` `yr:2026` `arxiv:2603.14056`
**Authors**: Gokul Puthumanaillam, M. Ornik
**Digest**: Ports the drift-field one-step training idea from image generation to conditional trajectory planning in offline RL, introducing a key-space neighborhood for the drift target and a stop-gradient drifted objective.
> "a one-step trajectory generator trained with a drift-field objective ... using a stop-gradient drifted target to amortize iterative refinement into training"

## Theoretical analyses

Papers that explain *why* drifting works, what it equals, when it fails, or how to fix the gaps Deng et al. left open. **A near-consensus has formed: the drift field equals a score difference on a kernel-smoothed distribution.**

### A Unified View of Drifting and Score-Based Models — `cit:7` `yr:2026` `arxiv:2603.07514`
**Authors**: Chieh-Hsin Lai, Bac Nguyen, N. Murata
**Digest**: Proves that the Drifting Models mean-shift field equals a score difference on kernel-smoothed distributions (Tweedie's formula for Gaussian kernels) and derives rigorous error bounds for the Laplace kernel, connecting drifting to score matching and DMD.
> "Drifting models train one-step generators by optimizing a mean-shift discrepancy ... we make its relationship to the score-matching principle behind diffusion models precise by showing that drifting admits a score-based formulation on kernel-smoothed distributions."

### Gradient Flow Drifting: Wasserstein Gradient Flows of KDE-Approximated Divergences — `cit:5` `yr:2026` `arxiv:2603.10592`
**Authors**: Jiarui Cao, Zixuan Wei, Yuxing Liu
**Digest**: Proves the Drifting Model's drift field equals (up to a bandwidth-squared factor) the Wasserstein-2 gradient flow of KL divergence under KDE approximation, unifying drifting and MMD generators, and proposes a mixed reverse-KL/chi-squared extension on Riemannian manifolds.
> "we prove that the drifting field of drifting model (arXiv:2602.04770) equals, up to a bandwidth-squared scaling factor, the difference of KDE log-density gradients ... which is exactly the particle velocity field of the Wasserstein-2 gradient flow of $KL(q\|p)$"

### Generative Drifting is Secretly Score Matching: A Spectral and Variational Perspective — `cit:3` `yr:2026` `arxiv:2603.09936`
**Authors**: Erkan Turan, M. Ovsjanikov
**Digest**: Shows the Gaussian-kernel drift operator is exactly a score difference on smoothed distributions, answers the three open questions left by Deng et al. (identifiability, kernel choice, stop-gradient necessity), and derives an exponential bandwidth annealing schedule via Fourier-space Landau-damping analysis.
> "Generative Modeling via Drifting has recently achieved state-of-the-art one-step image generation ... yet the success is largely empirical and its theoretical foundations remain poorly understood."

### Drifting Fields are not Conservative — `cit:2` `yr:2026` `arxiv:2604.06333`
**Authors**: Leonard Franz, Sebastian Hoffmann, Tim Weiland
**Digest**: Proves the Drifting Models drift field is non-conservative for general radial kernels (Gaussian is the unique exception) and introduces a "sharp kernel" normalization that makes the field a gradient of a scalar potential, recovering exact equilibrium identifiability while preserving generation quality.
> "Drifting models have recently gained attention ... We ask whether this procedure is equivalent to optimizing a scalar loss and find that, in general, it is not: drift fields are not conservative"

### Identifiability and Stability of Generative Drifting with Companion-Elliptic Kernel Families — `cit:1` `yr:2026` `arxiv:2604.24196`
**Authors**: Hakbong Lee
**Digest**: Defines a "companion-elliptic" kernel class (Gaussian + Matérn with ν≥1/2) for which the drifting field vanishes iff the two measures are equal, and characterizes the precise failure mode (mass escape along a one-dimensional ray) where field-norm control alone fails to give weak convergence.
> "This paper analyzes identifiability and stability for the drifting field underlying distributional matching in the Generative Drifting framework of Deng et al."

### Attraction, Repulsion, and Friction: DMF, a Friction-Augmented Drifting Model — `cit:0` `yr:2026` `arxiv:2604.18194`
**Authors**: A. Kazanskii, Tatiana Petrova, Konstantin Bagrianskii
**Digest**: Closes two open questions in Deng et al.: derives a contraction threshold and proves identifiability of the drift-field equilibrium under a Gaussian kernel, then introduces a friction-augmented variant (DMF) that matches/exceeds Optimal Flow Matching at 16× lower compute.
> "Drifting Models [Deng et al., 2026] ... The original analysis leaves two questions open ... we prove that the drift-field equilibrium is identifiable ... closing the converse of Proposition 3.1 of Deng et al."

### On the Wasserstein Gradient Flow Interpretation of Drifting Models — `cit:0` `yr:2026` `arxiv:2605.05118`
**Authors**: Arthur Gretton, W. Li, Alexandre Galashov
**Digest**: Analyzes Deng et al.'s Generative Modeling via Drifting through the lens of Wasserstein Gradient Flows, showing the proposed algorithm corresponds to a KL-WGF fixed point with Parzen smoothing while the implemented algorithm resembles a Sinkhorn-divergence WGF, and extends the construction to MMD, sliced Wasserstein, and GAN-critic WGFs.
> "Recently, Deng et al. (2026) proposed Generative Modeling via Drifting (GMD) ... This note presents an analysis of GMD through the lens of Wasserstein Gradient Flows"

### Learning Monge maps with constrained drifting models — `cit:0` `yr:2026` `arxiv:2603.25182`
**Authors**: Théo Dumont, Théo Lacombe, François-Xavier Vialard
**Digest**: Proposes a constrained gradient flow on the space of transport maps that converges to the optimal-transport (Monge) map, and connects its parameter-space natural-gradient interpretation directly to drifting generative models. Provides existence/convergence proofs and convexity-constrained NN discretizations.
> "this is equivalent to performing a natural gradient descent of the lift of the chosen divergence in the neural networks' parameter space, similarly to drifting generative models"

## Improvements

Papers claiming to beat Drifting Models on its own benchmarks.

### Representation Fréchet Loss for Visual Generation — `cit:2` `yr:2026` `arxiv:2604.28190`
**Authors**: Jiawei Yang, Zhengyang Geng, Xuan Ju
**Digest**: Proposes FD-loss, a directly optimized Fréchet Distance objective in representation space, that achieves **0.72 FID** on ImageNet-256 for a one-step generator — beating Drifting Models' 1.54 FID on the same benchmark.
> "Under the Inception feature space, a one-step generator achieves 0.72 FID on ImageNet 256x256."

### Analytical Correction for Subsampling Bias in Drifting Models (ABC) — `cit:0` `yr:2026` `arxiv:2604.27239`
**Authors**: Jiaru Zhang, Zeyun Deng, Juanwu Lu
**Digest**: Identifies an O(1/n) softmax self-normalization bias in the minibatch centroid estimator of Drifting Models and proposes Analytical Bias Correction, a closed-form plug-in that reduces the bias to O(1/n²), improves CIFAR-10 FID, and trains faster than the vanilla drifting baseline.
> "we begin by showing that the minibatch centroid is in general a biased estimator of the target centroid, with a pointwise O(1/n) bias arising from softmax self-normalization"

### Lookahead Drifting Model — `cit:0` `yr:2026` `arxiv:2605.04060`
**Authors**: Guoqiang Zhang, Kenta Niwa, W. B. Kleijn
**Digest**: Proposes a lookahead variant of the drifting model that computes a sequence of drifting terms per iteration to capture higher-order gradient information, optimizing the model via their weighted summation; reports better CIFAR-10 performance than the baseline.
> "Recently, a new paradigm named *drifting model* has been proposed ... Experimental results on toy examples and CIFAR10 demonstrate the better performance of the new method than the baseline."

## Applications

Domain transfer with no methodological change.

### MRI-to-CT synthesis using drifting models — `cit:1` `yr:2026` `arxiv:2603.28498`
**Authors**: Qing Lyu, Jianxu Wang, Jeremy Patton Hudson
**Digest**: Uses the Drifting Models paradigm as-is to synthesize pelvic CT from MRI, benchmarking against CNN, GAN, PPFM, and diffusion baselines and finding it yields high SSIM/PSNR with one-step millisecond inference.
> "we investigate recently proposed drifting models for synthesizing pelvis CT images from MRI"

## Benchmarks (cited as baseline only)

- **ELT: Elastic Looped Transformers for Visual Generation** — `cit:2` `arxiv:2604.09168` — Class-conditional ImageNet/UCF-101 generation via weight-shared looped transformers + intra-loop self-distillation; cites Drifting only as a one-step baseline.

## Mentions (passing citations, not load-bearing)

- **Consistency Regularised Gradient Flows for Inverse Problems** — `cit:0` `arxiv:2605.07907` — Euclidean-Wasserstein-2 gradient-flow framework for inverse problems with LDMs; abstract doesn't mention Drifting Models.
- **V-Co: A Closer Look at Visual Representation Alignment via Co-Denoising** — `cit:0` `arxiv:2603.16792` — Uses a "perceptual-drifting hybrid loss" as one of four ingredients, but the work is centered on iterative diffusion with REPA-style alignment, not the one-step Drifting paradigm.
- **ABounD: Adversarial Boundary-Driven Few-Shot Learning for Anomaly Detection** — `cit:0` `arxiv:2511.22436` — No methodological or empirical link to Drifting Models; generative modeling is not used.

## What's emerging

The dominant near-term thread is **drifting-as-a-portable-recipe**: take any iterative-refinement generative pipeline (diffusion, score, flow), replace it with a drifting field, get one-step inference. Within ~3 months 13 groups have ported this recipe to speech, medical imaging, robot policy, AR vision, motion planning, wireless decoding, Boltzmann sampling, and RL. The interesting sub-thread is **trajectory/policy generation** (Drift-Based Policy, MISTY, Keyed Drift Fields, Drifting MPC, PODPO — 5/13 extensions) where one-step inference unlocks real-time control and the "evolve distribution during training" framing is a natural fit for offline RL. The dominant longer-term thread is **theoretical consolidation**: three independent groups (Lai et al., Cao et al., Turan & Ovsjanikov) have converged on the same identity — drift = score difference on kernel-smoothed distribution — which means the next 6 months will likely see fewer "we discovered why drifting works" papers and more "we exploit this equivalence to fix X" papers (Sinkhorn-Drifting, DMF, sharp-kernel normalization are early instances). The obvious next open problem is **scaling the FID gap**: Representation Fréchet already beats 1.54 → 0.72 on ImageNet-256 one-step using a different loss; the question is whether drifting + better representation losses combine, or whether one paradigm absorbs the other.

## Verification log

- **Selected**: 29 of 29 follow-ups (no abstracts were missing — all 29 had non-empty abstracts in the Semantic Scholar response)
- **Tagged**: 29 of 29 (4 parallel subagents, 8+8+8+5 papers per batch)
- **Tag distribution**: Extends 13, Theoretical 8, Improves 3, Applies 1, Benchmarks 1, Mentions 3, Criticizes 0 → sums to 29 ✓
- **No fabrications**: every arxiv id in this report appears in `/tmp/followup/deng-drifting-2026/selected.json` (verified by id-cross-check)
- **Unprocessed**: 0
- **Seed `paperId`**: `da71d49479a34fa6f6e317cc477a9f8d8bb9f664`
- **Rate-limit incident**: the unauthenticated Semantic Scholar endpoint 429'd on the seed fetch even after a 30s pause; required 90s to clear. Documented in the skill as the measured floor.
