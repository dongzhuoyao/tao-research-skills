"""Fixture: paper_content.json for arxiv 2210.06462 (Self-Guided Diffusion Models).

Creates a structured paper with verifiable facts in each section.
Tests query these facts to verify the skill retrieves correct content.
"""
from __future__ import annotations

import json
from pathlib import Path

PAPER_ID = "2210.06462"

METADATA = {
    "title": "Self-Guided Diffusion Models",
    "authors": [
        "Vincent Tao Hu",
        "David W Zhang",
        "Yuki M. Asano",
        "Gertjan J. Burghouts",
        "Cees G. M. Snoek",
    ],
    "abstract": (
        "Diffusion models have recently enabled tremendous advancements in many "
        "computer vision fields related to image synthesis. Current diffusion models "
        "mostly rely on carefully annotated data for training. In this paper, we "
        "eliminate the need for any annotation by instead leveraging the "
        "self-supervised signal from a feature extraction function and a "
        "self-annotation function at multiple granularities: from holistic images to "
        "object boxes and segmentation masks. Our experiments demonstrate that "
        "self-labeled guidance always outperforms diffusion models without guidance "
        "and may even surpass guidance based on ground-truth labels, especially for "
        "imbalanced datasets."
    ),
    "year": 2022,
    "venue": "CVPR 2023",
}

SECTIONS = [
    {
        "name": "Introduction",
        "level": 1,
        "text": (
            "Diffusion models have recently enabled tremendous advancements in many "
            "computer vision fields related to image synthesis, but counterintuitively "
            "this often comes with the cost of requiring large annotated datasets. "
            "For instance, classifier-free guidance \\citep{ho2022classifierfree} "
            "requires image-annotation pairs, where the annotation can range from a "
            "single class label to dense segmentation masks. This dependency on "
            "human-generated annotations limits the applicability of guided diffusion "
            "models to domains where annotation is expensive or impractical.\n\n"
            "In this paper, we propose to eliminate the annotation requirement by "
            "leveraging self-supervision signals. We introduce a framework that uses "
            "a feature extraction function $g$ and a self-annotation function $f$ "
            "to generate guidance signals at multiple granularities. Our approach "
            "operates at three levels: (1) self-labeled guidance using cluster "
            "assignments from self-supervised features, (2) self-boxed guidance using "
            "unsupervised object localization, and (3) self-segmented guidance using "
            "unsupervised semantic segmentation.\n\n"
            "Our key finding is that self-labeled guidance always outperforms "
            "unguided diffusion and may even surpass guidance based on ground-truth "
            "labels. On ImageNet32, self-labeled guidance achieves an FID of 7.3 "
            "compared to 9.2 for ground-truth guidance. This surprising result "
            "suggests that self-supervised features capture visual structure more "
            "effectively than categorical labels for guiding diffusion."
        ),
    },
    {
        "name": "Related Work",
        "level": 1,
        "text": (
            "\\subsection{Conditional Generative Models}\n\n"
            "Earlier works on generative adversarial networks (GANs) have observed "
            "improvements in image quality by conditioning on ground-truth labels "
            "\\citep{brock2019biggan, mirza2014conditional}. BigGAN \\citep{brock2019biggan} "
            "demonstrated that class-conditional generation with truncation can produce "
            "high-fidelity images. In the diffusion model literature, classifier "
            "guidance \\citep{dhariwal2021diffusion} uses a separate classifier to "
            "steer the sampling process, while classifier-free guidance "
            "\\citep{ho2022classifierfree} eliminates the need for an external "
            "classifier but still requires class labels during training.\n\n"
            "\\subsection{Self-Supervised Learning in Generative Models}\n\n"
            "Self-supervised learning has achieved remarkable success in "
            "representation learning. DINO \\citep{caron2021dino} trains vision "
            "transformers with self-distillation and produces features that "
            "naturally contain semantic segmentation information. "
            "We focus on translating the benefits of pretrained self-supervised "
            "visual encoders to the generative domain, using DINO ViT-B/16 as "
            "our primary feature extractor."
        ),
    },
    {
        "name": "Approach",
        "level": 1,
        "text": (
            "\\subsection{Background}\n\n"
            "A diffusion model defines a forward process that gradually adds "
            "Gaussian noise to data $x_0 \\sim q(x)$ over $T$ timesteps. Learning "
            "the reverse process reduces to learning a denoiser "
            "$\\epsilon_\\theta(x_t, t)$ that predicts the noise added at step $t$. "
            "The training objective is:\n\n"
            "$\\mathcal{L} = \\mathbb{E}_{t, x_0, \\epsilon}[\\|\\epsilon - "
            "\\epsilon_\\theta(x_t, t)\\|^2]$\n\n"
            "Classifier-free guidance empowers the model with progressive control "
            "over the degree of alignment between the guidance signal and the "
            "sample by varying the guidance strength $w$. The guided score estimate "
            "becomes:\n\n"
            "$\\tilde{\\epsilon}_\\theta(x_t, t, c) = (1+w)\\epsilon_\\theta(x_t, t, c) "
            "- w\\epsilon_\\theta(x_t, t)$\n\n"
            "\\subsection{Self-Guided Diffusion Models}\n\n"
            "We replace the supervised annotation function $\\xi$ with self-supervised "
            "alternatives. Our framework uses $g$ (feature extraction function) and "
            "$f$ (self-annotation function) to produce guidance signals.\n\n"
            "\\textbf{Self-labeled guidance.} We use k-means as our self-annotation "
            "function $f$ on extracted DINO features, creating a one-hot embedding "
            "$k \\in \\mathbb{R}^K$ for each image. We explore $K \\in \\{100, 400, "
            "1000, 5000\\}$ clusters. At 5,000 clusters, self-labeled guidance "
            "outperforms the model using ground-truth labels, with an FID of 16.4 "
            "versus 17.9 on ImageNet64.\n\n"
            "\\textbf{Self-boxed guidance.} We use LOST \\citep{simeoni2021lost} "
            "for unsupervised object localization. Each bounding box is represented "
            "as a binary mask $k_s \\in \\mathbb{R}^{W \\times H}$ where 1 indicates "
            "that the pixel is inside the box and 0 outside.\n\n"
            "\\textbf{Self-segmented guidance.} We use STEGO \\citep{hamilton2022stego} "
            "to produce unsupervised segmentation masks. The mapping function "
            "$f_\\psi(\\cdot; \\mathcal{D}): \\mathbb{R}^{W \\times H \\times C} "
            "\\rightarrow \\mathbb{R}^{W \\times H \\times K}$ maps features to "
            "segmentation masks with $K$ clusters."
        ),
    },
    {
        "name": "Experiments",
        "level": 1,
        "text": (
            "\\subsection{Self-Labeled Guidance}\n\n"
            "We evaluate self-labeled guidance on ImageNet32, ImageNet64, CIFAR100, "
            "and LSUN-Churches 256x256.\n\n"
            "\\textbf{Feature extractor comparison.} Among self-supervised methods, "
            "DINO ViT-B/16 achieves the best FID performance. We compare DINO, "
            "MAE, MoCo-v3, and CLIP as feature extractors.\n\n"
            "\\textbf{Cluster analysis.} At 5,000 clusters, self-labeled guidance "
            "outperforms the model using ground-truth labels on ImageNet64, "
            "with an FID of 16.4 versus 17.9. On CIFAR100, self-labeled guidance "
            "with 400 clusters outperforms the baseline trained with ground-truth "
            "labels.\n\n"
            "\\textbf{ImageNet results.} Self-labeled achieves FID of 7.3 on "
            "ImageNet32 versus 9.2 for ground-truth guidance. On ImageNet64, "
            "the best self-labeled result is FID 16.4.\n\n"
            "\\textbf{High-resolution.} On 256x256 images, self-labeled guidance "
            "reaches FID of 16.1 on LSUN-Churches versus 19.2 without guidance.\n\n"
            "\\subsection{Self-Boxed Guidance}\n\n"
            "We evaluate on Pascal VOC and COCO\\_20K using LOST for box generation. "
            "Pascal VOC: self-boxed achieves FID of 18.4 versus 23.5 with "
            "ground-truth labels. COCO\\_20K: FID of 16.0 versus 19.3 with "
            "ground-truth labels. The model generates visually diverse yet "
            "semantically consistent images without box annotations.\n\n"
            "\\subsection{Self-Segmented Guidance}\n\n"
            "We evaluate on Pascal VOC and COCO-Stuff using STEGO for segmentation. "
            "Pascal VOC: self-segmented FID of 17.1 versus 23.5 with ground-truth "
            "labels. COCO-Stuff: train FID of 12.5 versus 23.5 with labels. "
            "This narrows the performance gap with ground-truth mask guidance "
            "while requiring no pixel-level annotations.\n\n"
            "\\subsection{Ablation Studies}\n\n"
            "We ablate the guidance strength $w$ and number of clusters $K$. "
            "Increasing $w$ from 0 to 4 steadily improves FID until $w=3$, after "
            "which quality saturates. For cluster count, there is a sweet spot: "
            "too few clusters (K=10) underperform, while K=5000 gives the best "
            "results, suggesting fine-grained self-labels are more informative "
            "than coarse categories. Figure \\ref{fig:ablation} shows the full "
            "ablation curves."
        ),
    },
    {
        "name": "Conclusion",
        "level": 1,
        "text": (
            "We have shown that self-supervision signals are an adequate replacement "
            "for existing guidance methods that generate images by relying on "
            "annotated image-label pairs during training. Our self-guided diffusion "
            "framework operates at three granularities: labels, boxes, and "
            "segmentation masks, all derived from self-supervised features without "
            "any human annotation. The key insight is that self-supervised features "
            "from DINO capture visual structure more effectively than categorical "
            "labels, especially when the number of self-labels exceeds the number "
            "of ground-truth categories. Future work will investigate the efficacy "
            "of our self-guidance approach on feature extractors trained on larger "
            "datasets and extending to text-guided generation."
        ),
    },
    {
        "name": "Acknowledgments",
        "level": 1,
        "text": (
            "This work is financially supported by the Allowance for Top Consortia "
            "for Knowledge and Innovation (TKIs) from the Netherlands Ministry of "
            "Economic Affairs and Climate Policy. Yuki M. Asano is supported by "
            "an Amsterdam Data Science Start-Up Grant."
        ),
    },
    {
        "name": "Appendix A: Implementation Details",
        "level": 1,
        "text": (
            "For all experiments, we use the ADM (Ablated Diffusion Model) "
            "architecture as our backbone. ImageNet32 and ImageNet64 models use "
            "256 channels with attention at resolutions 16, 8, and 4. Training "
            "uses a batch size of 128 and a learning rate of 1e-4 with linear "
            "warmup over 5000 steps. We use 4000 diffusion timesteps during "
            "training and DDIM with 250 steps for sampling. All models are "
            "trained on 4 NVIDIA A100 GPUs. The DINO ViT-B/16 feature extractor "
            "outputs 768-dimensional features per patch, which we aggregate via "
            "CLS token pooling for self-labeled guidance and spatial features "
            "for self-boxed and self-segmented guidance."
        ),
    },
]

FIGURES = [
    {"id": "fig:overview", "caption": "Overview of self-guided diffusion framework"},
    {"id": "fig:ablation", "caption": "Ablation on guidance strength and cluster count"},
    {"id": "fig:qualitative", "caption": "Qualitative results across guidance types"},
]

TABLES = [
    {"id": "tab:imagenet", "caption": "Self-labeled guidance results on ImageNet"},
    {"id": "tab:voc", "caption": "Self-boxed and self-segmented results on Pascal VOC"},
]

# ── Verifiable queries ──────────────────────────────────────────
# Each entry: (query_description, target_section, fact_to_find)
# Tests use these to verify retrieval correctness.
QUERIES = [
    # ── Introduction (8 facts) ──────────────────────────────────
    ("FID on ImageNet32 with self-labeled guidance",
     "Introduction", "FID of 7.3"),
    ("FID on ImageNet32 with ground-truth guidance",
     "Introduction", "9.2 for ground-truth"),
    ("Three granularity levels",
     "Introduction", "self-labeled guidance"),
    ("Second granularity level",
     "Introduction", "self-boxed guidance"),
    ("Third granularity level",
     "Introduction", "self-segmented guidance"),
    ("Feature extraction function symbol",
     "Introduction", "feature extraction function g"),
    ("Self-annotation function symbol",
     "Introduction", "self-annotation function f"),
    ("What annotation types are expensive",
     "Introduction", "dense segmentation masks"),

    # ── Related Work (6 facts) ──────────────────────────────────
    ("Which self-supervised feature extractor is primary",
     "Related Work", "DINO ViT-B/16"),
    ("What does DINO naturally contain",
     "Related Work", "semantic segmentation information"),
    ("BigGAN reference",
     "Related Work", "BigGAN"),
    ("What classifier guidance uses",
     "Related Work", "separate classifier"),
    ("DINO training method",
     "Related Work", "self-distillation"),
    ("Classifier-free guidance still requires what",
     "Related Work", "class labels during training"),

    # ── Approach (12 facts) ─────────────────────────────────────
    ("How many clusters for best performance on ImageNet64",
     "Approach", "5,000 clusters"),
    ("FID of self-labeled vs ground-truth on ImageNet64",
     "Approach", "16.4 versus 17.9"),
    ("Tool used for unsupervised object localization",
     "Approach", "LOST"),
    ("Tool used for unsupervised segmentation",
     "Approach", "STEGO"),
    ("Binary mask representation for boxes",
     "Approach", "binary mask"),
    ("Cluster counts explored",
     "Approach", "100, 400"),
    ("Self-annotation function for labels",
     "Approach", "k-means"),
    ("One-hot embedding dimensionality symbol",
     "Approach", "k \\in \\mathbb{R}^K"),
    ("Training loss formulation",
     "Approach", "\\|\\epsilon - \\epsilon_\\theta"),
    ("Guidance strength symbol",
     "Approach", "guidance strength w"),
    ("Noise prediction model notation",
     "Approach", "\\epsilon_\\theta(x_t, t)"),
    ("Segmentation output dimensionality",
     "Approach", "\\mathbb{R}^{W \\times H \\times K}"),

    # ── Experiments (14 facts) ──────────────────────────────────
    ("Self-boxed FID on Pascal VOC",
     "Experiments", "FID of 18.4"),
    ("Ground-truth FID on Pascal VOC",
     "Experiments", "23.5 with ground-truth"),
    ("Self-boxed FID on COCO_20K",
     "Experiments", "FID of 16.0"),
    ("Ground-truth FID on COCO_20K",
     "Experiments", "19.3 with ground-truth"),
    ("Self-segmented FID on Pascal VOC",
     "Experiments", "FID of 17.1"),
    ("COCO-Stuff train FID",
     "Experiments", "FID of 12.5"),
    ("LSUN-Churches FID with self-labeled guidance",
     "Experiments", "FID of 16.1"),
    ("LSUN-Churches FID without guidance",
     "Experiments", "19.2 without guidance"),
    ("Best ImageNet64 self-labeled FID",
     "Experiments", "FID 16.4"),
    ("CIFAR100 cluster count that outperforms GT",
     "Experiments", "400 clusters"),
    ("Optimal guidance strength w",
     "Experiments", "w=3"),
    ("Worst cluster count in ablation",
     "Experiments", "K=10"),
    ("Best cluster count in ablation",
     "Experiments", "K=5000"),
    ("Feature extractors compared",
     "Experiments", "MAE, MoCo-v3, and CLIP"),

    # ── Conclusion (4 facts) ────────────────────────────────────
    ("Key insight about self-supervised features",
     "Conclusion", "visual structure more effectively"),
    ("When self-labels excel over GT",
     "Conclusion", "exceeds the number of ground-truth categories"),
    ("Future work direction",
     "Conclusion", "feature extractors trained on larger datasets"),
    ("Another future direction",
     "Conclusion", "text-guided generation"),

    # ── Acknowledgments (2 facts) ──────────────────────────────
    ("Funding body",
     "Acknowledgments", "Netherlands Ministry of Economic Affairs"),
    ("Yuki Asano funding",
     "Acknowledgments", "Amsterdam Data Science Start-Up Grant"),

    # ── Appendix (8 facts) ─────────────────────────────────────
    ("Number of GPUs used for training",
     "Appendix A: Implementation Details", "4 NVIDIA A100"),
    ("DINO feature dimensionality",
     "Appendix A: Implementation Details", "768-dimensional"),
    ("Diffusion timesteps during training",
     "Appendix A: Implementation Details", "4000 diffusion timesteps"),
    ("DDIM sampling steps",
     "Appendix A: Implementation Details", "250 steps"),
    ("Backbone architecture",
     "Appendix A: Implementation Details", "ADM"),
    ("Batch size",
     "Appendix A: Implementation Details", "batch size of 128"),
    ("Learning rate",
     "Appendix A: Implementation Details", "1e-4"),
    ("Warmup steps",
     "Appendix A: Implementation Details", "5000 steps"),
]


def create_fixture(workspace_dir: Path) -> Path:
    """Write paper_content.json and return its path."""
    paper_dir = workspace_dir / PAPER_ID
    paper_dir.mkdir(parents=True, exist_ok=True)

    paper = {
        "metadata": METADATA,
        "sections": SECTIONS,
        "figures": FIGURES,
        "tables": TABLES,
    }

    path = paper_dir / "paper_content.json"
    path.write_text(json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
