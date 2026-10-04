# Paper Outline — IEEE Format

**Title:** Are Flood Risk Classifiers Measuring What They Are Trained On? Label Provenance in Vision-Transformer Flood Mapping

**Venue:** IEEE Access — single column, 6–8 pages. A CVPR/ICCV submission requires a novel *algorithm*; this contribution is methodological, so IEEE Access (or an applications-track conference such as ICASSP/ICIP) is the realistic target.

**Structure compliance:** see `docs/ieee_compliance.md` for the full IEEE Author Center requirements.

---

## Section order (mandatory per IEEE)

```
Title → Authors → Abstract → Keywords → First footnote
→ I. Introduction → II. Methodology → III. Results
→ IV. Discussion → V. Conclusion → References → Acknowledgments
```

There is **no standalone Related Work section.** IEEE folds the literature review into the Introduction.

---

## Abstract

Constraints: single paragraph, 250 words max, no abbreviations, no citations, no equations, no mathematical symbols.

Draft in this order once results exist:

1. **Context** — vision transformers are widely applied to flood mapping from satellite imagery, with reported accuracies above 90%
2. **Problem** — those labels are of unverified provenance; we show a common pipeline shortcut manufactures five-class severity labels from a binary dataset by random re-splitting
3. **Consequence** — reported gains measure the artifact rather than flood severity
4. **Method** — we formalise label-provenance auditing for flood-mapping pipelines, implement a 34-check test suite detecting four defect classes, and run a protocol-controlled comparison of a vision transformer against a convolutional network
5. **Result** — fill with measured numbers
6. **Contribution** — reproducible audit tooling and an honest protocol for the field

---

## I. Introduction

IEEE requires the introduction to include the literature review, position the work in the field, show novelty, and state the research question and its importance.

### Paragraph structure

| ¶ | Content |
|---|------|
| 1 | Stakes — flood detection supports disaster response; field has moved from manual interpretation to automated deep learning |
| 2 | The promise — transformers adopted on the argument that global context matters for hydrology; cite papers claiming 90%+ |
| 3 | **The problem (thesis)** — accuracy figures are only as good as label provenance; the standard benchmark dataset is binary and carries no severity ground truth |
| 4 | The failure mode — describe the re-split mechanism precisely, with code; show two risk tiers become the same images |
| 5 | Why it went unnoticed — five-class output is operationally attractive, so people synthesise tiers rather than abandon them |
| 6 | Contribution list, numbered (1)–(4) |

### Literature review (subsections A–D, folded in per IEEE)

- **A. Flood detection from optical imagery** — post-event UAV and satellite datasets; U-Net and ResNet baselines; binary framing predominates
- **B. Vision transformers for remote sensing** — ViT, Swin, and their flood/disaster applications; the "global context matters for hydrology" argument originated here and is largely untested
- **C. Label quality and dataset shortcuts** — Geirhos et al. on shortcut learning; spurious-correlation analysis; dataset documentation literature. **Key cluster — the paper's novelty is anchored here.**
- **D. Explainability in remote sensing** — Grad-CAM and attention rollout as applied to floods; maps are reported but rarely quantified

### Contributions

1. Formalisation of **label provenance auditing** as a prerequisite for flood-mapping evaluation, and a taxonomy of four defect classes: fabricated severity tiers, silent class-dropping, unseeded runs, and ignored pipeline parameters
2. `test_label_integrity.py` — 18 executable checks detecting fabricated tiers and verifying class/label consistency; plus `test_objectives.py` (16 checks) covering reproducibility and loader contracts
3. A protocol-controlled comparison of a vision transformer and a convolutional network on identical splits under verified labels and fixed seeding
4. A pointing-game metric measuring whether model attention actually localises flood regions, addressing the field's reliance on unquantified visual attention maps

---

## II. Methodology

### II-A The re-split mechanism

Show the six lines of the original `organize_raw()`. Derive formally: with *n* negatives, two risk tiers are disjoint subsets of the same class, so the class-conditional distribution given an image is not identifiable. Any accuracy is attributable to the random seed rather than the scene.

### II-B Datasets

- **FloodNet** (BinaLab v1.0) — UAV, post-Hurricane Harvey, binary
- **Synthetic demo set** — regression tests only; explicitly not evidence of flood performance
- State which runs use which. Note SEN12-FLOOD if unavailable.

### II-C Architectures

| | ViT-B/16 | EfficientNet-B3 |
|---|---|---|
| Source | timm, ImageNet-21k | ImageNet |
| Input | 224px, 16×16 patches, 196 tokens | 224px |
| Attention heads / width | 12 / 768 | — |
| Frozen | all but last 4 blocks | all but last 2 stages |
| **Trainable** | **28.7M** of 86.1M | **9.3M** of 11.5M |

Report trainable versus total. Earlier documentation cited total parameters only, obscuring that roughly a third of the transformer is optimised.

### II-D Controlled protocol

Identical across arms: splits, augmentations, AdamW at 1e-4 with weight decay 0.01, cosine schedule with 5-epoch warmup, weighted cross-entropy, early stopping patience 8, seed 42, deterministic cuDNN. Only architecture varies.

### II-E Metrics

Accuracy, macro-F1, per-class precision/recall/F1, IoU, confusion matrices. **Justify macro-F1 over accuracy** — imbalanced flood data rewards majority-class collapse.

### II-F Pointing-game metric

Define: binarise the attention or Grad-CAM map, define the ground-truth flood mask, compute the fraction of top-attended mass falling inside it. State how masks are derived and any limitations.

---

## III. Results — BLOCKED

**No numbers exist yet.** The CNN run is in progress; the transformer run has not started; available imagery is synthetic. Do not populate this section with previously documented figures — those trace to the fabricated-label pipeline.

Planned subsections:

- **III-A** Effect of label fabrication (fabricated vs verified, same architecture)
- **III-B** Architecture comparison under identical protocol
- **III-C** Per-class behaviour and confusion matrices
- **III-D** Attention localisation (pointing game)
- **III-E** Seed variance across ≥3 runs per architecture

---

## IV. Discussion

- **IV-A** Fabricated labels invert conclusions — a plausible-looking gain that disappears under audit
- **IV-B** Attention maps are not evidence; without pointing game, interpretability claims in this literature are unsupported
- **IV-C** Five-tier operational output needs real risk labels (DEM, slope, drainage density, SAR backscatter). Propose the pipeline; state it is not implemented
- **IV-D** Threats to validity — single dataset, single event, UAV rather than satellite, CPU-only training limits epochs, synthetic data used for regression tests only

IEEE guidance: describe what the results mean and how they contribute to the field.

---

## V. Conclusion

Short. Restate that label provenance must be audited before flood-mapping metrics are reported; tooling and an honest comparison are provided; genuine multi-tier risk requires terrain-derived labels.

IEEE guidance: *"Be careful not to inflate your findings."*

---

## References

IEEE numeric style, order of first citation. Format via the IEEE Reference Preparation Assistant. Target 25–35.

Ethics constraint (verbatim): *"Be sure to only cite references that directly support your work. Inflating citations by adding unnecessary references is considered a breach of publishing ethics."*

Key clusters: FloodNet / BinaLab; SEN12-FLOOD; ViT (Dosovitskiy et al.); Swin (Liu et al.); EfficientNet (Tan and Leung); SegFormer; Geirhos et al. on shortcut learning; Northcutt on spurious correlations; Grad-CAM (Selvaraju et al.); attention rollout (Abnar and Zuidema).

---

## Figures and tables

Graphics: 300 dpi halftone, 600 dpi line art, RGB, vector preferred for diagrams, 8–10 pt fonts, captions below figures and above tables.

| # | Content |
|---|------|
| Fig. 1 | Both architectures side by side |
| Fig. 2 | Label fabrication mechanism |
| Fig. 3 | Confusion matrices under verified labels |
| Fig. 4 | Grad-CAM and attention rollout with ground-truth mask overlaid |
| Fig. 5 | Training curves with seed variance band |
| Fig. 6 | Pipeline architecture with audit checkpoint marked |
| Table I | Dataset comparison |
| Table II | Trainable parameters and inference cost |
| Table III | Main results — accuracy, macro-F1, IoU, per-class F1 |
| Table IV | Fabricated vs verified labels, identical model |
| Table V | Defect taxonomy and detection |
| Table VI | Pointing-game scores |

---

## Blockers before submission

| Blocker | Detail |
|---|---|
| Affiliation | department, university, city, state, postal code, country, email |
| Prior-publication statement | is this part of an existing thesis or project report? |
| Funding statement | required in the first footnote regardless |
| Results section | no measurements exist |
| Real imagery | synthetic data cannot support a flood-mapping claim |
| Seed variance | ≥3 seeds per architecture |
| Ground-truth masks | required for the localisation metric |