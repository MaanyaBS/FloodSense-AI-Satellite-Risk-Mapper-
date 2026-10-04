# Paper Outline — IEEE Format

**Working title:** *Are Flood Risk Classifiers Measuring What We Think? Label Integrity in Vision-Transformer Flood Mapping*

**Venue recommendation:** IEEE Access — 6–8 pages, single column, suits a systems/benchmark contribution. A CVPR/ICCV submission would need a genuinely novel *algorithm*; this paper's contribution is methodological, so IEEE Access (or an IEEE conference in the applications/signal-processing track, e.g. ICASSP/ICIP) is the realistic target.

---

## Abstract (150–250 words)

Structure, in order:

1. **Context** — Vision transformers now widely applied to flood mapping from satellite imagery, with reported accuracies above 90%
2. **Problem** — but the underlying labels are of unverified provenance; we show a common pipeline shortcut manufactures five-class severity labels from a binary dataset by random re-splitting
3. **Consequence** — reported gains measure the artifact, not flood severity
4. **Method** — we formalise label-provenance auditing for flood-mapping pipelines, implement a 34-check test suite that detects four defect classes, and run a protocol-controlled ViT-B/16 vs EfficientNet-B3 comparison
5. **Result** — (fill with measured numbers; the fabricated-label pipeline scores X, verified binary scores Y)
6. **Contribution** — reproducible audit tooling plus honest protocol for the field

---

## I. Introduction

- **Para 1 — the stakes.** Flood detection from satellite imagery supports disaster response. Cite the field's shift from manual interpretation to automated DL.
- **Para 2 — the promise.** ViTs adopted on the argument that global context matters for flood mapping. Cite papers claiming 90%+ accuracy. This framing sets up the critique; don't strawman.
- **Para 3 — the problem (thesis).** Accuracy figures are only as good as label provenance. FloodNet, the field's standard benchmark, is binary: Flooded / Non-Flooded. It carries **no severity ground truth**.
- **Para 4 — the failure mode.** Describe the re-split mechanism precisely (with the actual code). Show `Low Risk` and `Non-Flooded` become the same images. This is a dataset-shortcut / spurious-correlation pathology.
- **Para 5 — why it went unnoticed.** Five-class schemes look more useful operationally; the intermediate tiers are exactly the ones an application wants, so people synthesise them rather than abandon them.
- **Para 6 — contributions.** Bullet list, numbered (1)–(4), matching the paper's sections.

### Contributions

1. Formalisation of **label provenance auditing** as a prerequisite step for flood-mapping evaluation, and a taxonomy of four defect classes (fabricated tiers, silent class-dropping, unseeded runs, ignored data-pipeline parameters)
2. `test_label_integrity.py` — 18 executable checks that detect fabricated tiers and verify class/label consistency, plus `test_objectives.py` (16 checks) covering reproducibility and loader contracts
3. A protocol-controlled comparison of ViT-B/16 and EfficientNet-B3 on identical splits under verified labels and fixed seeding
4. A pointing-game metric for measuring whether ViT attention actually localises flood regions, addressing the field's reliance on unquantified visual attention maps

---

## II. Related Work

Subsections, each 3–5 paragraphs:

- **II-A Flood detection from optical imagery** — post-event UAV and satellite datasets; U-Net and ResNet baselines; binary framing predominates
- **II-B Vision transformers for remote sensing** — ViT, Swin, and their flood/disaster applications; claim the "global context matters for hydrology" argument originated here and is largely untested
- **II-C Label quality and dataset shortcuts** — Geirhos et al. on shortcut learning; Northcutt's spurious correlation analysis; the "dataset documentation" literature. **This is your key citation cluster.**
- **II-D Explainability in remote sensing** — Grad-CAM and attention-rollout as used in flood work; note the field reports maps but rarely quantifies them

---

## III. The Label Provenance Problem

The technical core. Keep code minimal; reference the repo.

### III-A The re-split mechanism
Show the six lines of the original `organize_raw()`. Derive formally: with `n` negatives, `Low Risk` and `Non-Flooded` are disjoint subsets of the same class, so `p(class | image)` is not identifiable — the label carries no information the model can use. Any accuracy is attributable to the seed, not the scene.

### III-B Symptom: inflated metrics
Report your own measured contrast — fabricated vs verified pipeline, same architecture, same splits.

### III-C Four defect classes

| Defect | Mechanism | Consequence |
|---|---|---|
| Fabricated severity tiers | random re-split of binary labels | metrics measure seed, not severity |
| Silent class-dropping | `if not dir.exists(): continue` | trains 1-class, reports ~100% |
| Unseeded runs | `seed: 42` declared, never read | irreproducible, invalid comparison |
| Ignored pipeline params | `num_workers` hardcoded to 0 | declared protocol ≠ executed protocol |

### III-D Detection
Describe the test suite; note all defects were found in an existing codebase without access to its provenance.

---

## IV. Methodology

### IV-A Datasets
- **FloodNet** (BinaLab v1.0) — UAV, post-Hurricane Harvey, binary
- **Synthetic demo set** — for the CI-level regression tests only; explicitly *not* evidence of flood performance
- State clearly which runs use which. If SEN12-FLOOD is unavailable, say so.

### IV-B Architectures
- **ViT-B/16** (`timm`, ImageNet-21k), 224px, 16×16 patches, 196 tokens, 12 heads, 768 dim, last 4 blocks unfrozen, 28.7M trainable
- **EfficientNet-B3** (ImageNet), last 2 stages unfrozen, 9.3M trainable of 11.5M
- Report trainable vs total; the earlier README's "86M" obscured the 28.7M actually optimised

### IV-C Controlled protocol
Identical: splits, augmentations, optimiser (AdamW, lr 1e-4, wd 0.01), cosine schedule + 5-epoch warmup, weighted cross-entropy, early stopping patience 8, seed 42, `cudnn.deterministic=True`. Only architecture varies.

### IV-D Metrics
Accuracy, macro-F1, per-class precision/recall/F1, IoU, confusion matrices. **Justify macro-F1 over accuracy** — imbalanced flood data rewards majority-class collapse.

### IV-E Pointing-game metric
Define: for each image, binarise the attention/Grad-CAM map, define the ground-truth flood mask, compute the fraction of the top-attended mass falling inside it. State how you derive masks for FloodNet (pre/post-harvey registration) and any limitations.

---

## V. Results

> **Blocked until training completes.** Structure below; every number must be measured.

### V-A Effect of label fabrication
Table: identical model, fabricated vs verified labels. Expect a large gap.

### V-B ViT vs EfficientNet under identical protocol
Table with trainable params, epoch time, accuracy, macro-F1, IoU, per-class F1.

### V-C Per-class behaviour
Confusion matrices. Report which tiers fail and why — imbalanced minor tiers typically collapse into neighbours.

### V-D Attention localisation
Pointing-game scores per class. A ViT with high accuracy but low pointing game has learned texture priors, not hydrology — this is the interpretability result.

### V-E Reproducibility
Seed-variance across 3 runs (mean ± std). A protocol claiming control must quantify residual variance.

---

## VI. Discussion

- **VI-A** Fabricated labels invert conclusions — a plausible-looking gain that disappears
- **VI-B** Attention maps ≠ evidence; without pointing game, XAI claims in this literature are unsupported
- **VI-C** Five-tier operational output needs real risk labels (DEM, slope, drainage density, SAR backscatter). Propose the pipeline, note you have not implemented it
- **VI-D** Threats to validity: single dataset, single event, UAV-not-satellite, CPU-only training limits epochs, synthetic data used for regression tests only

---

## VII. Conclusion

Short. Restate: label provenance must be audited before flood-mapping metrics are reported; we provide tooling and an honest comparison; multi-tier risk needs real terrain labels.

---

## References

Target 25–35. Key clusters: FloodNet / BinaLab, SEN12-FLOOD, ViT (Dosovitskiy), Swin (Liu), EfficientNet (Tan), SegFormer, Geirhos *Shortcut Learning*, Northcutt, Grad-CAM (Selvaraju), attention rollout (Abnar & Zuidema), IEEE Access format guide.

---

## Figures and Tables

| # | Content |
|---|---|
| Fig. 1 | ViT vs CNN architecture, side by side |
| Fig. 2 | Label fabrication mechanism, diagram |
| Fig. 3 | Confusion matrices, ViT and CNN, verified labels |
| Fig. 4 | Grad-CAM and attention rollout with ground-truth flood mask overlaid |
| Fig. 5 | Training curves with seed variance band |
| Fig. 6 | Pipeline architecture with audit checkpoint marked |
| Table I | Dataset comparison — FloodNet vs SEN12-FLOOD vs synthetic |
| Table II | Trainable parameters and inference cost |
| Table III | Main results — accuracy, macro-F1, IoU, per-class F1 |
| Table IV | Fabricated vs verified labels, identical model |
| Table V | Defect taxonomy and detection |
| Table VI | Pointing-game scores |

---

## What you need before this is submittable

1. **Completed training runs** — CNN 8 epochs running now; ViT to follow
2. **At least 3 seeds per architecture** for variance
3. **FloodNet or SEN12-FLOOD** for real imagery. Demo data cannot support a flood paper.
4. **Ground-truth masks** for the pointing game
5. **Venue confirmation** — changes page budget and section weighting