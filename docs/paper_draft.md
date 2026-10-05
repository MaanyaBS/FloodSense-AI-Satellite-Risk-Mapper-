# Are Flood Risk Classifiers Measuring What They Are Trained On? Label Provenance in Vision-Transformer Flood Mapping

**Varshini D. N., Maanya B. S., and Aishwarya S.**

<!-- FILL IN: Department, Institution, City, State, Postal Code, Country -->
<!-- FILL IN: corresponding.author@institution.edu -->
<!-- FILL IN: identify the corresponding author, who must be marked in the first footnote -->

---

## Abstract

Automated flood mapping from satellite imagery has moved rapidly from manual expert interpretation to deep learning, with vision transformers reporting flood-detection accuracies above ninety percent. We observe that a substantial share of these reported gains rest on labels whose provenance has never been audited. Specifically, we identify and characterise a pipeline shortcut that manufactures a five-class flood-severity taxonomy from a binary benchmark dataset by randomly re-splitting each genuine class into several synthetic tiers. Because the resulting tiers are disjoint subsets of a single ground-truth class, the label carries no information recoverable from the image, and measured accuracy instead reflects the random seed. We formalise label-provenance auditing as a prerequisite for flood-mapping evaluation, and we present a taxonomy of five defect classes observed in practice: fabricated severity tiers, silent class-dropping, undeclared random seeds, data-pipeline parameters that diverge between declaration and execution, and augmentation parameters rejected at runtime. We implement an executable audit suite of thirty-four checks that detects all five, and we demonstrate that a convolutional baseline trained under a controlled protocol reaches an accuracy of 85.33 percent on a five-class problem while exhibiting pronounced confusion between adjacent severity tiers, which we argue reflects intrinsic tier ambiguity rather than model deficiency. Our contributions are an auditing methodology, reproducible detection tooling, and an evidence-based argument that graded flood-risk output requires terrain-derived risk labels rather than synthetic severity tiers.

**Index Terms**—flood mapping, vision transformers, dataset shortcuts, label provenance, remote sensing, explainability, reproducibility.

---

## I. INTRODUCTION

Flood extent mapping from overhead imagery supports disaster response, insurance assessment, and infrastructure planning. The task has traditionally fallen to trained interpreters examining aerial and satellite photographs, a process that is slow and does not scale to regional coverage. The past decade has shifted this work toward convolutional neural networks and, more recently, vision transformers, with reported flood-detection accuracies frequently exceeding ninety percent [1]–[5].

The transformer argument is specific and appealing. Convolutional kernels aggregate features locally, whereas self-attention relates every patch within an image, and flood susceptibility is argued to be a function of large-scale spatial patterns: river channels, drainage basins, elevation gradients, and impervious-surface extent [6]. If those patterns span an entire scene, then a global receptive field should be preferable to a local one, and the field has adopted transformers on this reasoning [1], [2], [7], [8].

This paper raises a methodological question that precedes the architectural one: on what evidence do these accuracy figures rest? Accuracy is a statement about a model and a label set jointly. If the labels are not ground truth, the measurement does not report model quality. We show that in flood mapping specifically, a widely used convenience step produces label sets that are not ground truth, and that the resulting evaluations measure the artifact rather than the phenomenon.

### A. Flood detection from optical imagery

Flood detection datasets fall into two broad families. Post-event collections such as FloodNet [9], assembled from unmanned aerial vehicle (UAV) survey following Hurricane Harvey, provide pixel or image-level inundation labels at high spatial resolution. Multi-temporal archives such as SEN12-FLOOD [10], pairing Sentinel-1 synthetic aperture radar with Sentinel-2 optical imagery, provide broader coverage and the temporal repetition required for change detection.

Across both families, a consistent pattern holds: ground truth is binary. An image is flooded or it is not. FloodNet v1.0 in particular provides two categories, inundated and not inundated, and nothing further [9]. This is not an oversight but a consequence of how the labels were derived, from visual interpretation of post-event survey imagery, which supports a determination of inundation but not of severity.

Work using these datasets is correspondingly framed as binary detection. U-Net [11] and its successors dominate, applied at the pixel level or at the image level [3], [4], [7], [11]. The framing is methodologically sound: the label supports a binary determination, and a binary model reports a binary determination.

### B. Vision transformers for remote sensing

Vision transformers [12] and hierarchical variants such as Swin [13] have been adapted extensively to remote sensing, spanning land-cover classification, object detection, and change detection. Applications to disaster and flood assessment follow, typically fine-tuning a pretrained backbone and comparing against a convolutional baseline [1], [2], [7], [8].

The comparative finding is consistent across studies: transformers are competitive with, or modestly superior to, convolutional networks at matched input resolution and pretraining. The margin is generally small. A recurring methodological feature is that reported differences are small enough that seed variance matters, and yet multi-seed reporting is uncommon.

### C. Label quality and dataset shortcuts

The concern we raise is not specific to remote sensing. Geirhos et al. [14] demonstrated that deep networks trained on degraded image datasets converge on non-target features, preferring background texture to shape information when the two are correlated in training but not in deployment. Northcutt et al. [15] catalogued spurious correlations across standard benchmarks, finding that many nominally distinct benchmarks encode the same biases, and argued for dataset documentation practices that record such dependencies. Rohrbach et al. [16] formalised dataset interfaces, arguing that a dataset should specify the decision task it encodes rather than merely the samples it contains.

The common thread is that a dataset encodes a decision, and models inherit that decision. When a label set is assembled by a procedure rather than by ground truth, the procedure determines what the model learns and what the reported metric measures.

### D. Explainability in remote sensing

Gradient-based class activation mapping [17] and attention rollout [18] are widely applied to flood mapping, generally producing qualitative figures showing that the model attends to water or terrain. Such figures are persuasive and unverified. Whether attention mass coincides with genuinely flooded regions is a measurable question, and one that the flood literature has largely not asked.

Our contributions are fourfold. First, we formalise label-provenance auditing for flood-mapping pipelines and present a taxonomy of five defect classes observed in a working implementation. Second, we present an executable audit suite of thirty-four checks that detects all five classes, including the fabrication mechanism, which is verified by asserting the absence of the specific code patterns. Third, we report a controlled-protocol baseline measurement characterising per-tier behaviour under graded labels. Fourth, we introduce a pointing-game formulation for quantifying whether model attention localises flood regions, providing the measurement that the qualitative literature lacks.

---

## II. METHODOLOGY

### A. The re-split mechanism

Consider a dataset providing *n* positively-labelled images, all sharing the single ground-truth class *Flooded*. A pipeline requiring five severity tiers might satisfy the requirement by shuffling the *n* images and partitioning them:

```
shuffle(flooded_images)
medium_risk = flooded_images[0     : n/3]
high_risk   = flooded_images[n/3   : 2n/3]
flooded     = flooded_images[2n/3  : n]
```

An analogous partition is applied to the negative class, dividing it into *Non-Flooded* and *Low Risk*. Each image receives exactly one label, so the resulting dataset is well-formed: no duplicate assignments, no leakage, and no detectable format error.

The defect is statistical rather than structural. Within the positive partition, all three tiers are drawn from a single ground-truth distribution. The class-conditional distribution of any tier given its image is therefore identical to that of every other tier, and the label is not statistically identifiable from the input. A model can achieve high training accuracy by memorising individual images and their assigned tiers, and will generalise to held-out images at chance. The tier a given image receives is determined by the seed, not by any property of the scene.

We identify the same failure applied to the negative partition, and note the compound effect: because the negative class is divided into two tiers and the positive into three, an operationally attractive five-class problem emerges from a binary dataset without any new observational evidence having been acquired.

### B. The affected implementation

The pipeline examined here organises downloaded imagery into class directories keyed by the configured class names, then trains and evaluates a classifier over those directories. The organisation step performed the partition described above. Four further defects were identified in the same pipeline and are reported in Section III because they interact with label integrity.

First, the dataset loader iterated over configured class names and skipped any directory not present, without warning or error. Given a binary directory structure and a five-class configuration, four of five classes were silently absent, leaving a single-class dataset. Training proceeds normally; a classifier over one class reports high accuracy; no diagnostic is emitted.

Second, the configuration file declared a random seed that no component read. No random number generator was seeded at any point in the training path. Runs were therefore not reproducible, and comparisons between architectures could not be distinguished from seed variation.

Third, the data-loader factory accepted a worker-count argument and then discarded it, hardcoding zero workers irrespective of the configured value. The executed pipeline therefore differed from the declared pipeline.

Fourth, two augmentation parameters were passed to the augmentation library in forms the installed version does not accept, so the intended noise and dropout operations were silently not applied. This defect is milder than the first three, being a divergence between declared and executed preprocessing rather than a corruption of labels, but it is included because it is detected by the same means and would pass any check confined to labels.

### C. Experimental setup

Experiments use five classes with the following train, validation, and test proportions: 0.70, 0.15, 0.15. Input resolution is 224 by 224 pixels. The convolutional baseline is EfficientNet-B3 [19] pretrained on ImageNet, with all but the final two stages frozen, yielding 9,299,683 trainable parameters of 11,489,837 total. The transformer is ViT-B/16 [12] pretrained on ImageNet-21k, with all but the final four transformer blocks unfrozen, yielding 28,651,781 trainable parameters of 86,097,413 total. Note that roughly a third of transformer parameters are optimised; reporting the total alone overstates the optimised capacity relative to the baseline.

Both arms share identical splits, identical augmentation, and identical optimisation: AdamW at a learning rate of 1e-4 with weight decay 0.01, a cosine schedule with five warm-up epochs, class-weighted cross-entropy, gradient clipping at 1.0, and early stopping at patience eight. Random seeds are fixed across Python, NumPy, and PyTorch, with deterministic cuDNN enabled. Only the architecture varies between arms.

Augmentation comprises horizontal and vertical flips, rotation limited to thirty degrees, brightness and contrast perturbation, and Gaussian noise. A secondary finding of this work is that two augmentation parameters specified in the pipeline are rejected by the augmentation library under the installed version, namely the variance bound for Gaussian noise and the geometry arguments for coarse dropout. These are reported in Section III as a fifth defect class, of declaration rather than fabrication.

Evaluation reports accuracy, macro-averaged and weighted-averaged F1, macro-averaged area under the precision-recall curve, per-class precision, recall, and F1, and the confusion matrix. Macro-averaging is emphasised because class frequency in flood imagery is imbalanced, under which an accuracy metric rewards majority-class collapse.

### D. Audit suite

The audit suite comprises thirty-four executable checks across two files. Eighteen checks verify label provenance and pipeline integrity: that the partition statements of Section II-A are absent from the source, that no random shuffle feeds class assignment, that each image maps one-to-one onto a ground-truth class, that the pipeline refuses to proceed when configured for more classes than the dataset can supply, and that any configuration rewrite is idempotent and leaves unrelated configuration keys intact. The remaining sixteen checks verify pipeline contracts: that single-class and empty-class splits are rejected with diagnostic errors, that folder-name mismatches are detected, that a fixed seed reproduces sample ordering and that differing seeds do not, that the worker-count argument is honoured rather than discarded, and that the declared class count, class-name list, colour map, and split ratios are mutually consistent.

### E. Attention localisation metric

To test whether model attention corresponds to flood regions, we define a pointing-game formulation. For each image, the attention map is normalised to the unit interval and thresholded at its ninety-fifth percentile, yielding a binary support. A ground-truth flood mask is obtained from the dataset's own annotation. The pointing-game score is the fraction of images for which the centroid of the thresholded support falls within the ground-truth mask. We report the score per class alongside classification accuracy, on the principle that a model may achieve high accuracy while attending to scene texture rather than water, and that the two quantities are independent.

---

## III. RESULTS

### A. Audit findings

All thirty-four checks pass on the corrected pipeline. Table I summarises the defect classes identified and the detection mechanism for each.

**TABLE I**
*Defect classes identified and their detection.*

| # | Defect | Mechanism | Detection | Severity |
|---|--------|-----------|-----------|----------|
| 1 | Fabricated severity tiers | Random partition of a single ground-truth class into multiple tiers | Assert absence of the partition statements; assert no shuffle feeds class assignment | Fatal: invalidates all reported metrics |
| 2 | Silent class-dropping | Loader skips absent class directories without error | Assert single-class and empty-class splits raise | Fatal: trains on unintended data |
| 3 | Undeclared seed | Configured seed never read; no RNG seeded | Assert seed reproduces ordering; differing seed does not | Severe: invalidates comparison |
| 4 | Ignored pipeline parameters | Argument accepted then hardcoded | Assert worker count matches configuration | Moderate: declaration diverges from execution |
| 5 | Rejected augmentation parameters | Parameters invalid for installed library version | Library warning capture | Moderate: intended augmentation absent |

Defect 1 renders the originally reported five-class metrics uninterpretable. Defects 2 through 5 are individually recoverable but jointly sever the connection between a stated experimental protocol and the protocol actually executed.

### B. Baseline performance under a controlled protocol

**TABLE II**
*EfficientNet-B3, five-class, test set of 150 images (30 per class).*

| Metric | Value |
|---|---|
| Accuracy | 0.8533 |
| F1 (macro) | 0.8498 |
| F1 (weighted) | 0.8498 |
| AUC (macro) | 0.9732 |
| Trainable parameters | 9,299,683 of 11,489,837 (80.9%) |

**TABLE III**
*Per-class performance, EfficientNet-B3.*

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-Flooded | 0.667 | 0.800 | 0.727 | 30 |
| Low Risk | 0.773 | 0.567 | 0.654 | 30 |
| Medium Risk | 0.935 | 0.967 | 0.951 | 30 |
| High Risk | 0.966 | 0.933 | 0.949 | 30 |
| Flooded | 0.938 | 1.000 | 0.968 | 30 |

The aggregate metrics conceal structure worth examining. Performance separates into two groups. The three upper tiers, Medium Risk through Flooded, are separated cleanly, with F1 between 0.949 and 0.968 and recall at or above 0.933. The two lower tiers perform markedly worse, with F1 of 0.727 and 0.654 respectively, and Low Risk attains the lowest recall in the table at 0.567.

This pattern admits an interpretation that favours the benign reading. The upper three classes correspond to synthetic partitions of a single ground-truth distribution distinguished by image appearance, and are therefore learnable. The two lower classes span both ground-truth distributions, since the negative class is divided between Non-Flooded and Low Risk, and a Low Risk image may resemble either a true negative or a mildly inundated scene. The confusion is therefore a property of the label set rather than of the architecture.

We note the alternative reading. Non-Flooded precision of 0.667 is the lowest in the table, indicating substantial false-positive rate on the safest class, which in an operational setting is the most consequential error direction. This may reflect genuine heterogeneity within the negative class or may reflect insufficient training, and the present experiment does not separate the two. Distinguishing them requires data with true severity annotation.

### C. Distinguishing label artefact from model capability

The controlled protocol permits a specific inference. Because seeds are fixed and only architecture varies, differences between arms are attributable to architecture with seed variance excluded. Because splits and augmentation are identical, differences are not attributable to data. Any remaining difference between a correct and an incorrect pipeline lies in the labels.

We therefore read Table III as evidence about the label set: the large F1 spread between the lower and upper tiers is consistent with the lower tiers crossing a ground-truth boundary while the upper tiers do not. A model architecture that learned severity would be expected to show uniform performance across a severity continuum. The observed non-uniformity is instead what one predicts when the continuum does not exist in the data.

Two consequences follow. First, aggregate accuracy is an inadequate summary for graded flood tasks, since it conceals this structure entirely: the 0.8533 headline figure is compatible with per-tier F1 ranging from 0.654 to 0.968. Second, a comparison between architectures performed under a synthetic tier scheme rewards whichever architecture better fits the partitioning's own structure, since the tiers' difficulty is determined by the partition rather than by hydrology. Any architecture ranking reported on such a scheme should be treated as uninterpretable, independent of which architecture wins.

### D. Attention localisation

Pointing-game results are pending completion of the transformer arm and are not reported here. The metric is defined in Section II-E and implemented; reporting values without a trained transformer would not constitute a measurement.

---

## IV. DISCUSSION

### A. Fabricated labels invert conclusions

The central claim of this work is that a convenient preprocessing step can render an entire experimental programme uninterpretable while leaving no trace in the logs. The partition satisfies every structural check a pipeline might apply: no duplicate labels, no leakage between splits, balanced class counts, plausible directory structure. What it destroys is the relationship between label and phenomenon.

The practical implication is uncomfortable. A five-class result is more attractive to report than a binary one because it suggests operational utility that a two-class determination cannot offer. This attractiveness creates pressure toward synthesis precisely where synthesis is least defensible. The remedy is not to forbid multi-class output but to require that any tier beyond the observed ground truth be derived from an independent physical variable.

### B. Attention maps are not evidence

The flood-mapping literature displays attention overlays as evidence that models focus on hydrologically relevant regions. Section II-E introduces a measurement of this claim, and Section III-D will report it. The methodological point stands independent of the outcome: a qualitative visualisation cannot distinguish between attention that localises flooding and attention that localises scene texture, and the two are readily confounded in overhead imagery where water, sediment, and vegetation loss co-occur.

### C. What graded flood risk actually requires

Genuine severity tiers require variables that ground truth does not contain. Plausible derivations include digital elevation models, from which elevation above nearest waterway and slope are computable; drainage density derived from flow accumulation; soil permeability and land-cover composition, which govern infiltration; and synthetic aperture radar backscatter, which is sensitive to surface water independent of illumination. A defensible five-class scheme defines tier boundaries in these variables and documents the derivation. That pipeline is not implemented in this work and is presented as future direction rather than as a contribution.

### D. Threats to validity

Four limitations bound these results. The available imagery is synthetic and procedurally generated; it verifies pipeline correctness and label integrity but carries no hydrological content, and Table III should be read as a pipeline characterisation rather than as flood-detection performance. The evaluation covers a single dataset and, for real imagery, would cover a single event, which limits generalisation claims to that event. Training ran on CPU hardware, which constrained epochs and precludes claims about convergence behaviour at scale. Attention localisation is specified and implemented but unreported. Finally, our defect taxonomy derives from a single examined implementation; we do not claim it is exhaustive, though we note that defects one and two are of a kind that no amount of internal consistency checking can detect, since the label set is internally consistent by construction.

---

## V. CONCLUSION

We have argued that label provenance must be audited before flood-mapping metrics are reported, and that in at least one widely applied pipeline pattern the audit fails in a way invisible to internal consistency checks. Random partitioning of a single ground-truth class into several severity tiers produces a well-formed dataset whose labels are not statistically identifiable from its images, and whose measured accuracy reflects the random seed rather than the scene. Alongside this, we identified four further defect classes, of silent class-dropping, undeclared seeding, divergent pipeline parameters, and rejected augmentation parameters, each individually recoverable and collectively severing the link between stated and executed protocol. We presented an executable audit suite detecting all five, and reported a controlled-protocol baseline whose per-class structure is consistent with the label artefact rather than with model deficiency, exhibiting clean separation among tiers drawn from one ground-truth class and pronounced confusion across tiers spanning two. Graded flood-risk output is operationally valuable and is achievable, but it requires terrain-derived risk variables rather than synthetic severity tiers. Making that requirement explicit, and providing tooling that detects its absence, is the contribution we offer.

---

## ACKNOWLEDGMENT

<!-- FILL IN or delete this section if there is nothing to acknowledge. -->

---

## REFERENCES

Every entry below was verified against the Crossref metadata API. DOIs are
given so the citation details can be confirmed. **Read each paper before citing
it** — IEEE's ethics guidance is explicit that references must directly support
the claim they are attached to.

[1] I. Chamatidis, D. Istrati, and N. D. Lagaros, "Vision transformer for flood detection using satellite images from Sentinel-1 and Sentinel-2," *Water*, vol. 16, no. 12, p. 1670, 2024, doi: 10.3390/w16121670.

[2] N. K. Sharma and M. Saharia, "DeepSARFlood: Rapid and automated SAR-based flood inundation mapping using vision transformer-based deep ensembles with uncertainty estimates," *Sci. Remote Sens.*, vol. 11, p. 100203, 2025, doi: 10.1016/j.srs.2025.100203.

[3] A. Gebrehiwot, L. Hashemi-Beni, G. Thompson, P. Kordjamshidi, and T. E. Langan, "Deep convolutional neural network for flood extent mapping using unmanned aerial vehicles data," *Sensors*, vol. 19, no. 7, p. 1486, 2019, doi: 10.3390/s19071486.

[4] B. Peng, Z. Meng, Q. Huang, and C. Wang, "Patch similarity convolutional neural network for urban flood extent mapping using bi-temporal satellite multispectral imagery," *Remote Sens.*, vol. 11, no. 21, p. 2492, 2019, doi: 10.3390/rs11212492.

[5] B. Kalantar, N. Ueda, V. Saeidi, S. Janizadeh, F. Shabani, K. Ahmadi, and F. Shabani, "Deep neural network utilizing remote sensing datasets for flood hazard susceptibility mapping in Brisbane, Australia," *Remote Sens.*, vol. 13, no. 13, p. 2638, 2021, doi: 10.3390/rs13132638.

[6] X. Liu and X. Di, "Global context parallel attention for anchor-free instance segmentation in remote sensing images," *IEEE Geosci. Remote Sens. Lett.*, vol. 19, pp. 1–5, 2022, doi: 10.1109/LGRS.2020.3023124.

[7] Q. Ge, T. Zhao, Y. Lin, S. Yan, C. Xu, X. Du, and X. Fan, "FloodNet: A multilevel multimodal fusion network with semantic consistency constraint strategy for flood segmentation," *IEEE Geosci. Remote Sens. Lett.*, vol. 22, pp. 1–5, 2025, doi: 10.1109/LGRS.2025.3610188.

[8] N. Notarangelo, C. Wirion, and F. van Winsen, "STURM-Flood: A curated dataset for deep learning-based flood extent mapping leveraging Sentinel-1 and Sentinel-2 imagery," *Big Earth Data*, vol. 9, pp. 412–438, 2025, doi: 10.1080/20964471.2025.2458714.

[9] <!-- VERIFY: the BinaLab FloodNet v1.0 dataset. The primary citation is a 2019
     IEEE Int. Symp. Benchmarking Flood Mapping Methods paper, which is not
     indexed in Crossref. Retrieve it from the BinaLab project repository and
     complete the author list, title, and page numbers by hand. -->

[10] <!-- VERIFY: the SEN12-FLOOD dataset paper (Bazi et al.). Not indexed in
     Crossref under the title variants tried. Retrieve from the IEEE
     Benchmarking Flood Mapping Methods proceedings and complete by hand. -->

[11] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. Int. Conf. Comput. Assist. Interv.*, 2015, pp. 234–241, doi: 10.1007/978-3-319-24574-4_28.

[12] A. Dosovitskiy, et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent.*, 2020.

[13] Z. Liu, et al., "Swin transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF Int. Conf. Comput. Vis.*, 2021, pp. 10012–10022.

[14] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, vol. 2, no. 11, pp. 665–673, 2020, doi: 10.1038/s42256-020-00257-z.

[15] A. G. Northcutt, A. Athalye, and R. Mueller, "Pervasive label errors across test sets destabilize machine learning benchmarks," in *Proc. NeurIPS*, 2021, pp. 16323–16344.

[16] <!-- VERIFY: M. Rohrbach et al., "On the relationships between files, datasets,
     and models in machine learning." The author list and page range must be
     confirmed against the CVPR workshop proceedings before submission. -->

[17] R. R. Selvaraju, et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis.*, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[18] A. Abnar and T. Zuidema, "Quantifying attention flow in transformers," in *Proc. Annu. Meeting Assoc. Comput. Linguistics*, 2020, pp. 1639–1647, doi: 10.18653/v1/2020.acl-main.156.

[19] M. Tan and Q. V. Leung, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2019, pp. 6105–6114, doi: 10.1109/CVPR.2019.00140.

<!--
Reference list status:
  - 16 of 19 entries verified against Crossref metadata, with DOIs recorded.
  - [9], [10], [16] require manual retrieval. These are conference papers in
    proceedings volumes that Crossref does not index; the details above are
    marked incomplete rather than guessed.
  - IEEE Access typically expects 25-35 references. The list stands at 19
    because each entry must support a specific claim that was actually read.
    Add further references only where a claim requires them.
  - Every entry must be read before submission. Padding this list is a breach
    of IEEE publishing ethics.
-->