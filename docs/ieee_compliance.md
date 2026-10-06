# IEEE Access Compliance Checklist — FloodSense

Sourced from the **official IEEE Access author guidance** at
<https://ieeeaccess.ieee.org/authors/submission-guidelines/> and from the
**official IEEE Access LaTeX template** (`ACCESS_latex_template_20260513`),
downloaded directly from that page. Items verified against the rendered PDF are
marked ✅.

> **Superseded guidance:** an earlier revision of this file listed a generic
> IEEE Transactions-style first footnote (three paragraphs, affiliation inside
> the footnote, no DOI, no biographies). That is **not** IEEE Access practice and
> has been replaced below.

---

## 0. Template and class

| Item | Requirement | Status |
|---|---|---|
| Template file | "prepared in a double column, single-spaced format using a **required IEEE Access template**" | ✅ `\documentclass{ieeeaccess}` |
| Source | `ACCESS_latex_template_20260513` (from the official page) | ✅ bundled in `docs/paper/` |
| Files submitted | **Word file and a PDF**, content must match exactly, ≤ 40 MB | ⚠️ PDF ✅ / Word **missing** |
| Manuscript type | "Research Article" | action at submission |
| ORCID | submitting author must have a **publicly visible, populated ORCID ID** | **blocked — need user** |
| Language | poor grammar is rejected outright | proofread |

The class lives in `docs/paper/` alongside `ieeeaccess.cls`, `spotcolor.sty`,
`IEEEtran.cls`, `IEEEtran.bst` and the bundled `t1-formata-*` / `giovannistd`
fonts. It must be compiled from that directory.

**Never substitute** `\documentclass[11pt,journal]{IEEEtran}`. That was the
original error in this project: IEEE Access ships its own class, and the 11pt
journal option is not it.

---

## 1. Front matter — all verified in the rendered PDF ✅

| Element | Requirement | Status |
|---|---|---|
| Title | no "new"/"novel", no math symbols | ✅ |
| Authors | `\author{...}` + `\authorrefmark{n}` | ✅ three names, verbatim |
| Affiliation | `\address[n]{...}` — dept, institution, road, locality, city, **country** | ✅ |
| Corresponding | `\corresp{Corresponding author: ... (e-mail: ...)}` | ✅ |
| Funding | in the first-page `\tfootnote`, **not** in Acknowledgment | ✅ no-specific-grant statement |
| Prior publication | first-page footnote | ✅ present, **still unconfirmed by authors** |
| Abstract | single paragraph, ≤ 250 words, no citations/equations | ✅ **220 words, 1 paragraph** |
| Keywords | `\begin{keywords}`, alphabetical, **3–10** | ✅ **7 terms, alphabetical** |
| `\history` | publication-date line | ✅ template placeholder |
| `\doi` | **mandatory** — `\maketitle` dereferences `\@doi` | ⚠️ fake placeholder `10.1109/ACCESS.XXXXXXX` |
| `\EOD` | **mandatory** after last biography | ✅ |
| Running head | first author + `et al.` | ✅ `Varshini et al.: ...` |

**Section order:** I. Introduction, II. Methodology, III. Results, IV. Discussion,
V. Conclusion, Acknowledgment, References, Biographies. ✅ No separate Related
Work section — literature is folded into the Introduction.

Section headings use **Roman numerals** (this class numbers them that way), and
table captions use **Arabic** (`TABLE 1`). Both are automatic.

---

## 2. Acknowledgment — AI disclosure (new, mandatory)

> "The use of artificial intelligence (AI)-generated text in an article shall be
> disclosed in the acknowledgements section. The sections of the paper that use
> AI-generated text shall have a citation to the AI system used to generate the
> text."

Present in `docs/paper/paper.tex` as an unnumbered `\section*{Acknowledgment}` ✅.

⚠️ **The guidance asks for a per-section citation to the AI system**, which our
draft does not attempt. Confirm the exact wording IEEE expects before submitting.

---

## 3. Biographies — required, currently BLOCKED

> "Short biographies are required for **ALL** authors ... directly within the
> article **below the references section**."

| Author | Status |
|---|---|
| Varshini D. N. | **placeholder** — degree, institution, year, role needed |
| Maanya B. S. | **placeholder** — same |
| Aishwarya S. | **placeholder** — same |

Rendered as `\begin{IEEEbiographynophoto}` (no photo on file). The PDF currently
shows `BIOGRAPHY REQUIRED -- degree(s), ...` for each. **This must not be
submitted as-is.**

---

## 4. Results and claims

- 34 audit checks pass (18 label/provenance + 16 pipeline contract) ✅
- CNN (EfficientNet-B3): acc 0.8533, F1-macro 0.8498, AUC 0.9732 ✅
- ViT-B/16: acc 0.7000, F1-macro 0.6989, AUC 0.9592 ✅
- Per-tier F1 spread: CNN 0.654–0.968, ViT 0.596–0.933 ✅
- **No architecture ranking is claimed** — 8 vs 6 epochs is unmatched ✅
- Tables II/III (now `TABLE 3`/`TABLE 4`) are pipeline characterisation on
  **synthetic** data, not flood-detection performance ✅ stated explicitly
- No pointing-game number reported (no pixel ground truth) ✅

---

## 5. References

- 19 references, `[1]`–`[19]`, contiguous, none undefined, none uncited ✅
- All 15 DOIs fetched and confirmed ✅
- Cite only what was actually read — "Inflating citations by adding unnecessary
  references is considered a breach of publishing ethics."

⚠️ IEEE permits `et al.` abbreviation only when the source has more than six
authors **or** the list is unreasonably long. Refs `[12] [13] [15] [17]` use
`et al.` and should be expanded to the full list.

---

## 6. Graphics

| Property | Requirement |
|---|---|
| Resolution | 300 dpi halftone, 600 dpi line art |
| Color space | RGB |
| Format | PDF, EPS, TIFF (**not JPEG for figures**) |
| Fonts | embedded, 8–10 pt at final size |
| Captions | below figures, **above tables** |
| Alt text | required for accessibility |

⚠️ The paper currently has **no figures**. Confusion matrices and training plots
exist in `outputs/plots/`. The class requires `logo.png`, `notaglinelogo.png`
and `bullet.png` for its own chrome — **do not delete them from `docs/paper/`**.

---

## Blockers before submission

1. **Author biographies** — need real credentials for all three authors. ❌
2. **DOI placeholder** — replace `10.1109/ACCESS.XXXXXXX` per the portal. ❌
3. **Prior-publication statement** — unconfirmed by the authors. ❌
4. **Co-author approval** — Maanya and Aishwarya must read and approve; author
   order to be settled with them. ❌
5. **ORCID ID** — submitting author, publicly visible. ❌
6. **Word version** — a Word file is required alongside the PDF, matching
   exactly. ❌
7. **AI disclosure wording** — confirm IEEE's expected form. ⚠️
8. **Reference expansion** — refs `[12] [13] [15] [17]` should list full
   author lists. ⚠️
9. **Real imagery** — synthetic data cannot support a flood-mapping claim;
   FloodNet (real, cited at `[9]`) would also supply pixel masks. ⚠️
10. **Seed variance** — ≥ 3 seeds per architecture before any comparative claim.
    ⚠️
