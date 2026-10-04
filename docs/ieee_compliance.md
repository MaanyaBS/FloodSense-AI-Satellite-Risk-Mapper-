# IEEE Authoring Compliance Checklist

Applies to the FloodSense paper. Sourced from the IEEE Author Center:
Structure Your Article, IEEE Editorial Style Manual for Authors, IEEE
Reference Guide, and IEEE Math Typesetting Guide.

---

## 1. Required section order (mandatory)

IEEE prescribes this sequence. Sections may not be reordered or renamed.

| # | Section | Status |
|---|---------|--------|
| 1 | Title | drafted |
| 2 | Authors | **blocked** — needs real affiliation |
| 3 | Abstract | drafted |
| 4 | Keywords | drafted |
| 5 | First footnote | **blocked** — needs funding + prior-publication statements |
| 6 | I. Introduction | drafted |
| 7 | II. Methodology | drafted |
| 8 | III. Results | **blocked** — no measurements |
| 9 | IV. Discussion | drafted (interim) |
| 10 | V. Conclusion | drafted |
| 11 | References | drafted |
| 12 | Acknowledgments | optional |

**Correction from the earlier outline:** there is no standalone "Related
Work" section in the IEEE structure. The literature review is folded into
the Introduction, per IEEE guidance that the introduction "includes a review
of the existing literature to position your research within the broader
scientific field and to show the novelty of your work."

Use Roman numerals for section headings. Subsections use letters
(A, B, C). IEEE Access uses Arabic numerals for headings, not Roman —
confirm against the target template.

---

## 2. Title

- Specific, concise, descriptive
- Keywords and short phrases, as few words as possible
- **Must NOT contain "new" or "novel"** — reader already assumes novelty
- **Must NOT contain mathematical symbols** (may not render)

---

## 3. Abstract

- **Single paragraph**
- **250 words maximum**
- **Self-contained:** no abbreviations, no citations, no footnotes, no equations
- States research conducted, conclusions reached, and implications
- Highlights what is novel
- **No mathematical symbols** in title or abstract

---

## 4. Keywords

- 3–5 terms or phrases
- Define all abbreviations
- Prefer standardized terms — use the IEEE Thesaurus (free access via IEEE)
- Applies the abbreviation consistently thereafter

---

## 5. First footnote (unnumbered, ≥3 paragraphs, exact order)

**Paragraph 1** — all of:
- Full financial support / funding (also NOT in Acknowledgments)
- Prior conference presentation of this or related work, with the DOI of the
  conference version (not a preprint)
- Corresponding author name and email
- **If part of a thesis or dissertation, state it in the last sentence**
- If research involves human subjects or animals: a required review-board
  statement. Not applicable here.

**Paragraph 2** — affiliations for each author:
department, university/corporation, city, state/province, postal code,
country. Country and corresponding author email are mandatory.

**Paragraph 3** — IEEE notice on supplementary materials and color figures.

All subsequent footnotes are numbered consecutively. **Do not use asterisks
or daggers.**

---

## 6. Introduction

Per IEEE: must include the literature review, position the work in the field,
show novelty, state the research question, and explain why it matters.

---

## 7. Methodology

Per IEEE: "A detailed methodology section will make your article
reproducible by other researchers." Must state what was done and how.

---

## 8. Results

Figures for trends and visual information. **Tables where exact values
matter.**

---

## 9. Discussion

Per IEEE: what the results mean and how they contribute to the field.

---

## 10. Conclusion

Per IEEE: "Be careful not to inflate your findings." May note broader
implications and areas needing further study.

---

## 11. References

IEEE numeric style, in order of citation. Format via the IEEE Reference
Preparation Assistant (refassist.ieee.org).

**Ethics rule (verbatim from IEEE):** *"Be sure to only cite references that
directly support your work. Inflating citations by adding unnecessary
references is considered a breach of publishing ethics."*

Rules:
- Numbered in order of first appearance in text
- Journal titles in full on first use, IEEE-abbreviated thereafter — use the
  official IEEE journal title list
- Author names as initials: A. B. Surname
- "et al." for 3+ authors (IEEE style uses all three then et al. for 4+ —
  verify against the Reference Guide)
- Include DOIs where available
- Target 25–35

---

## 12. Graphics

From IEEE graphics guidance:

| Property | Requirement |
|----------|-------------|
| Resolution | 300 dpi for halftone; 600 dpi for line art |
| Color space | RGB |
| Format | PDF, EPS, TIFF (not JPEG for figures) |
| Vector preferred | for line art and diagrams |
| Fonts | embed; use 8–10 pt at final size |
| Text in figures | must be legible at print size, same font family as body |
| Captions | below figures, above tables |
| Table captions | sentence case, ending in a period |
| Alt text | required for accessibility |

---

## 13. Language and style

- Active voice, present tense for what the paper does, past tense for what was done
- Merriam-Webster for spelling
- Chicago Manual of Style for grammar not covered by IEEE's manual
- Define every abbreviation at first use in the body text (not just the abstract)
- Spell out units; use SI units
- Numbers: spell out zero through nine unless used as measurements or in equations

---

## 14. Ethics and authorship

- Must meet IEEE authorship criteria (substantial contribution, drafting or
  revision, final approval). Sole authorship is acceptable for a sole
  contributor's work.
- Prior publication must be disclosed with DOI
- Funding must be disclosed in the first footnote
- Corrected papers and retractions follow defined procedures — not applicable here

---

## Blockers before submission

1. **Real affiliation** — department, university, city, state, postal code, country, email
2. **Prior-publication statement** — is this part of an existing thesis or project report?
3. **Funding statement** — was any part funded? If not, state that.
4. **Results section** — CNN run in progress; ViT not started. Synthetic demo
   data cannot support a flood-mapping claim.
5. **Real imagery** — FloodNet or SEN12-FLOOD required
6. **Seed variance** — ≥3 seeds per architecture to support a controlled-protocol claim
7. **Ground-truth masks** — required for the attention-localisation metric