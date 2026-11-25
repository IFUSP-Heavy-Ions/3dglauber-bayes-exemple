# Midrapidity dNch/dη: RHIC cross-detector table

Built 2026-11-17. Every number was parsed by script from a HEPData YAML submission or from the arXiv LaTeX source. Nothing was read off a figure. Raw downloads are in `_raw/`. The parsing scripts are in `_scripts/`.

## Files

| file | content |
|---|---|
| `midrapidity_dndeta.csv` | 260 rows: one per (experiment, system, √sNN, centrality class) |
| `sphenix_dndeta_vs_eta.csv` | 165 rows: sPHENIX Au+Au 200, dN/dη in 11 η bins (|η|<1.1) × 15 centralities |
| `_raw/` | downloaded arXiv tarballs (`src.tgz`, extracted to `x/`) and HEPData YAML submissions (`sub.tgz`, extracted to `y/` or `ins*/`) |
| `_scripts/build.py` | builds both CSVs. Run it as `python build.py <study_dir> <phenix_tex.json>` (needs pyyaml) |
| `_scripts/phenix_tex.py` | parses the PHENIX LaTeX Nch tables to JSON |
| `_scripts/cmp.py` | cross-checks PHENIX HEPData against the LaTeX tables |

### Column conventions in `midrapidity_dndeta.csv`

- **`dNdeta`** is the value as published, never divided by Npart/2. No conversion from dN/dη/(0.5 Npart) was needed anywhere.
- **`sys_err_plus` / `sys_err_minus`** hold the uncertainty exactly as published (all sources are symmetric).
  - For PHENIX, sPHENIX and BRAHMS Au+Au, this number is the **total (stat ⊕ sys)**. These rows carry `ERR=TOTAL` at the start of `notes`.
  - For PHOBOS it is a pure systematic error.
- **`stat_err`** is empty when the source gives no separate statistical error.
- **`sys_cl`**:
  - `90` for PHOBOS. It is explicitly 90% C.L., so divide by ≈1.645 to get 1σ.
  - `unknown` where the paper does not state a confidence level. PHENIX, sPHENIX and BRAHMS do not state one; they are conventionally ≈1σ, but they do not say so.
- **`eta_min` / `eta_max`**:
  - `-1/1` for PHOBOS: |η|<1, tracklets.
  - `-0.35/0.35` for PHENIX: taken from the paper text (drift-chamber acceptance), because it is not in the HEPData tables.
  - `-0.3/0.3` for sPHENIX: average over |η|<0.3.
  - `0/0` for BRAHMS: "η = 0" column or point.
- **`npart`, `npart_err`** come from the same table as dN/dη. Each experiment uses its own Glauber/HIJING model.

## Rows per (experiment, system, energy)

| experiment | system | √sNN (GeV) | rows | centrality span (%) |
|---|---|---|---|---|
| PHOBOS | Au+Au | 200 | 15 | 0–70 |
| PHOBOS | Au+Au | 130 | 11 | 0–50 |
| PHOBOS | Au+Au | 62.4 | 15 | 0–70 |
| PHOBOS | Au+Au | 19.6 | 13 | 0–60 |
| PHOBOS | Cu+Cu | 200 | 11 | 0–50 |
| PHOBOS | Cu+Cu | 62.4 | 11 | 0–50 |
| PHOBOS | Cu+Cu | 22.4 | 11 | 0–50 |
| PHENIX | Au+Au | 200 | 12 | 0–60 |
| PHENIX | Au+Au | 130 | 12 | 0–60 |
| PHENIX | Au+Au | 62.4 | 12 | 0–60 |
| PHENIX | Au+Au | 39 | 11 | 0–55 |
| PHENIX | Au+Au | 27 | 10 | 0–50 |
| PHENIX | Au+Au | 19.6 | 10 | 0–50 |
| PHENIX | Au+Au | 14.5 | 10 | 0–50 |
| PHENIX | Au+Au | 7.7 | 10 | 0–50 |
| PHENIX | Cu+Cu | 200 | 9 | 0–45 |
| PHENIX | Cu+Cu | 62.4 | 8 | 0–40 |
| PHENIX | Cu+Au | 200 | 12 | 0–60 |
| PHENIX | U+U | 193 | 10 | 0–50 |
| PHENIX | d+Au | 200 | 8 | 0–70 |
| PHENIX | 3He+Au | 200 | 8 | 0–70 |
| sPHENIX | Au+Au | 200 | 15 | 0–70 |
| BRAHMS | Au+Au | 200 | 6 | 0–50 |
| BRAHMS | Au+Au | 130 | 6 | 0–50 |
| BRAHMS | d+Au | 200 | 4 | 0–30, 30–60, 60–80, MB (MB coded 0–100) |
| **total** | | | **260** | |

## Provenance per source

### 1. PHOBOS compilation: arXiv:1011.1940 (PRC 83 024913)

- **Source:** https://arxiv.org/e-print/1011.1940, saved as `_raw/phobos/src.tgz`. The file used is `x/PhMultDataOnly_final_Nov03.TEX`.
- **Tables used:**
  - **Table V** (LaTeX `\label{table1}`) covers Au+Au 19.6 and 62.4.
  - **Table V continued** (`\label{table1a}`, after `\setcounter{table}{4}`) covers Au+Au 130 and 200.
  - **Table VI** (`\label{table2}`) covers Cu+Cu 22.4, 62.4 and 200.
  - These are the 5th and 6th table environments in the file, which confirms the numbering.
- **Columns:** Bin, N_part, dN/dη at |η|<1, and dN/dη/(N_part/2).
- **Error definitions:** the caption says "The errors are systematic errors at 90% C.L.; statistical errors are negligible." Errors are recorded as given, with `sys_cl=90`, and they apply to both dN/dη and N_part.
- **Parsing:** a regex over the active (non-`%`) table lines. Each row's note includes its .tex line number.
- **Check against column 4:** dN/dη ÷ (N_part/2) recomputed from the printed integers reproduces the published column 4 within integer rounding (worst cases are peripheral Cu+Cu, where Δ ≈ 0.04–0.07). The parsing is correct; the printed integers are rounded.
- **Points to note:**
  - **Typo in the source:** in Au+Au 62.4 the row between 50–55% and 60–65% is printed as **"50-60%"**. It is stored as **55–60%**, and the row's `notes` flags this.
  - **Commented-out lines:** Cu+Cu 62.4 and 200 each have a commented-out line, glued after a `%` comment, that gives 0–3% values of 139±10 (62.4) and Npart 108±3 (200). The active lines were used: 138±10 and 108±4.
  - **d+Au:** the paper has **no midrapidity dN/dη table** for d+Au. Table III gives only Npart and the total Nch. The d+Au dN/dη(η) values exist only in a figure, so this item is **missing**.
  - **Not extracted:** Table II (0–6%, dN/dη/(Npart/2) only) and Table IV (p+p 200/410, |η|<1: 2.25 +0.37/−0.30 and 2.87 +0.44/−0.43, 90% C.L.). Neither is a per-centrality A+A value.

### 2. PHENIX compilation: arXiv:1509.06727 (PRC 93 024901), HEPData ins1394433 v1

- **Sources:**
  - HEPData record listing: https://www.hepdata.net/record/ins1394433?format=json, saved as `_raw/phenix/record.json`.
  - Full YAML submission: https://www.hepdata.net/download/submission/ins1394433/1/yaml, saved as `_raw/phenix/sub.tgz`.
  - LaTeX source: https://arxiv.org/e-print/1509.06727, saved under `_raw/phenix/x/ppg174v1.tex`.
- **HEPData tables used** (the "Multiplicity …" tables): 8 (Au+Au 200), 10 (130), 12 (62.4), 14 (39), 16 (27), 18 (19.6), 20 (14.5), 22 (7.7), 24 (Cu+Cu 200), 26 (Cu+Cu 62.4), 28 (Cu+Au 200), 30 (U+U 193), 32 (d+Au 200) and 34 (³He+Au 200). Npart comes from the same table.
- **Error label:** HEPData gives a single error labelled `sys`. The matching paper tables say: "The uncertainties include the total statistical and systematic uncertainties." So the stored error is the **total**, `stat_err` is empty, and no C.L. is stated.
- **Possible gap in the error:** the paper says the trigger-efficiency uncertainty (0.3–16% in Table III) is drawn as a separate band in the plots. It is unclear whether the table errors include it. For example, the 0–5% total is 5.3%, while the quadrature sum of the Table III components without trigger efficiency is ≈5.6%.
- **η range:** "The DC covers the pseudorapidity region |η|<0.35" (paper text). The HEPData tables do not record it.
- **Cross-check of every HEPData table against the LaTeX tables** (`_scripts/cmp.py`): 12 of 14 are identical. Two problems were found:
  - **U+U 193 (HEPData Table 30) is mislabelled in HEPData.**
    - The paper table has no N_q column. HEPData nevertheless shows one, and the dN/dη values (830.4 …) sit in the "N_q" column, while dN/dη/(0.5 Npart) (3.97 …) sits in the "dN_ch/dη" column.
    - The U+U rows therefore come from the **LaTeX table `tab:nchUU193` (Table XXX)**.
  - **Au+Au 27 GeV, 45–50%** appears in the paper table `tab:nchAuAu027` (Table XVI): 45.7 ± 3.9, Npart 60.8 ± 6.0. It is **missing from HEPData Table 16**, so it was added from the LaTeX.
  - The Roman table numbers (VIII, X, …, XXXIV) come from the `% Table_…` comments in the LaTeX source.

### 3. sPHENIX: arXiv:2504.02240, HEPData ins2907537 v1

- **Sources:** the YAML from https://www.hepdata.net/download/submission/ins2907537/1/yaml and the LaTeX from https://arxiv.org/e-print/2504.02240 (`_raw/sphenix/x/result.tex`).
- **Midrapidity table:** HEPData "Table 4" (`Table.yaml`) = paper Table 4. It gives dN/dη at |η|<0.3, ⟨Npart⟩, and the normalized value for 15 classes (0–3, 3–6, 6–10, then 5% steps to 70%). All 15 rows match the LaTeX table exactly (checked by script).
- **Per-η table:** HEPData "Figure 5" (`Figure5.yaml`) goes to `sphenix_dndeta_vs_eta.csv`. It has 11 bins of width 0.2 over −1.1 < η < 1.1 for the same 15 classes. Some values carry only 2 significant figures as published (e.g. 740.0 ± 40.0).
- **Errors:** the single error, labelled `sys`, is the total uncertainty of the two-method weighted average (paper Sec. Results and the systematics table): correlated 3.5–7.9% plus uncorrelated <0.9%, total 3.5–7.9%. No separate stat error and no C.L. are stated, so `sys_cl=unknown`.

### 4. BRAHMS

**Au+Au 200, nucl-ex/0112001 (PRL 88 202301), ins567754**

- **Source:** the paper's LaTeX **Table I** (`\label{tb:dndeta}`, `_raw/brahms/src0112001/dndeta01.tex`), "η = 0" column, with Npart from the same table.
- **Error definition:** the caption reads "Total uncertainties, dominated by the systematics, are indicated."
- **Quoted systematic (text):** "In general, statistical errors on the measurements are less than 1%, with systematic errors of 8% and 10% for the SiMA and BBC arrays, respectively. The systematic errors are dominated by overall scaling uncertainties … there may be as much as a 3% relative scale error between the two energies."
- **Stat for 0–5% (text):** "dN_ch/dη|η=0 = 625 ± 1(stat) ± 55(syst)". So `stat_err=1` for 0–5% only.
- **What HEPData has:** ins567754 contains only per-η tables with stat errors (Tables 1a/1b/3) and Table 5 (dN/dη/(Npart/2), whose "stat" error is clearly the ~9% total). It does **not** contain the η=0 table. The paper table was therefore used.
- **For reference:** the HEPData Table 1a bins adjacent to η=0 are 619 ± 6.5 (−0.303…−0.003) and 627 ± 6.8 (−0.001…0.299) for 0–5%.

**Au+Au 130, nucl-ex/0108016 (PLB 523 227), ins561518**

- **Source:** HEPData Table 1 (`tab1.yaml`), the "ETARAP = 0.0" dN/dη column. Its error label is `tot`, matching the caption "Total uncertainties, dominated by the systematics".
- **dN/dη:** identical to the arXiv LaTeX table.
- **Npart differs** between HEPData (published version) and the arXiv LaTeX table:

  | centrality (%) | HEPData Npart | arXiv Npart |
  |---|---|---|
  | 0–5 | 350 | 352 |
  | 5–10 | 300 | 299 |
  | 10–20 | 240 | 235 |
  | 20–30 | 160 | 165 |
  | 30–40 | 110 | 114 |
  | 40–50 | 75 | 75 |

  The HEPData values are stored, and the arXiv values are given in `notes`.
- **Npart error:** not tabulated. The text says "<0.5% for 5% centrality, increasing to 4% at 50%".
- **Sys quoted in text:** "SiMA, ±8% for η<1.5, increasing to ±10% for η≥2.5; TMA ±12% …; BBC ±10%".

**d+Au 200, nucl-ex/0401025 (PRL 94 032301), ins643085** (optional item)

- **Source:** HEPData Table 2 (`tab2.yaml`), the single point at η = 0.0, with separate `stat` and `sys`.
- **Npart:** taken from the HEPData qualifier `NPART` (no error is given).
- **Classes:** 0–30, 30–60, 60–80 and MB. MB is coded as `cent_lo=0`, `cent_hi=100`.
- **Caveat:** d+Au is asymmetric, so η=0 is not a plateau. It is also a single bin, not an average.

## Missing items and why

| item | reason |
|---|---|
| PHOBOS d+Au midrapidity dN/dη per centrality | not tabulated in 1011.1940; figure only. HEPData has no PHOBOS records (all 404 per `cross_detector_overlaps.md`). |
| PHOBOS stat errors | stated "negligible"; not given |
| PHENIX separate stat vs sys | only the total (stat ⊕ sys) is published |
| sPHENIX separate stat vs sys | only the total is published (uncorrelated part <0.9%) |
| BRAHMS Au+Au 200 stat per centrality | only the 0–5% stat (±1) is quoted. The text says stat is <1% in general. |
| BRAHMS 130 Npart errors | not tabulated |
| C.L. for PHENIX, sPHENIX, BRAHMS | not stated in the papers, recorded as `unknown` |
| PHENIX Au+Au 130 standalone paper (nucl-ex/0012008) | not extracted; it overlaps with the compilation (see `cross_detector_overlaps.md`) |

## Things to watch when comparing detectors

1. **The η windows differ:** PHOBOS |η|<1, PHENIX |η|<0.35, sPHENIX |η|<0.3, BRAHMS η=0. The plateau is flat to a few %, but PHOBOS |η|<1 is not identical to η≈0.
2. **The centrality binning differs:**
   - PHOBOS and sPHENIX use 0–3, 3–6, 6–10, then 5% steps.
   - PHENIX uses 0–5, then 5% steps; for d+Au/³He+Au it uses 10% steps.
   - BRAHMS uses 0–5, 5–10, then 10% steps.
   - The only exact matches between experiments are the 5% bins from 10% upward (PHOBOS, PHENIX and sPHENIX) and the BRAHMS 0–5 and 5–10 bins, which match PHENIX. Anything else needs rebinning, and each experiment's Npart comes from its own Glauber model.
3. **The error meaning differs:** PHOBOS is 90% C.L. sys only. The others give a total at an unstated C.L. (≈1σ by convention). Do not mix them without rescaling PHOBOS by 1/1.645.
4. **PHENIX U+U on HEPData is mislabelled.** Anyone pulling it straight from HEPData would get dN/dη/(0.5 Npart) in place of dN/dη.
