# Multidimensional Validation Framework for Multilingual RDF-to-Text Resources

This repository contains the code, data, intermediate outputs, and downstream experiments associated with the manuscript **“A Multidimensional Validation Framework for Knowledge Preservation in Multilingual RDF-to-Text Resources.”**

The main contribution is a **multidimensional validation framework for aligned multilingual RDF-to-text resources**. The framework is independent of how a resource was constructed. In the empirical WebNLG instantiation provided here, human-authored English is treated as the reliable aligned source for source-dependent checks, while Spanish and Catalan are used as target-language case studies.

The repository also contains the construction pipeline for the Catalan and back-translated English WebNLG variants, few-shot and LoRA downstream experiments, and an additional translationese analysis. The translationese analysis is retained as a supplementary repository component and is **not part of the main empirical study reported in the manuscript**.

---

## Validation framework

The framework organises validation into three complementary Quality Assessment (QA) dimensions. Their diagnostic evidence is kept separate rather than collapsed into a single global score.

### QA1 — Structural validity

- **QA1.1: Structural and instance integrity**  
  Checks target triple parity, target-text presence, lexicalisation-ID alignment and uniqueness, and resource-specific RDF sanity conditions.
- **QA1.2: Recurring-component consistency**  
  Examines whether recurring RDF entities, values, and predicates receive stable target representations while retaining legitimate variation for contextual review.

### QA2 — Knowledge preservation and factual faithfulness

- **QA2.1: Source-to-target knowledge preservation**  
  Compares aligned source and target RDF/text representations using verbalisation similarity, triple-set similarity, predicate diagnostics, and exact numerical/date preservation.
- **QA2.2: Target RDF–text factual faithfulness**  
  Evaluates whether the target verbalisation covers its target RDF input and avoids unsupported content, using minimum triple coverage, minimum text groundedness, and target triple-to-text literal retention.

### QA3 — Target-language validity and legitimate variation

- **QA3.1: Target-language validity and source-language leakage**  
  Combines whole-text language identification with high-confidence wrong-language, exact-source-copy, and local source-language leakage checks.
- **QA3.2: Legitimate linguistic and multi-reference variation**  
  Uses expansion ratio and aligned-reference advantage as descriptive evidence for interpreting valid cross-lingual and multi-reference variation.

The current WebNLG implementation uses Spanish and Catalan as target languages. English provides the reliable aligned source for source-dependent comparisons in this empirical setting. The framework itself is not restricted to English as the source language.

---

## Empirical validation in the manuscript

The repository supports the four stages used in the paper:

1. **Controlled validation benchmark** — known structural, factual, and linguistic perturbations are introduced to test diagnostic responsiveness and selectivity.
2. **Blinded human–automatic agreement assessment** — automatic review priorities are compared with human judgements on risk-stratified original-resource cases.
3. **Application to the unmodified resources** — QA1–QA3 are applied to the complete Spanish and Catalan WebNLG adaptations.
4. **Secondary downstream utility study** — few-shot generation and LoRA fine-tuning evaluate whether the characterised resources remain useful for RDF-to-text generation.

The controlled benchmark and human–automatic comparison provide the main evidence about framework behaviour. Full-resource application demonstrates deployment, while the downstream experiments provide secondary evidence about resource utility.

---

## Repository structure

```text
.
├── Dataset_generation/
│   ├── translate_WebNLG_triples_catalan.py
│   ├── translate_WebNLG_verbalisations_catalan.py
│   ├── translate_WebNLG_bt_triples_catalan.py
│   ├── revoting_triples.ipynb
│   ├── revoting_verbalisations.ipynb
│   ├── rebuild_triples_xml.ipynb
│   ├── rebuild_verbalisations_xml.ipynb
│   └── results/
│
├── Framework/
│   ├── Code/
│   │   ├── 01_QA1_resource_integrity_consistency.ipynb
│   │   ├── 02_QA2_knowledge_preservation_faithfulness.ipynb
│   │   ├── 03_QA3_target_language_quality_variation.ipynb
│   │   ├── 04_ControlledStress.ipynb
│   │   ├── 05_framework_results_summary_exports.ipynb
│   │   ├── audit_metric_core.py
│   │   └── qa11_metric_core.py
│   └── Results/
│       ├── QA1/
│       ├── QA2/
│       ├── QA3/
│       ├── Stress_audit/
│       └── Manual_audit/
│
├── Few-Shot_evaluation/
│   ├── Few-Shot_generation.py
│   ├── Few-Shot_evaluation.ipynb
│   └── results/
│
├── LoRA/
│   ├── verbalisation_LoRA_training.py
│   ├── verbalisation_LoRA_generation.py
│   ├── verbalisation_LoRA_evaluation.ipynb
│   └── results/
│
├── Translationese/
│   └── eval_translationese.ipynb
│
├── WebNLG_CA_BT/
│   ├── train/
│   ├── dev/
│   └── test/
│
├── requirements.txt
└── LICENSE
```

---

## Multilingual WebNLG data

`WebNLG_CA_BT/` contains aligned WebNLG records with the language variants used across the repository:

- **English (`en`)** — human-authored WebNLG source lexicalisations and modified triples.
- **Spanish (`es`)** — semi-automatically adapted Spanish WebNLG with targeted human revision.
- **Catalan (`ca`)** — automatically adapted Catalan WebNLG.
- **Back-translated English (`en_bt`)** — English obtained through the Catalan translation/back-translation path, used in downstream and supplementary analyses.

The XML entries keep the aligned structured and textual representations together. Typical tags include:

```text
<modifiedtripleset> ... <mtriple> ...        # English source RDF
<spanishtripleset>  ... <striple> ...        # Spanish RDF
<catalantripleset>  ... <ctriple> ...        # Catalan RDF
<enbttripleset>     ... <bttriple> ...       # Back-translated English RDF
<lex lang="en"> ...
<lex lang="es"> ...
<lex lang="ca"> ...
<lex lang="en_bt"> ...
```

For each target language used by the validation framework, the included results cover **16,653 aligned RDF records** and **45,031 lexicalisations**.

---

## Framework implementation

### Shared metric code

`Framework/Code/audit_metric_core.py` contains shared implementations used across the QA notebooks, including semantic similarity, literal extraction/preservation, recurring-component consistency, QA2 support measures, and QA3 language/variation logic.

`Framework/Code/qa11_metric_core.py` contains the shared QA1.1 structural-integrity implementation.

### Framework notebooks

Run the framework notebooks in this conceptual order:

1. `01_QA1_resource_integrity_consistency.ipynb`
2. `02_QA2_knowledge_preservation_faithfulness.ipynb`
3. `03_QA3_target_language_quality_variation.ipynb`
4. `04_ControlledStress.ipynb`
5. `05_framework_results_summary_exports.ipynb`

The first three notebooks apply the framework to the unmodified Spanish and Catalan resources. `04_ControlledStress.ipynb` constructs and evaluates controlled perturbations for framework validation. `05_framework_results_summary_exports.ipynb` reads the exported QA and controlled-benchmark results and produces consolidated publication-oriented summaries.

### Included framework outputs

Precomputed outputs are stored under `Framework/Results/`:

- `QA1/` — structural-integrity summaries, recurring-component mappings, review candidates, plots, and run metadata.
- `QA2/` — lexicalisation-level metrics, source/target preservation summaries, English calibration thresholds, literal-retention results, plots, and run metadata.
- `QA3/` — target-language diagnostics, variation summaries, lexicalisation-level metrics, plots, and run metadata.
- `Stress_audit/` — publication-oriented controlled-perturbation summary tables, figures, and a paper-output manifest.
- `Manual_audit/` — the reproducible human–automatic comparison pipeline together with the generated blinded-assessment materials.

---

## Blinded human–automatic agreement assessment

`Framework/Results/Manual_audit/manual_audit_pipeline_reproducible.py` implements the sampling and evaluation workflow for the six sub-QAs.

The generated materials currently included in the repository are stored under:

```text
Framework/Results/Manual_audit/manual_audit/
```

They include:

- `manual_audit_annotator.xlsx` — blinded workbook shown to the human assessor.
- `manual_audit_key.csv` / `.xlsx` — automatic review strata and diagnostic basis, kept separate from the annotator workbook.
- `manual_audit_sampling_summary.csv` — available and sampled Low/Medium/High strata by sub-QA and language.
- `manual_audit_thresholds.json` — thresholds used to construct the automatic review-priority strata.

The script supports two modes:

```bash
python Framework/Results/Manual_audit/manual_audit_pipeline_reproducible.py generate \
    --results-root <RESULTS_ROOT> \
    --output-dir <OUTPUT_DIR>
```

and, after human annotation:

```bash
python Framework/Results/Manual_audit/manual_audit_pipeline_reproducible.py evaluate \
    --annotated <COMPLETED_WORKBOOK.xlsx> \
    --key-csv <manual_audit_key.csv> \
    --output-dir <EVALUATION_OUTPUT_DIR>
```

The human comparison uses **Low / Medium / High / Cannot judge** labels. The manuscript reports sample-level exact agreement, within-one-level agreement, review-priority recall, and automatic-Low confirmation. Because the sample is risk-stratified, it is not intended to estimate corpus-level defect prevalence.

---

## Controlled validation benchmark

`Framework/Code/04_ControlledStress.ipynb` creates controlled variants that isolate predefined failure types while preserving the rest of each record whenever possible.

The perturbations cover:

- **QA1** — deleted/duplicated triples, missing text, identifier corruption, malformed or placeholder components, source markup, and recurring-mapping inconsistency.
- **QA2** — record mismatch, entity/predicate/literal changes, omissions, and plausible/unrelated unsupported additions.
- **QA3** — complete English copies, inserted English clauses, and benign alternative lexicalisations.

The benchmark is designed for **diagnostic isolation**, not for estimating the natural frequency of errors in the original resources. The publication-oriented outputs under `Framework/Results/Stress_audit/` summarise diagnostic response, omission/addition behaviour, and selectivity controls.

---

## Dataset construction utilities

`Dataset_generation/` contains the utilities used to construct the Catalan and back-translated English WebNLG variants retained in this repository.

The Catalan construction pipeline uses multiple translation systems, including:

- `facebook/nllb-200-distilled-1.3B`
- `google/madlad400-3b-mt`
- `BSC-LT/salamandraTA-7b-instruct`

The triple-generation code can also use Wikidata labels through `pywikibot`. Translation candidates are stored and subsequently re-evaluated/revoted using language checks and similarity evidence. The rebuild notebooks then write the selected representations back into WebNLG-style XML.

Key files:

- `translate_WebNLG_triples_catalan.py` — translates RDF entities/relations into Catalan.
- `translate_WebNLG_verbalisations_catalan.py` — translates English lexicalisations into Catalan.
- `translate_WebNLG_bt_triples_catalan.py` — constructs the back-translated English condition.
- `revoting_triples.ipynb` and `revoting_verbalisations.ipynb` — re-evaluate candidate translations and final selections.
- `rebuild_triples_xml.ipynb` and `rebuild_verbalisations_xml.ipynb` — rebuild multilingual WebNLG XML files.
- `Dataset_generation/results/` — selected entity/relation mappings and multilingual registries used by the reconstruction pipeline.

These construction utilities document the provenance of the empirical case studies. They are **not a requirement of the validation framework**, which can be applied to resources constructed by other manual, automatic, or hybrid procedures.

---

## Few-shot downstream evaluation

`Few-Shot_evaluation/` contains the secondary few-shot RDF-to-text generation study.

`Few-Shot_generation.py` evaluates four multilingual instruction-model families across English, Spanish, Catalan, and back-translated English demonstrations:

- Salamandra-2B-Instruct
- tiny-aya-global
- SmolLM3-3B
- Qwen3-4B-Instruct-2507

The script supports command-line configuration, for example:

```bash
cd Few-Shot_evaluation
python Few-Shot_generation.py \
    --data_root ../WebNLG_CA_BT \
    --output_dir ./outputs_fewshot_webnlg
```

`Few-Shot_evaluation.ipynb` evaluates generations against all available references and computes BLEU, ROUGE-L, METEOR, chrF++, BERTScore, multilingual embedding similarity, and output-length/expansion statistics. Precomputed generation files and aggregated summaries are included under `Few-Shot_evaluation/results/`.

---

## LoRA fine-tuning experiments

`LoRA/` contains the secondary fine-tuning study.

- `verbalisation_LoRA_training.py` trains LoRA adapters for the configured language/resource conditions.
- `verbalisation_LoRA_generation.py` generates RDF-to-text outputs from the fine-tuned adapters.
- `verbalisation_LoRA_evaluation.ipynb` evaluates the generations and creates aggregate summaries and heatmaps.

Example training call:

```bash
python LoRA/verbalisation_LoRA_training.py \
    --data_root WebNLG_CA_BT \
    --output_root LoRA/outputs \
    --model Qwen/Qwen3-4B-Instruct-2507
```

The training script retains several model choices, while the manuscript takes **Qwen3** and **SmolLM3** forward for the reported LoRA experiments based on the preceding prompt/few-shot results. It also retains historical filtered Spanish/Catalan conditions used as an ablation comparison. These filtered subsets come from an earlier single-score selection policy and should not be interpreted as outputs of the multidimensional validation framework.

Precomputed summary tables and heatmaps are available under `LoRA/results/`.

---

## Additional analysis: translationese

`Translationese/eval_translationese.ipynb` contains an **exploratory supplementary analysis** comparing human-authored English with English obtained through the Catalan back-translation path.

It measures:

- type–token ratio,
- lexical density,
- mean dependency distance,
- token count.

This notebook is retained for completeness because it was part of the broader experimental repository. **The translationese analysis is not part of the validation framework and is not included in the main empirical study reported in the accompanying manuscript.**

---

## Installation

A Python 3.10+ environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Several experiments download pretrained models from Hugging Face and are substantially faster on a CUDA-capable GPU.

### Component-specific dependencies

The current `requirements.txt` covers most of the validation and evaluation stack, but some scripts use additional packages that are not pinned in the repository snapshot:

- `trl` for `LoRA/verbalisation_LoRA_training.py`;
- `pywikibot` for Wikidata lookup during dataset construction.

Install them when reproducing those components:

```bash
pip install trl pywikibot
```

---

## Path configuration and reproducibility

The committed framework results are organised under `Framework/Results/`. Some notebooks retain development-time paths, historical directory names, or placeholder path values in their configuration cells. Before rerunning the framework on another machine, review those configuration cells and point them to the current repository layout.

The intended local layout is:

```text
REPO_ROOT      = <repository root>
DATA_ROOT      = <repository root>/WebNLG_CA_BT
CORE_PATH      = <repository root>/Framework/Code/audit_metric_core.py
QA11_CORE_PATH = <repository root>/Framework/Code/qa11_metric_core.py

QA1 results    = <repository root>/Framework/Results/QA1
QA2 results    = <repository root>/Framework/Results/QA2
QA3 results    = <repository root>/Framework/Results/QA3
```

For `05_framework_results_summary_exports.ipynb`, explicit environment-variable overrides can be used to avoid relying on historical default paths:

```bash
export WEBNLG_REPO_ROOT="$PWD"
export QA1_RESULTS_DIR="$PWD/Framework/Results/QA1"
export QA2_RESULTS_DIR="$PWD/Framework/Results/QA2"
export QA3_RESULTS_DIR="$PWD/Framework/Results/QA3"
export CONTROLLEDSTRESS_RESULTS_DIR="<raw ControlledStress output directory>"
export CHAPTER_FRAMEWORK_EXPORT_DIR="$PWD/Framework/Results/Framework_summary"
```

`05_framework_results_summary_exports.ipynb` expects the **raw/intermediate output directory generated by `04_ControlledStress.ipynb`**, including files such as `combined_detection_summary.csv` and the perturbation-specific summary CSVs. The committed `Framework/Results/Stress_audit/` folder contains publication-oriented controlled-benchmark tables and figures, not the complete intermediate bundle expected by notebook 05.

Similarly, the first framework notebooks and `04_ControlledStress.ipynb` contain machine-specific or placeholder path configuration that should be adapted before a clean rerun. The dataset-generation scripts/notebooks also contain development-time relative or machine-specific paths and GPU/model settings, so review their configuration blocks before reconstructing the data.

---

## Suggested reproduction order

For the validation study itself, the recommended order is:

1. Install the dependencies.
2. Use the committed `WebNLG_CA_BT/` data directly. Dataset reconstruction is only necessary if you want to reproduce the construction pipeline itself.
3. Run QA1, QA2, and QA3 notebooks.
4. Run `04_ControlledStress.ipynb`.
5. Run the human–automatic sampling/evaluation pipeline if reproducing the human comparison.
6. Run `05_framework_results_summary_exports.ipynb` to consolidate the framework outputs.
7. Run few-shot and LoRA experiments only if reproducing the secondary downstream utility study.
8. Run the translationese notebook only if reproducing the supplementary analysis.

---

## License

The repository is distributed under the **Apache License 2.0**. See `LICENSE` for details.

---

## Citation

Citation information for the accompanying manuscript will be added when the final bibliographic record is available.
