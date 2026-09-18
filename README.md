# Multidimensional Validation Framework for Multilingual RDF-to-Text Resources

This repository contains the code, data, precomputed outputs, and downstream experiments associated with the manuscript **“A Multidimensional Validation Framework for Knowledge Preservation in Multilingual RDF-to-Text Resources.”**

The main contribution is a **multidimensional validation framework for aligned multilingual RDF-to-text resources**. The framework is independent of the procedure used to construct a resource. In the empirical WebNLG instantiation provided here, human-authored English is treated as the reliable aligned source for source-dependent checks, while Spanish and Catalan are used as target-language case studies.

The repository also includes the Catalan and back-translated English WebNLG construction utilities, few-shot and LoRA downstream experiments, and an additional translationese analysis.

---

## Framework overview

The framework organises validation into three complementary Quality Assessment (QA) dimensions. Their diagnostic evidence is retained separately rather than collapsed into a single global score.

| Dimension | Sub-question | Main diagnostic evidence |
|---|---|---|
| **QA1: Structural validity** | **QA1.1 Structural and instance integrity** | Triple parity, target-text presence, lexicalisation-ID alignment and uniqueness, RDF-format sanity checks, record integrity |
|  | **QA1.2 Recurring-component consistency** | Dominant mapping rates and mapping-review flags for recurring entities, values, and predicates |
| **QA2: Knowledge preservation and factual faithfulness** | **QA2.1 Source-to-target knowledge preservation** | Cross-lingual verbalisation and triple-set similarity, predicate diagnostics, source-to-target numerical/date preservation |
|  | **QA2.2 Target RDF–text factual faithfulness** | Minimum triple coverage, minimum text groundedness, target triple-to-text literal retention |
| **QA3: Target-language validity and legitimate variation** | **QA3.1 Target-language validity and source-language leakage** | Whole-text language identification, wrong-language, exact-copy, and local source-language leakage checks |
|  | **QA3.2 Legitimate linguistic and multi-reference variation** | Expansion ratio and aligned-reference advantage |

The current WebNLG instantiation uses English as the reliable aligned source for source-dependent comparisons. The conceptual framework is not restricted to English and can be instantiated with a different aligned source considered sufficiently reliable for the intended validation setting.

---

## Empirical study

The repository supports the four stages reported in the manuscript:

1. **Controlled validation benchmark** — controlled structural, factual, and linguistic perturbations test diagnostic responsiveness and selectivity.
2. **Blinded human–automatic agreement assessment** — automatic review priorities are compared with human judgements on risk-stratified original-resource cases.
3. **Application to the unmodified resources** — QA1–QA3 are applied to the complete Spanish and Catalan WebNLG adaptations.
4. **Secondary downstream utility study** — few-shot generation and LoRA fine-tuning assess whether the characterised resources remain useful for RDF-to-text generation.

The controlled benchmark and human–automatic comparison provide the principal evidence about framework behaviour. The full-resource application demonstrates deployment at corpus scale, while the downstream experiments provide complementary evidence about resource utility.

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
│       ├── Controlled_test/
│       └── Manual_agreement/
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

A small number of implementation filenames retain the earlier internal term `audit`; they belong to the same validation pipeline described in the manuscript.

---

## Multilingual WebNLG data

`WebNLG_CA_BT/` contains the aligned WebNLG data used throughout the repository:

- **English (`en`)** — human-authored WebNLG source lexicalisations and modified triples.
- **Spanish (`es`)** — semi-automatically adapted Spanish WebNLG with targeted human revision.
- **Catalan (`ca`)** — automatically adapted Catalan WebNLG.
- **Back-translated English (`en_bt`)** — English obtained through the Catalan translation/back-translation path and used in downstream and supplementary analyses.

The XML entries preserve the aligned structured and textual representations. Typical tags include:

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

For each target language used by the validation framework, the repository contains results for **16,653 aligned RDF records** and **45,031 lexicalisations**.

---

## Installation

Create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Several experiments download pretrained models from Hugging Face and are substantially faster on a CUDA-capable GPU.

### Additional component-specific dependencies

Some optional repository components use packages that are not pinned in `requirements.txt`:

```bash
pip install trl pywikibot langid langdetect
```

They are used by:

- `trl` — LoRA training;
- `pywikibot` — Wikidata lookup during dataset construction;
- `langid` and `langdetect` — dataset-generation revoting utilities.

---

## Framework implementation

### Shared metric code

`Framework/Code/audit_metric_core.py` contains shared implementations used across the validation notebooks, including semantic similarity, literal extraction and preservation, recurring-component consistency, QA2 support measures, and QA3 language/variation logic.

`Framework/Code/qa11_metric_core.py` contains the shared implementation of the QA1.1 structural-integrity checks.

### Main notebooks

The framework notebooks are organised in the following order:

1. `01_QA1_resource_integrity_consistency.ipynb`
2. `02_QA2_knowledge_preservation_faithfulness.ipynb`
3. `03_QA3_target_language_quality_variation.ipynb`
4. `04_ControlledStress.ipynb`
5. `05_framework_results_summary_exports.ipynb`

The first three notebooks apply QA1–QA3 to the unmodified Spanish and Catalan resources. `04_ControlledStress.ipynb` constructs and evaluates the controlled validation benchmark. `05_framework_results_summary_exports.ipynb` consolidates QA and controlled-benchmark outputs into publication-oriented summaries.

---

## Precomputed framework outputs

Precomputed results are included under `Framework/Results/`.

### `QA1/`

Contains structural-integrity and recurring-component-consistency outputs, including:

- record-level QA1.1 metrics;
- structural summaries;
- recurring-component mapping profiles;
- variable mappings selected for review;
- figures and run metadata.

### `QA2/`

Contains source-to-target and target RDF–text faithfulness outputs, including:

- lexicalisation-level QA2 metrics;
- corpus summaries;
- English calibration thresholds;
- relative literal-retention results;
- bootstrap summaries;
- figures and run metadata.

### `QA3/`

Contains target-language and variation outputs, including:

- lexicalisation-level language diagnostics;
- target-language summaries;
- variation summaries;
- figures and run metadata.

### `Controlled_test/`

Contains the publication-oriented outputs from the controlled validation benchmark:

```text
Framework/Results/Controlled_test/
├── figures/
├── tables/
├── qa2_omission_severity_summary.csv
└── paper_output_manifest.json
```

These outputs include QA1 structural-response figures, QA2 corruption and severity analyses, QA3 language/selectivity analyses, and overall controlled-benchmark summaries.

### `Manual_agreement/`

Contains the blinded human–automatic comparison materials and exported evaluation results:

```text
Framework/Results/Manual_agreement/
├── manual_agreement_pipeline_reproducible.py
└── manual_agreement/
    ├── manual_agreement_annotator.xlsx
    ├── manual_agreement_key.csv
    ├── manual_agreement_key.xlsx
    ├── manual_agreement_sampling_summary.csv
    ├── manual_agreement_thresholds.json
    └── evaluation/
        ├── manual_agreement_evaluation.xlsx
        ├── manual_agreement_evaluation_by_subqa_language.csv
        └── manual_agreement_disagreements.csv
```

The annotator workbook is blinded to automatic scores and review-priority assignments. The evaluation outputs contain the human–automatic comparison used in the manuscript.

---

## Controlled validation benchmark

`Framework/Code/04_ControlledStress.ipynb` constructs controlled variants that isolate predefined failure types while preserving the rest of each record whenever possible.

The perturbations cover:

- **QA1** — deleted or duplicated triples, missing lexicalisations, identifier corruption, malformed RDF components or placeholders, source markup, and recurring-mapping inconsistency;
- **QA2** — wrong-record text, entity/predicate/literal substitutions, omissions, and plausible or unrelated unsupported additions;
- **QA3** — complete English copies, inserted English clauses, and benign alternative lexicalisations.

The benchmark is designed for **diagnostic isolation**. It tests whether the corresponding diagnostics react in the expected direction and remain selective on clean or benign controls; it is not intended to estimate the natural prevalence of errors in the original resources.

Before rerunning this notebook, configure the benchmark/output paths in its initial configuration cell.

---

## Blinded human–automatic agreement assessment

`Framework/Results/Manual_agreement/manual_agreement_pipeline_reproducible.py` implements the risk-stratified sampling and comparison workflow for the six sub-QAs.

The generation stage creates a blinded annotator workbook and a separate automatic key:

```bash
python Framework/Results/Manual_agreement/manual_agreement_pipeline_reproducible.py generate \
    --results-root <QA_RESULTS_ROOT> \
    --output-dir <OUTPUT_DIR>
```

After annotation, the comparison can be reproduced with:

```bash
python Framework/Results/Manual_agreement/manual_agreement_pipeline_reproducible.py evaluate \
    --annotated <ANNOTATED_WORKBOOK.xlsx> \
    --key-csv <AUTOMATIC_KEY.csv> \
    --output-dir <EVALUATION_OUTPUT_DIR>
```

The human labels use **Low / Medium / High / Cannot judge**. The manuscript reports exact ordinal agreement, within-one-level agreement, review-priority recall, and automatic-Low confirmation. Because the sample is risk-stratified, these measures characterise sample-level prioritisation correspondence rather than corpus-level defect prevalence.

---

## Configuration for local reruns

The committed results can be inspected directly without rerunning the notebooks.

For a fresh local execution, update the path/configuration cells in the framework notebooks to point to the local checkout. The expected data root is:

```text
<repository root>/WebNLG_CA_BT
```

The QA notebooks generate their full intermediate outputs before the publication-oriented files are consolidated. `05_framework_results_summary_exports.ipynb` supports explicit environment-variable overrides for those intermediate directories:

```bash
export WEBNLG_REPO_ROOT="<repository root>"
export QA1_RESULTS_DIR="<QA1 intermediate results>"
export QA2_RESULTS_DIR="<QA2 intermediate results>"
export QA3_RESULTS_DIR="<QA3 intermediate results>"
export CONTROLLEDSTRESS_RESULTS_DIR="<ControlledStress intermediate results>"
export CHAPTER_FRAMEWORK_EXPORT_DIR="<summary export directory>"
```

The precomputed files under `Framework/Results/Controlled_test/` are publication-oriented exports. A complete rerun of `05_framework_results_summary_exports.ipynb` requires the raw/intermediate CSV outputs produced by `04_ControlledStress.ipynb`.

---

## Dataset construction utilities

`Dataset_generation/` contains the utilities used to construct the Catalan and back-translated English variants included in the repository.

The Catalan pipeline uses multiple translation systems:

- `facebook/nllb-200-distilled-1.3B`
- `google/madlad400-3b-mt`
- `BSC-LT/salamandraTA-7b-instruct`

The triple-generation workflow can additionally use Wikidata labels through `pywikibot`. Candidate translations are stored and subsequently re-evaluated/revoted using language and similarity evidence before the selected representations are written back into WebNLG-style XML.

Key files include:

- `translate_WebNLG_triples_catalan.py`
- `translate_WebNLG_verbalisations_catalan.py`
- `translate_WebNLG_bt_triples_catalan.py`
- `revoting_triples.ipynb`
- `revoting_verbalisations.ipynb`
- `rebuild_triples_xml.ipynb`
- `rebuild_verbalisations_xml.ipynb`

`Dataset_generation/results/` contains the selected entity/relation mappings and multilingual registries used by the reconstruction workflow.

These utilities document the provenance of the empirical case studies. The validation framework itself can be applied independently of this construction pipeline.

---

## Few-shot downstream evaluation

`Few-Shot_evaluation/` contains the secondary few-shot RDF-to-text generation study.

`Few-Shot_generation.py` evaluates four multilingual instruction-model families using English, Spanish, Catalan, and back-translated English demonstrations:

- `BSC-LT/salamandra-2b-instruct`
- `CohereLabs/tiny-aya-global`
- `HuggingFaceTB/SmolLM3-3B`
- `Qwen/Qwen3-4B-Instruct-2507`

Example:

```bash
cd Few-Shot_evaluation

python Few-Shot_generation.py \
    --data_root ../WebNLG_CA_BT \
    --output_dir ./outputs_fewshot_webnlg
```

`Few-Shot_evaluation.ipynb` evaluates the generated texts using lexical, semantic, and output-length measures. Precomputed generations and aggregated summaries are included under `Few-Shot_evaluation/results/`.

---

## LoRA fine-tuning experiments

`LoRA/` contains the secondary fine-tuning study:

- `verbalisation_LoRA_training.py` — LoRA training;
- `verbalisation_LoRA_generation.py` — generation from the fine-tuned adapters;
- `verbalisation_LoRA_evaluation.ipynb` — evaluation and cross-resource summaries.

Example training command:

```bash
python LoRA/verbalisation_LoRA_training.py \
    --data_root WebNLG_CA_BT \
    --output_root LoRA/outputs \
    --model Qwen/Qwen3-4B-Instruct-2507
```

The training code supports Qwen3, tiny-aya, SmolLM3, and Salamandra. The manuscript reports the LoRA experiments for **Qwen3** and **SmolLM3**, selected based on the preceding prompt/few-shot results.

Historical filtered Spanish and Catalan conditions are retained as an ablation comparison. They originate from an earlier single-score filtering procedure and are not outputs of the multidimensional validation framework.

Precomputed evaluation summaries and figures are available under `LoRA/results/`.

---

## Additional analysis

`Translationese/eval_translationese.ipynb` contains a supplementary comparison between human-authored English and English obtained through the Catalan back-translation path. It analyses vocabulary diversity, lexical density, dependency distance, and text length.

This analysis is retained as an additional repository component and is separate from the validation study reported in the manuscript.

---

## Suggested reproduction order

For the main validation study:

1. Install the required dependencies.
2. Use the committed `WebNLG_CA_BT/` data.
3. Run the QA1, QA2, and QA3 notebooks.
4. Run `04_ControlledStress.ipynb`.
5. Run the human–automatic agreement pipeline if reproducing the human comparison.
6. Run `05_framework_results_summary_exports.ipynb` to consolidate the framework outputs.

The dataset-generation utilities are only required to reconstruct the Catalan/back-translated data. Few-shot and LoRA experiments reproduce the secondary downstream utility study. The translationese notebook is an additional analysis.

---

## License

This repository is distributed under the **Apache License 2.0**. See `LICENSE` for the full licence text.

---

## Citation

Citation information for the accompanying manuscript will be added when the final bibliographic record is available.
