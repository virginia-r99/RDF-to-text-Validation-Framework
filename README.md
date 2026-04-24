# Is Machine-Translated Data Useful Enough for Verbalisation?  
RDF-to-Text Generation with Machine-Translated Supervision

This repository contains the code, datasets, and experiment scripts accompanying the paper *“Is Machine-Translated Data Useful Enough for Verbalisation? Evaluating Machine-Translated Supervision for Multilingual RDF-to-Text Generation.”*  The project studies whether machine-translated “silver” WebNLG data in Spanish and Catalan can be reliably used for RDF-to-text training and evaluation, including back-translated English. 

We provide:
- The WebNLG_CA_BT dataset (Catalan + back-translated English) aligned with the original English WebNLG and the existing Spanish WebNLG corpus.
- Scripts and notebooks to reconstruct the dataset from translation outputs.
- Intrinsic and translationese analyses of the silver data.
- Few-shot and LoRA-based downstream evaluations on multiple LLMs. 

---

## Repository structure

```text
.
│   .gitattributes
│   LICENSE
│   requirements.txt
│
├── Dataset_generation/
│   ├── rebuild_triples_xml.ipynb
│   ├── rebuild_verbalisations_xml.ipynb
│   ├── revoting_triples.ipynb
│   ├── revoting_verbalisations.ipynb
│   ├── translate_WebNLG_bt_triples_catalan.py
│   ├── translate_WebNLG_triples_catalan.py
│   ├── translate_WebNLG_verbalisations_catalan.py
│   └── results/
│       ├── entity_translations_ca_en_revoted_with_output_token.csv
│       ├── entity_translations_ca_revoted_with_output_token.csv
│       ├── output_triples_format.ipynb
│       ├── registry_webnlg_ca_en_backtranslation.revoted.csv
│       ├── registry_webnlg_en_ca.revoted.csv
│       ├── relation_translations_ca_en_revoted_with_output_token.csv
│       └── relation_translations_ca_revoted_with_output_token.csv
│
├── Few-Shot_evaluation/
│   ├── Few-Shot_evaluation.ipynb
│   ├── Few-Shot_generation.py
│   └── results/
│       ├── generations__BSC-LT__salamandra-2b-instruct.csv
│       ├── generations__CohereLabs__tiny-aya-global.csv
│       ├── generations__HuggingFaceTB__SmolLM3-3B.csv
│       ├── generations__Qwen__Qwen3-4B-Instruct-2507.csv
│       ├── summary_by_model_lang.csv
│       ├── summary_by_model_lang_category.csv
│       └── summary_by_model_lang_split.csv
│
├── Intrinsic_evaluation/
│   ├── triples_intrinsic_evaluation.ipynb
│   └── verbalisations_intrinsic_evaluation.ipynb
│
├── LoRA/
│   ├── verbalisation_LoRA_evaluation.ipynb
│   ├── verbalisation_LoRA_generation.py
│   ├── verbalisation_LoRA_training.py
│   └── results/
│       ├── Qwen_bertscore_heatmap.png
│       ├── Qwen_bleu_heatmap.png
│       ├── SmolLM_bertscore_heatmap.png
│       ├── SmolLM_bleu_heatmap.png
│       ├── summary_by_model_lang.csv
│       ├── summary_by_model_lang_category.csv
│       └── summary_by_model_lang_split.csv
│
├── Translationese/
│   └── eval_translationese.ipynb
│
└── WebNLG_CA_BT/
    ├── train/
    ├── dev/
    └── test/
```

---

## WebNLG_CA_BT dataset

The `WebNLG_CA_BT` folder contains the multilingual WebNLG variants used in the paper: original English (gold), Spanish (MT with weak human supervision), Catalan (fully MT), and English back-translation from Catalan. 

- `train/`, `dev/`, `test/` reproduce the WebNLG data splits in XML format, organised by number of triples (`1triples`–`7triples`). 
- For Catalan, verbalisations are obtained by translating the English sentences with an agreement-based selection over three MT systems (NLLB-200-distilled-1.3B, madlad400-3b-mt, SalamandraTA-7B-Instruct), using language ID, string similarity clustering, and semantic similarity to the English reference. 
- Entity and relation labels are translated separately, using Wikidata Catalan labels when available and falling back to MT plus revoting; the CSVs in `Dataset_generation/results/` store these decisions.

These resources enable reproducible experiments on machine-translated supervision for RDF-to-text generation in English, Spanish, and Catalan, plus back-translated English.

---

## Dataset generation scripts

The `Dataset_generation` directory contains the pipeline to construct the Catalan and back-translated WebNLG data from raw translations and metadata. 

Key components:

- `translate_WebNLG_triples_catalan.py`: translates RDF triples (entities and relations) into Catalan, using multiple MT systems and a revoting strategy. 
- `translate_WebNLG_verbalisations_catalan.py`: translates English verbalisations into Catalan, generating multiple candidates and selecting a final output based on language identification, string similarity, and semantic similarity to the English source. 
- `translate_WebNLG_bt_triples_catalan.py`: constructs back-translated English triples from Catalan verbalisations (en_bt condition). 
- `rebuild_triples_xml.ipynb`, `rebuild_verbalisations_xml.ipynb`: rebuild WebNLG-style XML files from CSV registries and translation outputs.
- `revoting_triples.ipynb`, `revoting_verbalisations.ipynb`: apply revoting and filtering to translation candidates to ensure language validity and semantic closeness. 
- `results/*.csv`: store entity and relation translations, revoted choices, and registries linking triples and verbalisations across languages. 

Typical use:

1. Run the translation scripts to generate raw Catalan (and back-translated English) triples and verbalisations.
2. Optionally adjust the revoting notebooks if you want to experiment with different similarity thresholds or selection criteria.
3. Use the rebuild notebooks to generate final WebNLG-style XML corpora in `WebNLG_CA_BT/`.

---

## Intrinsic evaluation

The `Intrinsic_evaluation` folder contains notebooks that implement the multidimensional intrinsic evaluation framework described in the paper. 

- `verbalisations_intrinsic_evaluation.ipynb`:
  - Computes verbalisation integrity metrics: entity consistency, structural symmetry (POS density vectors), expansion ratio, and language consistency. 
  - Quantifies how Catalan and Spanish silver data differ from English gold at sentence level. 
- `triples_intrinsic_evaluation.ipynb`:
  - Computes triple-level faithfulness metrics: structural parity (triple counts), digit preservation, and character-level lexical alignment (chrF) for predicates and entities. 
  - Analyses the preservation of RDF structure and content between gold and silver variants. 

These notebooks directly instantiate RQ1 (Intrinsic Silver Quality) from the paper. 

---

## Translationese analysis

The `Translationese/eval_translationese.ipynb` notebook analyses translationese effects for the English back-translation (en_bt) compared to the original English gold references. 

- Uses spaCy to compute type–token ratio, lexical density, mean dependency distance, and token counts for English and back-translated English. 
- Quantifies how back-translated text differs in linguistic complexity and style, supporting the translationese discussion in the paper. 

---

## Few-shot evaluation

The `Few-Shot_evaluation` directory contains the few-shot probing experiments (RQ3) for several LLMs across English, Spanish, Catalan, and back-translated English. 

- `Few-Shot_generation.py`:
  - Loads the relevant WebNLG split.
  - Builds few-shot prompts for each language variant (en, es, ca, en_bt).
  - Queries different models (e.g., Salamandra-2B-Instruct, tiny-aya-global, SmolLM3-3B, Qwen3-4B-Instruct-2507) to generate verbalisations. 
- `Few-Shot_evaluation.ipynb`:
  - Evaluates generations using automatic metrics such as BLEU and BERTScore, and aggregates results by model, language, category, and split. 
- `results/`:
  - `generations__*csv`: raw generations for each model.
  - `summary_by_model_lang*.csv`: aggregated scores per model-language pair, language-category, and language-split. 

---

## LoRA fine-tuning experiments

The `LoRA` directory implements the LoRA-based fine-tuning and evaluation experiments used in RQ3. 

- `verbalisation_LoRA_training.py`:
  - Fine-tunes selected LLMs with LoRA adapters on different combinations of gold and silver WebNLG data (including full vs filtered silver subsets). 
- `verbalisation_LoRA_generation.py`:
  - Uses the fine-tuned models to generate verbalisations on the evaluation splits and dumps the outputs to CSV. 
- `verbalisation_LoRA_evaluation.ipynb`:
  - Computes BLEU, BERTScore, and other metrics, and produces heatmaps and aggregated summaries. 
- `results/`:
  - `*_bleu_heatmap.png`, `*_bertscore_heatmap.png`: visualisations of LoRA performance across languages and setups.
  - `summary_by_model_lang*.csv`: numeric summaries by model, language, category, and split. 

---

## Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/.../....git
cd ...

python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

All experiments were run in Python 3.10+, and using a virtual environment or Conda is recommended.

---

## Usage

Example workflows:

- **Rebuild the Catalan dataset**:
  - Run `translate_WebNLG_triples_catalan.py` and `translate_WebNLG_verbalisations_catalan.py`.
  - Use `revoting_*` notebooks to re-select candidates if needed.
  - Execute `rebuild_*_xml.ipynb` to regenerate XML files in `WebNLG_CA_BT/`. 

- **Run intrinsic evaluation**:
  - Open `Intrinsic_evaluation/verbalisations_intrinsic_evaluation.ipynb` and `triples_intrinsic_evaluation.ipynb`.
  - Point them to the desired gold–silver pairs (e.g., en vs ca, en vs es, en vs en_bt). 

- **Few-shot and LoRA experiments**:
  - Use `Few-Shot_evaluation/Few-Shot_generation.py` and `LoRA/verbalisation_LoRA_training.py` to generate outputs.
  - Inspect results and visualisations in the corresponding `results/` folders. 

---

## Contact

For questions, feedback, or issues related to this repository or the WebNLG_CA_BT dataset, please open a GitHub issue or contact:

- ...
