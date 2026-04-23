#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import gc
import json
import random
import re
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 180)

TOKEN = "..."

DEFAULT_DATA_ROOT = Path("../WebNLG_CA_BT")
DEFAULT_FINETUNE_ROOT = Path("./outputs_qwen_finetuned")

PARSE_SPLITS = ["train", "dev", "test"]
EVAL_SPLITS = ["test"]
TARGET_LANGS = ["en", "es", "ca", "en_bt"]

BASE_QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BASE_SMOLLM_MODEL = "HuggingFaceTB/SmolLM3-3B"

MAX_NEW_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0
REPETITION_PENALTY = 1.0

TRUST_REMOTE_CODE = True
SAVE_EVERY = 50
LIMIT_PER_SPLIT = None
OVERWRITE_EXISTING = False

TRIPLESET_MAP = {
    "en": ["modifiedtripleset"],
    "es": ["spanishtripleset"],
    "ca": ["catalantripleset"],
    "en_bt": ["enbttripleset", "backtranslationtripleset"],
}

LANG_NAME = {
    "en": "English",
    "es": "Spanish",
    "ca": "Catalan",
    "en_bt": "English",
}

LEXICALISATION_TAG_CANDIDATES = ["lex", "text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned causal LM adapters with the same zero-shot prompt used in SFT"
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--finetune_root", type=Path, default=DEFAULT_FINETUNE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs_eval_finetuned_zeroshot"))
    parser.add_argument("--limit_per_split", type=int, default=LIMIT_PER_SPLIT)
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", type=str, default=BASE_QWEN_MODEL)
    parser.add_argument("--adapter_dirs", type=Path, nargs="*", default=None)
    return parser.parse_args()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def safe_text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return normalize_ws("".join(node.itertext()))


def split_triple_text(triple_text: str) -> Tuple[str, str, str]:
    txt = normalize_ws(triple_text)
    parts = [p.strip() for p in re.split(r"\s*\|\s*", txt)]
    if len(parts) >= 3:
        s = parts[0]
        p = parts[1]
        o = " | ".join(parts[2:])
        return s, p, o
    return txt, "", ""


def entry_attr(entry: ET.Element, *names: str) -> str:
    for name in names:
        if name in entry.attrib:
            return entry.attrib[name]
    lower_map = {k.lower(): v for k, v in entry.attrib.items()}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return ""


def read_triples(entry: ET.Element, lang: str) -> List[str]:
    container = None
    for tag in TRIPLESET_MAP[lang]:
        container = entry.find(tag)
        if container is not None:
            break
    if container is None:
        return []

    triples = []
    for child in list(container):
        txt = safe_text(child)
        if txt:
            triples.append(txt)
    return triples


def extract_lexicalisations(entry: ET.Element, lang: str) -> List[Dict[str, str]]:
    valid_tags = {t.lower() for t in LEXICALISATION_TAG_CANDIDATES}
    lex_rows = []

    for node in entry.iter():
        if node.tag.lower() not in valid_tags:
            continue
        node_lang = (node.attrib.get("lang", "") or "").lower()
        if node_lang != lang.lower():
            continue
        text = safe_text(node)
        if not text:
            continue
        lex_rows.append(
            {
                "lex": text,
                "comment": node.attrib.get("comment", ""),
                "lid": node.attrib.get("lid", ""),
                "lex_lang": node_lang,
            }
        )
    return lex_rows


def lexicalisations_to_json(lexicalisations: List[Dict[str, str]]) -> str:
    payload = []
    for i, lex in enumerate(lexicalisations, start=1):
        lid = (lex.get("lid") or f"lex_{i}").strip()
        payload.append({lid: lex.get("lex", "")})
    return json.dumps(payload, ensure_ascii=False)


def serialize_triples(triples: List[str]) -> str:
    return "\n".join(f"- {t}" for t in triples)


def make_prompt(lang: str, triples: List[str]) -> str:
    return (
        f"Generate one faithful {LANG_NAME[lang]} sentence from the RDF triples below. "
        f"Preserve the facts and do not add unsupported information.\n\n"
        f"RDF triples:\n{serialize_triples(triples)}"
    )


def infer_triple_bucket_from_path(xml_path: Path, split: str) -> str:
    split_lower = split.lower()
    path_parts = [p.lower() for p in xml_path.parts]
    if split_lower in path_parts:
        split_idx = path_parts.index(split_lower)
        rel_parts = xml_path.parts[split_idx + 1:]
        if len(rel_parts) >= 2:
            return rel_parts[0]
    return ""


def build_align_key(split: str, category: str, eid: str, triple_count: int) -> str:
    return f"{split}|||{category}|||{eid}|||{triple_count}"


def parse_xml_file(xml_path: Path, split: str, lang: str) -> List[Dict]:
    fallback_category = xml_path.stem
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []

    triple_bucket = infer_triple_bucket_from_path(xml_path, split)
    entry_nodes = [n for n in root.iter() if n.tag.lower() == "entry"]

    for entry_idx, entry in enumerate(entry_nodes):
        eid = entry_attr(entry, "eid", "id") or f"entry_{entry_idx}"
        size_attr = entry_attr(entry, "size")
        category_attr = entry_attr(entry, "category") or fallback_category

        triples = read_triples(entry, lang)
        if not triples:
            continue

        triple_count = int(size_attr) if str(size_attr).isdigit() else len(triples)
        triples_struct = [
            {"subject": s, "predicate": p, "object": o, "raw": raw}
            for raw in triples
            for (s, p, o) in [split_triple_text(raw)]
        ]

        lexicalisations = extract_lexicalisations(entry, lang)
        if not lexicalisations:
            continue

        align_key = build_align_key(split, category_attr, eid, triple_count)
        reference = lexicalisations[0]["lex"]
        prompt = make_prompt(lang, triples)

        rows.append(
            {
                "align_key": align_key,
                "split": split,
                "lang": lang,
                "category": category_attr,
                "xml_file": xml_path.name,
                "xml_path": str(xml_path),
                "triple_bucket": triple_bucket,
                "eid": eid,
                "size": triple_count,
                "num_triples": triple_count,
                "entry_idx": entry_idx,
                "num_lexicalisations": len(lexicalisations),
                "reference": reference,
                "lexicalisations": lexicalisations_to_json(lexicalisations),
                "triples": triples,
                "triples_struct": triples_struct,
                "prompt": prompt,
                "messages": json.dumps([{"role": "user", "content": prompt}], ensure_ascii=False),
            }
        )

    return rows


def parse_language_split(data_root: Path, split: str, lang: str) -> pd.DataFrame:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    xml_files = sorted(split_dir.rglob("*.xml")) if split.lower() in {"train", "dev"} else sorted(split_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in: {split_dir}")

    rows = []
    for xml_file in tqdm(xml_files, desc=f"Parsing {lang}/{split}"):
        rows.extend(parse_xml_file(xml_file, split=split, lang=lang))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Parsed 0 rows from {split_dir} for lang={lang}")
    return df


def _hf_kwargs() -> Dict[str, str]:
    return {"token": TOKEN} if TOKEN else {}


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name)


def adapter_eval_slug(run_name: str, experiment_name: str) -> str:
    return f"{model_slug(run_name)}__{model_slug(experiment_name)}"


def output_csv_path(output_dir: Path, eval_name: str) -> Path:
    return output_dir / f"generations__{model_slug(eval_name)}.csv"


def get_model_family(model_name: str) -> str:
    m = (model_name or "").lower()
    if "smollm" in m:
        return "smollm"
    if "qwen" in m:
        return "qwen"
    return "other"


def prepare_dataset(data_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_dfs = []
    for lang in TARGET_LANGS:
        for split in PARSE_SPLITS:
            all_dfs.append(parse_language_split(data_root, split, lang))
    df_all = pd.concat(all_dfs, ignore_index=True)

    run_rows = []
    for lang in TARGET_LANGS:
        df_lang = df_all[df_all["lang"] == lang].copy()
        for split in EVAL_SPLITS:
            df_split = df_lang[df_lang["split"] == split].copy()
            if LIMIT_PER_SPLIT is not None:
                df_split = df_split.head(LIMIT_PER_SPLIT).copy()

            for _, row in df_split.iterrows():
                run_rows.append(
                    {
                        "lang": row["lang"],
                        "split": row["split"],
                        "category": row["category"],
                        "eid": row["eid"],
                        "size": row["size"],
                        "num_triples": row["num_triples"],
                        "triple_bucket": row["triple_bucket"],
                        "xml_file": row["xml_file"],
                        "xml_path": row["xml_path"],
                        "align_key": row["align_key"],
                        "num_lexicalisations": row["num_lexicalisations"],
                        "lexicalisations": row["lexicalisations"],
                        "triples": json.dumps(row["triples"], ensure_ascii=False),
                        "triples_struct": json.dumps(row["triples_struct"], ensure_ascii=False),
                        "prompt": row["prompt"],
                        "messages": row["messages"],
                    }
                )

    run_df = pd.DataFrame(run_rows)
    return df_all, run_df


def write_manifests(output_dir: Path, run_df: pd.DataFrame, args: argparse.Namespace):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(output_dir / "evaluation_manifest.csv", index=False)

    config_snapshot = {
        "base_model": args.base_model,
        "base_model_family": get_model_family(args.base_model),
        "parse_splits": PARSE_SPLITS,
        "eval_splits": EVAL_SPLITS,
        "target_langs": TARGET_LANGS,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "limit_per_split": args.limit_per_split,
        "overwrite_existing": args.overwrite_existing,
        "data_root": str(args.data_root),
        "finetune_root": str(args.finetune_root),
        "output_dir": str(output_dir),
        "prompt_style": "same_as_training_zero_shot_user_only_one_generation_per_entry",
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)


def discover_finetuned_runs(finetune_root: Path, expected_base_model: str) -> List[Dict[str, str]]:
    if not finetune_root.exists():
        raise FileNotFoundError(f"Missing fine-tune root: {finetune_root}")

    runs = []
    for run_dir in sorted([p for p in finetune_root.iterdir() if p.is_dir()]):
        final_adapter_dir = run_dir / "final_adapter"
        run_config_path = run_dir / "run_config.json"

        if not final_adapter_dir.exists():
            continue
        if not (final_adapter_dir / "adapter_config.json").exists():
            continue

        experiment_name = run_dir.name
        base_model_name = expected_base_model

        if run_config_path.exists():
            with run_config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            experiment_name = cfg.get("experiment_name", experiment_name)
            base_model_name = cfg.get("model_name", expected_base_model)

        if base_model_name != expected_base_model:
            continue

        runs.append(
            {
                "eval_name": adapter_eval_slug(run_dir.name, experiment_name),
                "run_dir": str(run_dir),
                "adapter_dir": str(final_adapter_dir),
                "base_model_name": base_model_name,
                "base_model_family": get_model_family(base_model_name),
                "experiment_name": experiment_name,
            }
        )

    if not runs:
        raise ValueError(
            f"No fine-tuned runs found under {finetune_root} for base model {expected_base_model}"
        )

    return runs


def normalize_adapter_inputs(adapter_dirs: List[Path], expected_base_model: str) -> List[Dict[str, str]]:
    runs = []
    for p in adapter_dirs:
        p = p.resolve()
        if p.name == "final_adapter":
            run_dir = p.parent
            adapter_dir = p
        else:
            run_dir = p
            adapter_dir = p / "final_adapter"

        if not adapter_dir.exists():
            raise FileNotFoundError(f"Missing final_adapter in {p}")
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")

        run_config_path = run_dir / "run_config.json"
        experiment_name = run_dir.name
        base_model_name = expected_base_model

        if run_config_path.exists():
            with run_config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            experiment_name = cfg.get("experiment_name", experiment_name)
            base_model_name = cfg.get("model_name", expected_base_model)

        if base_model_name != expected_base_model:
            raise ValueError(f"Run {run_dir} was trained from {base_model_name}, not {expected_base_model}")

        runs.append(
            {
                "eval_name": adapter_eval_slug(run_dir.name, experiment_name),
                "run_dir": str(run_dir),
                "adapter_dir": str(adapter_dir),
                "base_model_name": base_model_name,
                "base_model_family": get_model_family(base_model_name),
                "experiment_name": experiment_name,
            }
        )
    return runs


def load_finetuned_model(base_model_name: str, adapter_dir: str):
    tokenizer_source = adapter_dir if (Path(adapter_dir) / "tokenizer_config.json").exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=TRUST_REMOTE_CODE,
        **_hf_kwargs(),
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=TRUST_REMOTE_CODE,
        **_hf_kwargs(),
    )

    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "left"
    return tokenizer, model


def apply_chat_template_or_fallback(tokenizer, prompt_text: str, base_model_name: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    family = get_model_family(base_model_name)
    if family == "smollm":
        return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    return f"USER: {prompt_text}\n\nASSISTANT:"


def build_model_inputs(tokenizer, prompt_text: str, model) -> Dict[str, torch.Tensor]:
    return tokenizer([prompt_text], return_tensors="pt").to(model.device)


def extract_generation(text: str) -> str:
    return normalize_ws(text)


@torch.inference_mode()
def generate_one(model, tokenizer, prompt_text: str, max_new_tokens: int) -> Dict[str, str]:
    model_inputs = build_model_inputs(tokenizer, prompt_text, model)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": REPETITION_PENALTY,
    }
    if DO_SAMPLE:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        })
    else:
        gen_kwargs.update({"do_sample": False})

    generated = model.generate(**model_inputs, **gen_kwargs)
    input_len = model_inputs["input_ids"].shape[1]
    output_ids = generated[0][input_len:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return {"raw_generation": text, "extracted_verbalization": extract_generation(text)}


def cleanup_model(model=None, tokenizer=None):
    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass
        del model

    if tokenizer is not None:
        del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_generation_for_adapter(
    run_info: Dict[str, str],
    run_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    eval_name = run_info["eval_name"]
    adapter_dir = run_info["adapter_dir"]
    base_model_name = run_info["base_model_name"]
    experiment_name = run_info["experiment_name"]

    out_path = output_csv_path(output_dir, eval_name)
    key_cols = ["lang", "split", "category", "eid", "align_key"]

    existing_df = None
    completed_keys = set()

    if out_path.exists() and not args.overwrite_existing:
        print(f"Found existing results at {out_path}")
        existing_df = pd.read_csv(out_path)

        if not existing_df.empty:
            required_cols = set(key_cols + ["generation_status"])
            if required_cols.issubset(existing_df.columns):
                completed_df = existing_df[existing_df["generation_status"] == "ok"].copy()
                completed_keys = set(
                    tuple(x) for x in completed_df[key_cols].drop_duplicates().itertuples(index=False, name=None)
                )

    run_df = run_df.copy()
    run_df["_run_key"] = list(run_df[key_cols].itertuples(index=False, name=None))
    missing_df = run_df[~run_df["_run_key"].isin(completed_keys)].copy()
    run_df.drop(columns=["_run_key"], inplace=True)
    missing_df.drop(columns=["_run_key"], inplace=True)

    if missing_df.empty and existing_df is not None and not args.overwrite_existing:
        print(f"All entries already present in {out_path}")
        return existing_df

    tokenizer, model = load_finetuned_model(base_model_name, adapter_dir)
    new_results = []

    prompt_text_cache: Dict[str, str] = {}
    for prompt in missing_df["prompt"].unique().tolist():
        prompt_text_cache[prompt] = apply_chat_template_or_fallback(tokenizer, prompt, base_model_name)

    sorted_df = missing_df.copy()
    sorted_df["prompt_len_chars"] = sorted_df["prompt"].str.len()
    sorted_df = sorted_df.sort_values(
        ["lang", "split", "num_triples", "prompt_len_chars", "category", "eid"]
    )

    print(
        f"Running {eval_name} on {len(sorted_df)} missing entries"
        + (f" / {len(run_df)} total" if len(sorted_df) != len(run_df) else "")
    )

    try:
        for idx, (_, row) in enumerate(
            tqdm(sorted_df.iterrows(), total=len(sorted_df), desc=f"Generating with {eval_name}"),
            start=1
        ):
            prompt_text = prompt_text_cache[row["prompt"]]
            t0 = time.time()

            try:
                gen = generate_one(model, tokenizer, prompt_text, args.max_new_tokens)
                raw_generation = gen["raw_generation"]
                extracted = gen["extracted_verbalization"]
                status = "ok"
                error = ""
            except Exception as e:
                raw_generation = ""
                extracted = ""
                status = "error"
                error = repr(e)

            new_results.append(
                {
                    **row.drop(labels=["prompt_len_chars"]).to_dict(),
                    "model_name": eval_name,
                    "base_model_name": base_model_name,
                    "base_model_family": get_model_family(base_model_name),
                    "experiment_name": experiment_name,
                    "adapter_dir": adapter_dir,
                    "raw_generation": raw_generation,
                    "extracted_verbalization": extracted,
                    "generation_status": status,
                    "generation_error": error,
                    "latency_sec": round(time.time() - t0, 4),
                    "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                }
            )

            if idx % args.save_every == 0:
                checkpoint_df = pd.DataFrame(new_results)
                if existing_df is not None and not existing_df.empty:
                    merged_df = pd.concat([existing_df, checkpoint_df], ignore_index=True)
                else:
                    merged_df = checkpoint_df

                merged_df = (
                    merged_df.sort_values("timestamp_utc")
                    .drop_duplicates(subset=key_cols, keep="last")
                )
                merged_df.to_csv(out_path, index=False)
                print(f"Saved checkpoint: {out_path} ({idx}/{len(sorted_df)})")

        new_result_df = pd.DataFrame(new_results)
        if existing_df is not None and not existing_df.empty:
            result_df = pd.concat([existing_df, new_result_df], ignore_index=True)
        else:
            result_df = new_result_df

        result_df = (
            result_df.sort_values("timestamp_utc")
            .drop_duplicates(subset=key_cols, keep="last")
        )

        result_df.to_csv(out_path, index=False)
        return result_df

    finally:
        cleanup_model(model, tokenizer)


def main() -> None:
    global LIMIT_PER_SPLIT, OVERWRITE_EXISTING
    args = parse_args()
    LIMIT_PER_SPLIT = args.limit_per_split
    OVERWRITE_EXISTING = args.overwrite_existing

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Visible GPU:", torch.cuda.get_device_name(0))
    print("Base model:", args.base_model)
    print("Model family:", get_model_family(args.base_model))

    df_all, run_df = prepare_dataset(args.data_root)

    print(f"Parsed rows: {len(df_all)}")
    print("\nSummary by language/split:")
    summary = (
        df_all.groupby(["lang", "split"])
        .agg(
            rows=("align_key", "size"),
            unique_instances=("align_key", "nunique"),
            categories=("category", "nunique"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    write_manifests(args.output_dir, run_df, args)
    print(f"\nManifest rows: {len(run_df)}")
    print(
        run_df[["lang", "split", "category", "eid", "size", "num_triples", "num_lexicalisations"]]
        .head(10)
        .to_string(index=False)
    )

    if args.adapter_dirs:
        adapter_runs = normalize_adapter_inputs(args.adapter_dirs, args.base_model)
    else:
        adapter_runs = discover_finetuned_runs(args.finetune_root, args.base_model)

    print("\nFine-tuned runs to evaluate:")
    for run in adapter_runs:
        print(json.dumps(run, ensure_ascii=False, indent=2))

    all_results = []
    for run_info in adapter_runs:
        model_df = run_generation_for_adapter(run_info, run_df, args.output_dir, args)
        all_results.append(model_df)

    results_df = pd.concat(all_results, ignore_index=True)
    combined_path = args.output_dir / f"generations_all_finetuned_{model_slug(args.base_model)}_zeroshot.csv"
    results_df.to_csv(combined_path, index=False)
    print(f"\nCombined results: {combined_path}")

    qc = (
        results_df.groupby(["model_name", "experiment_name", "lang", "split"])
        .agg(
            rows=("align_key", "size"),
            ok=("generation_status", lambda s: (s == "ok").sum()),
            nonempty=("extracted_verbalization", lambda s: s.fillna("").str.len().gt(0).sum()),
            avg_latency_sec=("latency_sec", "mean"),
        )
        .reset_index()
    )
    print("\nQuick sanity check:")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()