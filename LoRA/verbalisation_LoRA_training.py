#!/usr/bin/env python3

#  python train_lora_webnlg_co.py \
#  --data_root ../../WebNLG_CO_BT \
#  --output_root ./runs_qwen \
#  --model Qwen/Qwen3-4B-Instruct-2507 \
#  --gradient_checkpointing     


import argparse
import csv
import gc
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


LLM_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "CohereLabs/tiny-aya-global",
    "HuggingFaceTB/SmolLM3-3B",
    "BSC-LT/salamandra-2b-instruct",
]

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

EXPERIMENTS = [
    {"name": "EN-gold", "lang": "en", "quality": "all"},
    {"name": "ES-silver", "lang": "es", "quality": "all"},
    {"name": "CA-silver", "lang": "ca", "quality": "all"},
    {"name": "EN-backtrans-from-CA", "lang": "en_bt", "quality": "all"},
    {"name": "ES-silver-filtered", "lang": "es", "quality": "high"},
    {"name": "CA-silver-filtered", "lang": "ca", "quality": "high"},
]


@dataclass
class Example:
    split: str
    bucket: str
    xml_file: str
    category: str
    eid: str
    lid: str
    lang: str
    triples: List[str]
    target: str
    prompt: str
    align_key: str


class MeanPoolingEncoder(nn.Module):
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**toks)
            hidden = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            vecs = summed / counts
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
            all_vecs.append(vecs.cpu().numpy())

            del toks, out, hidden, mask, summed, counts, vecs

        return np.vstack(all_vecs)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


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
        if child.text and normalize_space(child.text):
            triples.append(normalize_space(child.text))
    return triples


def read_lexes(entry: ET.Element, lang: str) -> List[Tuple[str, str]]:
    rows = []
    for lex in entry.findall("lex"):
        if lex.attrib.get("lang") == lang:
            lid = lex.attrib.get("lid", "NA")
            text = normalize_space(lex.text or "")
            if text:
                rows.append((lid, text))
    return rows


def serialize_triples(triples: Sequence[str]) -> str:
    return "\n".join(f"- {t}" for t in triples)


def make_prompt(lang: str, triples: Sequence[str]) -> str:
    return (
        f"Generate one faithful {LANG_NAME[lang]} sentence from the RDF triples below. "
        f"Preserve the facts and do not add unsupported information.\n\n"
        f"RDF triples:\n{serialize_triples(triples)}"
    )


def parse_split(root_dir: Path, split: str, lang: str) -> List[Example]:
    split_dir = root_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    examples: List[Example] = []
    bucket_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    for bucket_dir in bucket_dirs:
        for xml_path in sorted(bucket_dir.glob("*.xml")):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for entry in root.findall(".//entry"):
                category = entry.attrib.get("category", "NA")
                eid = entry.attrib.get("eid", "NA")
                triples = read_triples(entry, lang)
                if not triples:
                    continue
                lex_rows = read_lexes(entry, lang)
                if not lex_rows:
                    continue
                prompt = make_prompt(lang, triples)
                for lid, target in lex_rows:
                    align_key = f"{split}|{bucket_dir.name}|{xml_path.name}|{eid}|{lid}|{lang}"
                    examples.append(
                        Example(
                            split=split,
                            bucket=bucket_dir.name,
                            xml_file=xml_path.name,
                            category=category,
                            eid=eid,
                            lid=lid,
                            lang=lang,
                            triples=triples,
                            target=target,
                            prompt=prompt,
                            align_key=align_key,
                        )
                    )
    return examples


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def knee_threshold_from_bottom_tail(scores: List[float], tail_fraction: float = 0.10) -> float:
    vals = np.array(sorted(float(x) for x in scores), dtype=np.float32)
    if len(vals) == 0:
        raise ValueError("Empty score list")
    if len(vals) < 10:
        return float(np.quantile(vals, tail_fraction))

    tail_n = max(3, int(math.ceil(len(vals) * tail_fraction)))
    tail = vals[:tail_n]
    x = np.linspace(0.0, 1.0, num=len(tail), dtype=np.float32)
    y_min, y_max = float(tail.min()), float(tail.max())
    if abs(y_max - y_min) < 1e-8:
        return float(tail[-1])
    y = (tail - y_min) / (y_max - y_min)
    line = x
    distances = y - line
    knee_idx = int(np.argmax(distances))
    return float(tail[knee_idx])


def build_quality_registry(
    examples: List[Example],
    aligned_en_examples: List[Example],
    output_csv: Path,
    embedding_model: str,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[str, float], float]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    by_entry_en: Dict[Tuple[str, str, str, str], List[Example]] = {}
    for ex in aligned_en_examples:
        k = (ex.split, ex.bucket, ex.xml_file, ex.eid)
        by_entry_en.setdefault(k, []).append(ex)

    rows = []
    src_texts = []
    ref_texts = []
    spans = []
    for ex in examples:
        k = (ex.split, ex.bucket, ex.xml_file, ex.eid)
        refs = by_entry_en.get(k, [])
        if not refs:
            continue
        start = len(ref_texts)
        src_texts.append(ex.target)
        ref_texts.extend([r.target for r in refs])
        end = len(ref_texts)
        spans.append((ex, start, end))

    if not spans:
        raise ValueError(f"No aligned English references found for registry: {output_csv}")

    encoder = MeanPoolingEncoder(embedding_model, cache_dir=cache_dir)
    src_emb = encoder.encode(src_texts)
    ref_emb = encoder.encode(ref_texts)

    src_to_score: Dict[str, float] = {}
    for i, (ex, start, end) in enumerate(spans):
        sims = cosine_matrix(src_emb[i:i + 1], ref_emb[start:end])[0]
        best = float(np.max(sims))
        src_to_score[ex.align_key] = best
        rows.append({
            "align_key": ex.align_key,
            "split": ex.split,
            "bucket": ex.bucket,
            "xml_file": ex.xml_file,
            "category": ex.category,
            "eid": ex.eid,
            "lid": ex.lid,
            "lang": ex.lang,
            "quality_score": best,
        })

    threshold = knee_threshold_from_bottom_tail(
        [r["quality_score"] for r in rows],
        tail_fraction=0.25,
    )
    for row in rows:
        row["threshold"] = threshold
        row["keep_high_quality"] = int(row["quality_score"] > threshold)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "align_key", "split", "bucket", "xml_file", "category", "eid", "lid", "lang",
                "quality_score", "threshold", "keep_high_quality",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return src_to_score, threshold


def load_quality_registry(registry_csv: Path) -> Tuple[Dict[str, float], Dict[str, int], float]:
    scores: Dict[str, float] = {}
    keep: Dict[str, int] = {}
    threshold = 0.0
    with registry_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["align_key"]] = float(row["quality_score"])
            keep[row["align_key"]] = int(row["keep_high_quality"])
            threshold = float(row["threshold"])
    return scores, keep, threshold


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cleanup_training_objects(trainer=None, model=None, tokenizer=None):
    try:
        if trainer is not None:
            if hasattr(trainer, "model") and trainer.model is not None:
                try:
                    trainer.model.cpu()
                except Exception:
                    pass
            if hasattr(trainer, "accelerator"):
                try:
                    trainer.accelerator.free_memory()
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if model is not None:
            model.cpu()
    except Exception:
        pass

    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def find_lora_target_modules(model: nn.Module) -> List[str]:
    names = set()
    linear_classes = (nn.Linear,)
    try:
        import bitsandbytes as bnb
        linear_classes = linear_classes + (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except Exception:
        pass

    for name, module in model.named_modules():
        if isinstance(module, linear_classes):
            leaf = name.split(".")[-1]
            if leaf != "lm_head":
                names.add(leaf)

    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj",
        "Wqkv", "out_proj", "fc1", "fc2",
    ]
    picked = [n for n in preferred if n in names]
    return picked if picked else sorted(names)


class SaveAdapterConfigCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        return control


def sanitize_run_name(model_name: str, experiment_name: str) -> str:
    model_part = model_name.replace("/", "__")
    exp_part = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_name)
    return f"{model_part}__{exp_part}"


def filter_examples(examples: List[Example], keep_map: Dict[str, int]) -> List[Example]:
    return [ex for ex in examples if keep_map.get(ex.align_key, 0) == 1]


def prepare_quality_registry_if_needed(
    lang: str,
    split: str,
    split_examples: List[Example],
    split_en_examples: List[Example],
    quality_root: Path,
    embedding_model: str,
    cache_dir: Optional[str],
) -> Tuple[Dict[str, int], float]:
    registry_csv = quality_root / f"quality_registry_{split}_{lang}.csv"
    if registry_csv.exists():
        _, keep_map, threshold = load_quality_registry(registry_csv)
        return keep_map, threshold

    build_quality_registry(
        examples=split_examples,
        aligned_en_examples=split_en_examples,
        output_csv=registry_csv,
        embedding_model=embedding_model,
        cache_dir=cache_dir,
    )
    _, keep_map, threshold = load_quality_registry(registry_csv)
    return keep_map, threshold


def build_hf_dataset(examples: List[Example], model_name: str) -> Dataset:
    rows = []
    is_smollm = "smollm" in model_name.lower()

    for ex in examples:
        prompt_messages = [{"role": "user", "content": ex.prompt}]

        if is_smollm:
            prompt_messages.insert(0, {"role": "system", "content": "/no_think"})

        rows.append(
            {
                "prompt": prompt_messages,
                "completion": [
                    {"role": "assistant", "content": ex.target}
                ],
                "align_key": ex.align_key,
                "lang": ex.lang,
                "bucket": ex.bucket,
                "category": ex.category,
            }
        )

    return Dataset.from_list(rows)


def estimate_warmup_steps(
    n_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    warmup_ratio: float,
) -> int:
    effective_batch = max(1, per_device_train_batch_size * gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(n_examples / effective_batch))
    total_steps = max(1, int(math.ceil(steps_per_epoch * num_train_epochs)))
    if warmup_ratio <= 0:
        return 0
    return max(1, int(round(total_steps * warmup_ratio)))


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if m:
            checkpoints.append((int(m.group(1)), p))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def is_experiment_already_finished(output_dir: Path) -> bool:
    final_adapter_dir = output_dir / "final_adapter"
    final_metrics_path = output_dir / "final_metrics.json"
    run_config_path = output_dir / "run_config.json"

    return (
        final_adapter_dir.exists()
        and final_adapter_dir.is_dir()
        and final_metrics_path.exists()
        and run_config_path.exists()
    )


def load_existing_metrics(output_dir: Path) -> Dict[str, float]:
    metrics_path = output_dir / "final_metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_single_experiment(
    model_name: str,
    experiment: Dict[str, str],
    parsed_data: Dict[str, Dict[str, List[Example]]],
    output_root: Path,
    quality_root: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    lang = experiment["lang"]
    quality = experiment["quality"]
    run_name = sanitize_run_name(model_name, experiment["name"])
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = list(parsed_data[lang]["train"])
    dev_examples = list(parsed_data[lang]["dev"])

    metadata = {
        "experiment_name": experiment["name"],
        "model_name": model_name,
        "lang": lang,
        "quality": quality,
        "n_train_before_filter": len(train_examples),
        "n_dev_before_filter": len(dev_examples),
    }

    if quality == "high":
        train_keep_map, train_threshold = prepare_quality_registry_if_needed(
            lang=lang,
            split="train",
            split_examples=train_examples,
            split_en_examples=parsed_data["en"]["train"],
            quality_root=quality_root,
            embedding_model=args.embedding_model,
            cache_dir=args.cache_dir,
        )
        train_examples = filter_examples(train_examples, train_keep_map)
        metadata["quality_threshold_train"] = train_threshold

        if args.filter_dev:
            dev_keep_map, dev_threshold = prepare_quality_registry_if_needed(
                lang=lang,
                split="dev",
                split_examples=dev_examples,
                split_en_examples=parsed_data["en"]["dev"],
                quality_root=quality_root,
                embedding_model=args.embedding_model,
                cache_dir=args.cache_dir,
            )
            dev_examples = filter_examples(dev_examples, dev_keep_map)
            metadata["quality_threshold_dev"] = dev_threshold

    metadata["n_train_after_filter"] = len(train_examples)
    metadata["n_dev_after_filter"] = len(dev_examples)

    if len(train_examples) == 0:
        raise ValueError(f"No training examples after filtering for {experiment['name']}")
    if len(dev_examples) == 0:
        raise ValueError(f"No dev examples after filtering for {experiment['name']}")

    final_adapter_dir = output_dir / "final_adapter"

    # Skip fully completed runs
    if is_experiment_already_finished(output_dir):
        print(f"Skipping {experiment['name']} because final adapter already exists: {final_adapter_dir}")
        existing_metrics = load_existing_metrics(output_dir)
        return {
            "status": "skipped_already_trained",
            "experiment_name": experiment["name"],
            "output_dir": str(output_dir),
            "final_adapter_dir": str(final_adapter_dir),
            "metrics": existing_metrics,
            "metadata": metadata,
        }

    latest_checkpoint = find_latest_checkpoint(output_dir)
    if latest_checkpoint is not None:
        print(f"Resuming {experiment['name']} from checkpoint: {latest_checkpoint}")
    else:
        print(f"Starting fresh training for {experiment['name']}")

    dtype = get_torch_dtype()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = find_lora_target_modules(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    train_dataset = build_hf_dataset(train_examples, model_name)
    eval_dataset = build_hf_dataset(dev_examples, model_name)

    warmup_steps = estimate_warmup_steps(
        n_examples=len(train_examples),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
    )

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        seed=args.seed,
        max_length=args.max_length,
        completion_only_loss=True,
        assistant_only_loss=False,
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), SaveAdapterConfigCallback()],
    )

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **metadata,
                "target_modules": target_modules,
                "training_args": {
                    "max_length": args.max_length,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "per_device_eval_batch_size": args.per_device_eval_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "num_train_epochs": args.num_train_epochs,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "warmup_ratio_requested": args.warmup_ratio,
                    "warmup_steps_effective": warmup_steps,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "embedding_model": args.embedding_model,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    train_result = trainer.train(
        resume_from_checkpoint=str(latest_checkpoint) if latest_checkpoint is not None else None
    )

    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    metrics = trainer.evaluate()
    if train_result is not None and hasattr(train_result, "metrics"):
        for k, v in train_result.metrics.items():
            metrics[f"train_{k}" if not k.startswith("train_") else k] = v

    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    result = {
        "status": "ok_resumed" if latest_checkpoint is not None else "ok_trained",
        "experiment_name": experiment["name"],
        "output_dir": str(output_dir),
        "final_adapter_dir": str(final_adapter_dir),
        "metrics": metrics,
        "metadata": metadata,
    }

    del trainer, model, tokenizer, train_dataset, eval_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to WebNLG_CO root")
    parser.add_argument("--output_root", type=str, required=True, help="Directory where all runs will be saved")
    parser.add_argument("--model", type=str, required=True, choices=LLM_MODELS)
    parser.add_argument("--embedding_model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter_dev", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_root = output_root / "quality_registries"
    quality_root.mkdir(parents=True, exist_ok=True)

    print("Parsing datasets once...")
    parsed_data = {
        "en": {
            "train": parse_split(data_root, "train", "en"),
            "dev": parse_split(data_root, "dev", "en"),
        },
        "es": {
            "train": parse_split(data_root, "train", "es"),
            "dev": parse_split(data_root, "dev", "es"),
        },
        "ca": {
            "train": parse_split(data_root, "train", "ca"),
            "dev": parse_split(data_root, "dev", "ca"),
        },
        "en_bt": {
            "train": parse_split(data_root, "train", "en_bt"),
            "dev": parse_split(data_root, "dev", "en_bt"),
        },
    }

    summary = []
    for experiment in EXPERIMENTS:
        print(f"\n===== Running {experiment['name']} | model={args.model} =====")
        result = run_single_experiment(
            model_name=args.model,
            experiment=experiment,
            parsed_data=parsed_data,
            output_root=output_root,
            quality_root=quality_root,
            args=args,
        )
        summary.append(
            {
                "experiment_name": result["experiment_name"],
                "status": result["status"],
                "output_dir": result["output_dir"],
                "final_adapter_dir": result["final_adapter_dir"],
                "eval_loss": result["metrics"].get("eval_loss") if result.get("metrics") else None,
                "n_train_after_filter": result["metadata"].get("n_train_after_filter"),
                "n_dev_after_filter": result["metadata"].get("n_dev_after_filter"),
            }
        )
        print(json.dumps(summary[-1], indent=2))

    with (output_root / f"summary__{args.model.replace('/', '__')}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nAll experiments finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()