# CELL 2: imports + model IDs + dtypes
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import re
import csv
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple

import torch
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

import os
import warnings
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1") 
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------------
# Model IDs
# -----------------------------
NLLB_ID = "facebook/nllb-200-distilled-1.3B"
MADLAD_ID = "google/madlad400-3b-mt"
SALAMANDRA_ID = "BSC-LT/salamandraTA-7b-instruct"

def pick_dtype():
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32

def pick_sala_dtype():
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32

DTYPE = pick_dtype()
SALA_DTYPE = pick_sala_dtype()

print("DTYPE:", DTYPE, "| SALA_DTYPE:", SALA_DTYPE, "| CUDA:", torch.cuda.is_available())

# CELL 3: load models
from collections import Counter

def report_model_placement(name, model):
    print(f"\n--- {name} placement ---")

    # Device map
    if hasattr(model, "hf_device_map"):
        print("hf_device_map:", model.hf_device_map)

    # Show embedding device (good proxy for input device)
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            print("embedding device:", emb.weight.device)
            print("embedding dtype:", emb.weight.dtype)
    except Exception:
        pass

    # Count parameters per device
    device_counter = Counter(str(p.device) for p in model.parameters())
    print("param device counts:", dict(device_counter))


print("Loading NLLB...")
nllb_tok = AutoTokenizer.from_pretrained(NLLB_ID, use_fast=True)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
    NLLB_ID,
    device_map="auto",
    dtype=DTYPE,
)
report_model_placement("NLLB", nllb_model)


print("\nLoading MADLAD...")
madlad_tok = AutoTokenizer.from_pretrained(MADLAD_ID, use_fast=True)
madlad_model = AutoModelForSeq2SeqLM.from_pretrained(
    MADLAD_ID,
    device_map="auto",
    dtype=DTYPE,
)
report_model_placement("MADLAD", madlad_model)


print("\nLoading SALAMANDRA...")
sala_tok = AutoTokenizer.from_pretrained(SALAMANDRA_ID)
sala_model = AutoModelForCausalLM.from_pretrained(
    SALAMANDRA_ID,
    device_map="auto",
    dtype=SALA_DTYPE,
)
report_model_placement("SALAMANDRA", sala_model)

print("Models loaded.")

# CELL 4: translators
@torch.inference_mode()
def translate_nllb(text: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 256) -> str:
    nllb_tok.src_lang = src_lang
    forced_bos_token_id = nllb_tok.convert_tokens_to_ids(tgt_lang)

    inputs = nllb_tok(text, return_tensors="pt", truncation=True).to(nllb_model.device)
    gen = nllb_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        do_sample=False,
    )
    return nllb_tok.decode(gen[0], skip_special_tokens=True).strip()

@torch.inference_mode()
def translate_madlad(text: str, tgt_lang_2letter: str, max_new_tokens: int = 256) -> str:
    prompt = f"<2{tgt_lang_2letter}> {text}"
    inputs = madlad_tok(prompt, return_tensors="pt", truncation=True).to(madlad_model.device)
    gen = madlad_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        do_sample=False,
    )
    return madlad_tok.decode(gen[0], skip_special_tokens=True).strip()

@torch.inference_mode()
def translate_salamandra(
    sentence: str,
    source: str,
    target: str,
    max_new_tokens: int = 400,
    num_beams: int = 5,
) -> str:
    text = f"Translate the following text from {source} into {target}.\n{source}: {sentence} \n{target}:"
    message = [{"role": "user", "content": text}]
    date_string = datetime.today().strftime("%Y-%m-%d")

    prompt = sala_tok.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        date_string=date_string,
    )

    inputs = sala_tok.encode(prompt, add_special_tokens=False, return_tensors="pt")
    input_length = inputs.shape[1]

    outputs = sala_model.generate(
        input_ids=inputs.to(sala_model.device),
        max_new_tokens=max_new_tokens,
        early_stopping=True,
        num_beams=num_beams,
    )

    return sala_tok.decode(outputs[0, input_length:], skip_special_tokens=True).strip()

# CELL 5: voting helpers
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a).lower(), _norm(b).lower()).ratio()

def majority_vote_one(cands, threshold: float = 0.90) -> str:
    cands = [_norm(c) for c in cands if _norm(c)]
    if not cands:
        return ""

    clusters = []
    for c in cands:
        for cl in clusters:
            if _sim(c, cl[0]) >= threshold:
                cl.append(c)
                break
        else:
            clusters.append([c])

    clusters.sort(key=len, reverse=True)

    if len(clusters[0]) >= 2:
        cl = clusters[0]
        return max(cl, key=lambda s: sum(_sim(s, t) for t in cl) / len(cl))

    return max(cands, key=lambda s: sum(_sim(s, t) for t in cands) / len(cands))

def vote_details(t1: str, t2: str, t3: str, threshold: float) -> Dict[str, str]:
    s12 = _sim(t1, t2)
    s13 = _sim(t1, t3)
    s23 = _sim(t2, t3)
    return {
        "sim_nllb_madlad": f"{s12:.4f}",
        "sim_nllb_salamandra": f"{s13:.4f}",
        "sim_madlad_salamandra": f"{s23:.4f}",
        "agree_nllb_madlad": str(int(s12 >= threshold)),
        "agree_nllb_salamandra": str(int(s13 >= threshold)),
        "agree_madlad_salamandra": str(int(s23 >= threshold)),
    }

def translate_with_vote_one(
    text: str,
    *,
    src_nllb: str,
    tgt_nllb: str,
    tgt_madlad_2letter: str,
    source_name: str,
    target_name: str,
    max_new_tokens_nllb: int = 256,
    max_new_tokens_madlad: int = 256,
    max_new_tokens_sala: int = 400,
    vote_threshold: float = 0.90,
) -> Dict[str, str]:
    t1 = translate_nllb(text, src_lang=src_nllb, tgt_lang=tgt_nllb, max_new_tokens=max_new_tokens_nllb)
    t2 = translate_madlad(text, tgt_lang_2letter=tgt_madlad_2letter, max_new_tokens=max_new_tokens_madlad)
    t3 = translate_salamandra(text, source=source_name, target=target_name, max_new_tokens=max_new_tokens_sala)
    voted = majority_vote_one([t1, t2, t3], threshold=vote_threshold)
    det = vote_details(t1, t2, t3, threshold=vote_threshold)
    return {"nllb": t1, "madlad": t2, "salamandra": t3, "voted": voted, **det}

# CELL 6: XML utilities
def find_english_lex(entry_el: ET.Element) -> list[Tuple[ET.Element, str]]:
    # Collect all <lex lang="en"> (any lid); if none, return empty list
    out = []
    for lex in entry_el.findall("lex"):
        if lex.get("lang") == "en":
            out.append((lex, lex.get("lid") or "Id1"))
    return out

def has_catalan_lex(entry_el: ET.Element, lid: str) -> Optional[ET.Element]:
    for lex in entry_el.findall("lex"):
        if lex.get("lang") == "ca" and (lex.get("lid") or "") == lid:
            return lex
    return None

def insert_catalan_lex(entry_el: ET.Element, lid: str, ca_text: str) -> ET.Element:
    new_lex = ET.Element("lex", {"lang": "ca", "lid": lid})
    new_lex.text = ca_text

    lex_nodes = entry_el.findall("lex")
    if lex_nodes:
        last_lex = lex_nodes[-1]
        children = list(entry_el)
        idx = children.index(last_lex)
        entry_el.insert(idx + 1, new_lex)
    else:
        entry_el.append(new_lex)

    return new_lex

def indent_xml(elem: ET.Element, level: int = 0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def write_xml_atomic(tree: ET.ElementTree, path: str):
    tmp_path = path + ".tmp"
    root = tree.getroot()
    indent_xml(root, 0)
    tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
    os.replace(tmp_path, path)

# CELL 7: CSV resume utilities
CSV_FIELDS = [
    "status",
    "processed_at",
    "xml_path",
    "category",
    "eid",
    "source_lid",
    "source_en",
    "ca_nllb",
    "ca_madlad",
    "ca_salamandra",
    "vote_threshold",
    "sim_nllb_madlad",
    "sim_nllb_salamandra",
    "sim_madlad_salamandra",
    "agree_nllb_madlad",
    "agree_nllb_salamandra",
    "agree_madlad_salamandra",
    "voted_ca",
    "final_ca",
    "error",
]

def ensure_csv_header(csv_path: str):
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

def append_csv_row(csv_path: str, row: Dict[str, str]):
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        f.flush()

def load_processed_set(csv_path: str) -> set:
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            st = (row.get("status") or "").strip()
            if st in {"OK", "ALREADY_PRESENT"}:
                done.add((row.get("xml_path") or "", row.get("eid") or "", row.get("source_lid") or ""))
    return done

# CELL 8: file discovery
def iter_xml_files(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".xml"):
                yield os.path.join(dirpath, fn)

# CELL 9: main processing with tqdm + prints
def process_dataset(
    dataset_root: str,
    csv_path: str,
    *,
    vote_threshold: float = 0.90,
    src_nllb: str = "eng_Latn",
    tgt_nllb: str = "cat_Latn",
    tgt_madlad_2letter: str = "ca",
    source_name: str = "English",
    target_name: str = "Catalan",
    max_new_tokens_nllb: int = 256,
    max_new_tokens_madlad: int = 256,
    max_new_tokens_sala: int = 400,
    verbose_every: int = 200,
):
    ensure_csv_header(csv_path)
    processed = load_processed_set(csv_path)

    xml_files = sorted(iter_xml_files(dataset_root))
    print(f"Found {len(xml_files)} XML files under: {dataset_root}")
    print(f"CSV registry: {csv_path}")
    print(f"Already processed entries (OK/ALREADY_PRESENT): {len(processed)}")

    # First pass: count entries for a global progress bar
    total_entries = 0
    for xp in xml_files:
        try:
            t = ET.parse(xp)
            ep = t.getroot().find("entries")
            if ep is not None:
                total_entries += len(ep.findall("entry"))
        except Exception:
            pass
    print(f"Total <entry> elements (approx): {total_entries}")

    pbar = tqdm(total=total_entries, desc="Entries", unit="entry")
    n_ok = n_skip = n_err = 0
    seen = 0

    for xml_path in tqdm(xml_files, desc="XML files", unit="file"):
        try:
            tree = ET.parse(xml_path)
        except Exception as e:
            n_err += 1
            append_csv_row(csv_path, {
                "status": "FILE_PARSE_ERROR",
                "processed_at": datetime.now().isoformat(timespec="seconds"),
                "xml_path": xml_path,
                "error": repr(e),
            })
            continue

        root = tree.getroot()
        entries_parent = root.find("entries")
        if entries_parent is None:
            n_err += 1
            append_csv_row(csv_path, {
                "status": "NO_ENTRIES_NODE",
                "processed_at": datetime.now().isoformat(timespec="seconds"),
                "xml_path": xml_path,
                "error": "Missing <entries> root child",
            })
            continue

        entries = entries_parent.findall("entry")

        for entry in entries:
            category = entry.get("category", "")
            eid = entry.get("eid", "")

            # Translate ALL verbalisations: all <lex lang="en" ...>
            en_lexes = find_english_lex(entry)
            if not en_lexes:
                # still advance the progress bar for this entry
                seen += 1
                pbar.update(1)
                n_err += 1
                append_csv_row(csv_path, {
                    "status": "NO_ENGLISH_LEX",
                    "processed_at": datetime.now().isoformat(timespec="seconds"),
                    "xml_path": xml_path,
                    "category": category,
                    "eid": eid,
                    "error": "No <lex lang='en'> found",
                })
                continue

            # We count one progress tick per entry (not per verbalisation), as in the original.
            seen += 1
            pbar.update(1)

            for en_lex, lid_used in en_lexes:
                lid_used = lid_used or "Id1"
                key = (xml_path, eid, lid_used)

                if key in processed:
                    n_skip += 1
                    continue

                source_en = (en_lex.text or "").strip()
                if not source_en:
                    n_err += 1
                    append_csv_row(csv_path, {
                        "status": "EMPTY_ENGLISH_LEX",
                        "processed_at": datetime.now().isoformat(timespec="seconds"),
                        "xml_path": xml_path,
                        "category": category,
                        "eid": eid,
                        "source_lid": lid_used,
                        "error": "Empty <lex lang='en'> text",
                    })
                    continue

                existing_ca = has_catalan_lex(entry, lid_used)
                if existing_ca is not None and (existing_ca.text or "").strip():
                    n_skip += 1
                    final_ca = (existing_ca.text or "").strip()
                    append_csv_row(csv_path, {
                        "status": "ALREADY_PRESENT",
                        "processed_at": datetime.now().isoformat(timespec="seconds"),
                        "xml_path": xml_path,
                        "category": category,
                        "eid": eid,
                        "source_lid": lid_used,
                        "source_en": source_en,
                        "final_ca": final_ca,
                    })
                    processed.add(key)
                    continue

                # Translate + vote
                try:
                    out = translate_with_vote_one(
                        source_en,
                        src_nllb=src_nllb,
                        tgt_nllb=tgt_nllb,
                        tgt_madlad_2letter=tgt_madlad_2letter,
                        source_name=source_name,
                        target_name=target_name,
                        max_new_tokens_nllb=max_new_tokens_nllb,
                        max_new_tokens_madlad=max_new_tokens_madlad,
                        max_new_tokens_sala=max_new_tokens_sala,
                        vote_threshold=vote_threshold,
                    )
                    final_ca = _norm(out["voted"])

                    if seen % verbose_every == 1:
                        print(f"\nTranslating: file={os.path.relpath(xml_path, dataset_root)} | eid={eid} | cat={category} | lid={lid_used}")
                        print("EN: ", source_en)
                        print("CA: " + final_ca)

                except Exception as e:
                    n_err += 1
                    append_csv_row(csv_path, {
                        "status": "TRANSLATION_ERROR",
                        "processed_at": datetime.now().isoformat(timespec="seconds"),
                        "xml_path": xml_path,
                        "category": category,
                        "eid": eid,
                        "source_lid": lid_used,
                        "source_en": source_en,
                        "vote_threshold": str(vote_threshold),
                        "error": repr(e),
                    })
                    continue

                # Insert + save immediately
                try:
                    insert_catalan_lex(entry, lid_used, final_ca)
                    write_xml_atomic(tree, xml_path)
                except Exception as e:
                    n_err += 1
                    append_csv_row(csv_path, {
                        "status": "XML_WRITE_ERROR",
                        "processed_at": datetime.now().isoformat(timespec="seconds"),
                        "xml_path": xml_path,
                        "category": category,
                        "eid": eid,
                        "source_lid": lid_used,
                        "source_en": source_en,
                        "ca_nllb": out.get("nllb", ""),
                        "ca_madlad": out.get("madlad", ""),
                        "ca_salamandra": out.get("salamandra", ""),
                        "vote_threshold": str(vote_threshold),
                        "sim_nllb_madlad": out.get("sim_nllb_madlad", ""),
                        "sim_nllb_salamandra": out.get("sim_nllb_salamandra", ""),
                        "sim_madlad_salamandra": out.get("sim_madlad_salamandra", ""),
                        "agree_nllb_madlad": out.get("agree_nllb_madlad", ""),
                        "agree_nllb_salamandra": out.get("agree_nllb_salamandra", ""),
                        "agree_madlad_salamandra": out.get("agree_madlad_salamandra", ""),
                        "voted_ca": out.get("voted", ""),
                        "final_ca": final_ca,
                        "error": repr(e),
                    })
                    continue

                # Log OK and mark processed
                n_ok += 1
                append_csv_row(csv_path, {
                    "status": "OK",
                    "processed_at": datetime.now().isoformat(timespec="seconds"),
                    "xml_path": xml_path,
                    "category": category,
                    "eid": eid,
                    "source_lid": lid_used,
                    "source_en": source_en,
                    "ca_nllb": out.get("nllb", ""),
                    "ca_madlad": out.get("madlad", ""),
                    "ca_salamandra": out.get("salamandra", ""),
                    "vote_threshold": str(vote_threshold),
                    "sim_nllb_madlad": out.get("sim_nllb_madlad", ""),
                    "sim_nllb_salamandra": out.get("sim_nllb_salamandra", ""),
                    "sim_madlad_salamandra": out.get("sim_madlad_salamandra", ""),
                    "agree_nllb_madlad": out.get("agree_nllb_madlad", ""),
                    "agree_nllb_salamandra": out.get("agree_nllb_salamandra", ""),
                    "agree_madlad_salamandra": out.get("agree_madlad_salamandra", ""),
                    "voted_ca": out.get("voted", ""),
                    "final_ca": final_ca,
                    "error": "",
                })
                processed.add(key)

                # Optional periodic prints
                if n_ok % verbose_every == 0:
                    print(f"\nProgress summary: OK={n_ok} | SKIP={n_skip} | ERR={n_err}")

    pbar.close()
    print(f"\nDONE. OK={n_ok} | SKIP={n_skip} | ERR={n_err} | total_seen={seen}")

# CELL 10: run
# Set these two paths and run.

DATASET_ROOT = "./WebNLG_ES"               # <- change
CSV_OUT = "./Registry_translations/registry_webnlg_en_ca.csv"    # <- change

process_dataset(
    DATASET_ROOT,
    CSV_OUT,
    vote_threshold=0.90,
    src_nllb="eng_Latn",
    tgt_nllb="cat_Latn",
    tgt_madlad_2letter="ca",
    source_name="English",
    target_name="Catalan",
    max_new_tokens_nllb=512,
    max_new_tokens_madlad=512,
    max_new_tokens_sala=512,
    verbose_every=50,  # prints every N OKs, and prints the EN sample every N entries
)

