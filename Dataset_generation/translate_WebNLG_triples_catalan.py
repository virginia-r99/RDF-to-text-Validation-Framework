# 1: imports + env
import os
import re
import csv
import json
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import torch
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# Pywikibot (Wikidata)
os.environ["PYWIKIBOT_NO_USER_CONFIG"] = "2"
import pywikibot

import warnings
from transformers.utils import logging as hf_logging
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") 
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================
# 2: model IDs + dtypes
# ============================================================
NLLB_ID = "facebook/nllb-200-distilled-1.3B"
MADLAD_ID = "google/madlad400-3b-mt"
SALAMANDRA_ID = "BSC-LT/salamandraTA-7b-instruct"

def pick_dtype():
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32

DTYPE = pick_dtype()
print("DTYPE:", DTYPE, "| CUDA:", torch.cuda.is_available())


# ============================================================
# 3: load models
# ============================================================
print("Loading NLLB...")
nllb_tok = AutoTokenizer.from_pretrained(NLLB_ID, use_fast=True)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID, device_map="auto", dtype=DTYPE)

print("Loading MADLAD...")
madlad_tok = AutoTokenizer.from_pretrained(MADLAD_ID, use_fast=True)
madlad_model = AutoModelForSeq2SeqLM.from_pretrained(MADLAD_ID, device_map="auto", dtype=DTYPE)

print("Loading SALAMANDRA...")
sala_tok = AutoTokenizer.from_pretrained(SALAMANDRA_ID)
sala_model = AutoModelForCausalLM.from_pretrained(
    SALAMANDRA_ID,
    device_map="auto",
    dtype=DTYPE,
)
print("Models loaded.")


# ============================================================
# 4: translators
# ============================================================
@torch.inference_mode()
def translate_nllb(text: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 128) -> str:
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
def translate_madlad(text: str, tgt_lang_2letter: str, max_new_tokens: int = 128) -> str:
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
    max_new_tokens: int = 220,
    num_beams: int = 5,
) -> str:
    text = f"Translate the following text from {source} into {target}.\n{source}: {sentence}\n{target}:"
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


# ============================================================
# 5: voting helpers
# ============================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a).lower(), _norm(b).lower()).ratio()

def majority_vote_one(cands: List[str], threshold: float = 0.90) -> str:
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
    s12 = _sim(t1, t2); s13 = _sim(t1, t3); s23 = _sim(t2, t3)
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
    vote_threshold: float = 0.90,
    max_new_tokens_nllb: int = 128,
    max_new_tokens_madlad: int = 128,
    max_new_tokens_sala: int = 220,
) -> Dict[str, str]:
    t1 = translate_nllb(text, src_lang=src_nllb, tgt_lang=tgt_nllb, max_new_tokens=max_new_tokens_nllb)
    t2 = translate_madlad(text, tgt_lang_2letter=tgt_madlad_2letter, max_new_tokens=max_new_tokens_madlad)
    t3 = translate_salamandra(text, source=source_name, target=target_name, max_new_tokens=max_new_tokens_sala)
    voted = majority_vote_one([t1, t2, t3], threshold=vote_threshold)
    det = vote_details(t1, t2, t3, threshold=vote_threshold)
    return {"nllb": t1, "madlad": t2, "salamandra": t3, "voted": voted, **det}


# ============================================================
# 6: XML utilities
# ============================================================
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

def iter_xml_files(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".xml"):
                yield os.path.join(dirpath, fn)


# ============================================================
# 7: triple parsing + text normalization
# ============================================================
TRIPLE_SPLIT = " | "

def parse_triple_line(line: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in line.split(TRIPLE_SPLIT, 2)]
    if len(parts) != 3:
        parts = [p.strip() for p in line.split("|", 2)]
    if len(parts) != 3:
        raise ValueError(f"Bad triple: {line!r}")
    return parts[0], parts[1], parts[2]

def strip_quotes_and_lang(s: str) -> str:
    s = (s or "").strip()
    m = re.match(r'^"(.*)"(@[a-zA-Z\-]+)?$', s)
    if m:
        return m.group(1).strip()
    return s

def to_wikidata_query_text(token: str) -> str:
    t = strip_quotes_and_lang(token)
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def to_mt_source_text_entity(token: str) -> str:
    return to_wikidata_query_text(token)

def split_camelcase(s: str) -> str:
    return re.sub(r"(?<=[a-záéíóúàèìòùñü])(?=[A-ZÁÉÍÓÚÀÈÌÒÙÑÜ])", " ", s)

def to_mt_source_text_relation(token: str) -> str:
    t = strip_quotes_and_lang(token)
    t = t.replace("_", " ")
    t = split_camelcase(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def to_triple_token(text: str) -> str:
    t = _norm(text)
    t = t.replace(" ", "_")
    t = re.sub(r"_+", "_", t)
    return t


# ============================================================
# 8: Wikidata lookup (EN term -> QID -> labels/aliases in ca)
# ============================================================
WD_LANGS = ("ca",)

def wikidata_connect():
    wd_site = pywikibot.Site("wikidata", "wikidata")
    repo = wd_site.data_repository()
    return repo

def choose_best_wd_search_result(query_en: str, results: List[dict]) -> Optional[str]:
    if not results:
        return None
    qlow = query_en.strip().lower()
    for r in results:
        lab = (r.get("label") or "").strip().lower()
        if lab == qlow:
            return r.get("id")
    return results[0].get("id")

def wikidata_labels_aliases_for_en_query(
    repo,
    query_en: str,
    total: int = 5,
) -> Dict[str, object]:
    out = {"qid": "", "labels": {l: "" for l in WD_LANGS}, "aliases": {l: [] for l in WD_LANGS}}
    try:
        results = repo.search_entities(query_en, language="en", total=total)
        qid = choose_best_wd_search_result(query_en, results)
        if not qid:
            return out
        item = pywikibot.ItemPage(repo, qid)
        item.get()
        out["qid"] = qid
        for l in WD_LANGS:
            out["labels"][l] = item.labels.get(l, "")
            out["aliases"][l] = item.aliases.get(l, []) or []
        return out
    except Exception:
        return out


# ============================================================
# 9: registries (store ALL translations)
# ============================================================
ENTITY_FIELDS = [
    "kind", "lang",
    "source_token", "source_query_text",
    "qid",
    "wikidata_label", "wikidata_aliases_json",
    "mt_nllb", "mt_madlad", "mt_salamandra",
    "vote_threshold",
    "sim_nllb_madlad", "sim_nllb_salamandra", "sim_madlad_salamandra",
    "agree_nllb_madlad", "agree_nllb_salamandra", "agree_madlad_salamandra",
    "mt_voted",
    "final_text", "final_source",
    "processed_at",
]

REL_FIELDS = [
    "kind", "lang",
    "source_token", "source_query_text",
    "mt_nllb", "mt_madlad", "mt_salamandra",
    "vote_threshold",
    "sim_nllb_madlad", "sim_nllb_salamandra", "sim_madlad_salamandra",
    "agree_nllb_madlad", "agree_nllb_salamandra", "agree_madlad_salamandra",
    "mt_voted",
    "final_text", "final_source",
    "processed_at",
]

def ensure_csv(path: str, fields: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

def append_row(path: str, fields: List[str], row: Dict[str, str]):
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writerow({k: row.get(k, "") for k in fields})
        f.flush()

def load_final_map(path: str, kind: str) -> Dict[Tuple[str, str], str]:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("kind") or "") != kind:
                continue
            st = (row.get("source_token") or "")
            lg = (row.get("lang") or "")
            ft = (row.get("final_text") or "")
            if st and lg and ft:
                out[(st, lg)] = ft
    return out


# ============================================================
# 10: language profiles
# ============================================================
LANG_PROFILES = {
    "ca": {
        "name": "Catalan",
        "nllb_tgt": "cat_Latn",
        "madlad_tgt": "ca",
    }
}

# SOURCE IS ENGLISH NOW
SRC_NLLB = "eng_Latn"


# ============================================================
# 11: extract unique entities + relations from ALL XMLs (English triples: modifiedtripleset)
# ============================================================
def extract_en_entities_relations(dataset_root: str) -> Tuple[List[str], List[str]]:
    entities = set()
    relations = set()

    xml_files = sorted(iter_xml_files(dataset_root))
    for xp in tqdm(xml_files, desc="Extracting from XML (EN modifiedtripleset)", unit="file"):
        try:
            tree = ET.parse(xp)
        except Exception:
            continue

        root = tree.getroot()
        entries_parent = root.find("entries")
        if entries_parent is None:
            continue

        for entry in entries_parent.findall("entry"):
            en_node = entry.find("modifiedtripleset")
            if en_node is None:
                continue
            for st in en_node.findall("mtriple"):
                line = (st.text or "").strip()
                if not line:
                    continue
                try:
                    s, p, o = parse_triple_line(line)
                except Exception:
                    continue
                entities.add(s.strip())
                entities.add(o.strip())
                relations.add(p.strip())

    return sorted(entities), sorted(relations)


# ============================================================
# 12: translate entities with Wikidata priority + store all candidates
# ============================================================
DATE_US_NUMERIC_RE = re.compile(r"^\s*(\d{1,2})(\D+)(\d{1,2})(\D+)(\d{2,4})\s*$")

def _has_any_letter(s: str) -> bool:
    return any(ch.isalpha() for ch in (s or ""))

def is_number_symbol_no_letters(raw: str) -> bool:
    raw = (raw or "").strip()
    if not raw:
        return False
    if _has_any_letter(raw):
        return False
    return any(ch.isdigit() for ch in raw)

def maybe_convert_us_date_to_eu(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    m = DATE_US_NUMERIC_RE.fullmatch(raw)
    if not m:
        return None

    mm, sep1, dd, sep2, yyyy = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)

    try:
        mm_i = int(mm)
        dd_i = int(dd)
    except Exception:
        return None

    if mm_i > 12:
        return None
    if not (1 <= mm_i <= 12 and 1 <= dd_i <= 31):
        return None

    return f"{dd}{sep1}{mm}{sep2}{yyyy}"

def detect_passthrough_token_value(token: str) -> Optional[Tuple[str, str]]:
    raw = strip_quotes_and_lang(token).strip()

    converted = maybe_convert_us_date_to_eu(raw)
    if converted is not None:
        return converted, "passthrough_date_us_to_eu"

    if is_number_symbol_no_letters(raw):
        return raw, "passthrough_number_symbols"

    return None

def build_entity_translation_maps(
    entities: List[str],
    entity_csv_path: str,
    *,
    vote_threshold: float = 0.90,
    do_mt_even_if_wikidata: bool = True,
) -> Dict[str, Dict[str, str]]:
    """
    returns: final_map[lang][source_token] = final_text (human text, spaces)
    source_token is the EN token from modifiedtripleset.
    """
    ensure_csv(entity_csv_path, ENTITY_FIELDS)
    repo = wikidata_connect()

    final_resume = load_final_map(entity_csv_path, kind="entity")

    final_map = {l: {} for l in LANG_PROFILES.keys()}
    wd_cache = {}  # query_text -> wd_payload

    for source_token in tqdm(entities, desc="Entities", unit="entity"):

        passthrough = detect_passthrough_token_value(source_token)
        if passthrough is not None:
            final_text_raw, final_source = passthrough
            for lang in LANG_PROFILES.keys():
                if (source_token, lang) in final_resume:
                    final_map[lang][source_token] = final_resume[(source_token, lang)]
                    continue

                final_map[lang][source_token] = final_text_raw
                append_row(entity_csv_path, ENTITY_FIELDS, {
                    "kind": "entity",
                    "lang": lang,
                    "source_token": source_token,
                    "source_query_text": strip_quotes_and_lang(source_token).strip(),
                    "qid": "",
                    "wikidata_label": "",
                    "wikidata_aliases_json": "[]",
                    "mt_nllb": "",
                    "mt_madlad": "",
                    "mt_salamandra": "",
                    "vote_threshold": str(vote_threshold),
                    "sim_nllb_madlad": "",
                    "sim_nllb_salamandra": "",
                    "sim_madlad_salamandra": "",
                    "agree_nllb_madlad": "",
                    "agree_nllb_salamandra": "",
                    "agree_madlad_salamandra": "",
                    "mt_voted": "",
                    "final_text": final_text_raw,
                    "final_source": final_source,
                    "processed_at": datetime.now().isoformat(timespec="seconds"),
                })
            continue

        query_text = to_wikidata_query_text(source_token)

        if query_text not in wd_cache:
            wd_cache[query_text] = wikidata_labels_aliases_for_en_query(repo, query_text)

        wd = wd_cache[query_text]
        qid = wd.get("qid", "") or ""

        for lang, prof in LANG_PROFILES.items():
            if (source_token, lang) in final_resume:
                final_map[lang][source_token] = final_resume[(source_token, lang)]
                continue

            wd_label = (wd.get("labels", {}).get(lang) or "").strip()
            wd_aliases = wd.get("aliases", {}).get(lang) or []
            wd_alias = (wd_aliases[0].strip() if wd_aliases else "")

            mt_src = to_mt_source_text_entity(source_token)
            mt_out = {"nllb": "", "madlad": "", "salamandra": "", "voted": "", **vote_details("", "", "", vote_threshold)}
            if do_mt_even_if_wikidata or (not wd_label and not wd_alias):
                mt_out = translate_with_vote_one(
                    mt_src,
                    src_nllb=SRC_NLLB,
                    tgt_nllb=prof["nllb_tgt"],
                    tgt_madlad_2letter=prof["madlad_tgt"],
                    source_name="English",
                    target_name=prof["name"],
                    vote_threshold=vote_threshold,
                    max_new_tokens_nllb=96,
                    max_new_tokens_madlad=96,
                    max_new_tokens_sala=160,
                )

            if wd_label:
                final_text = wd_label
                final_source = "wikidata_label"
            elif wd_alias:
                final_text = wd_alias
                final_source = "wikidata_alias"
            else:
                final_text = _norm(mt_out.get("voted", ""))
                final_source = "mt_vote"

            final_map[lang][source_token] = final_text

            append_row(entity_csv_path, ENTITY_FIELDS, {
                "kind": "entity",
                "lang": lang,
                "source_token": source_token,
                "source_query_text": query_text,
                "qid": qid,
                "wikidata_label": wd_label,
                "wikidata_aliases_json": json.dumps(wd_aliases, ensure_ascii=False),
                "mt_nllb": mt_out.get("nllb", ""),
                "mt_madlad": mt_out.get("madlad", ""),
                "mt_salamandra": mt_out.get("salamandra", ""),
                "vote_threshold": str(vote_threshold),
                "sim_nllb_madlad": mt_out.get("sim_nllb_madlad", ""),
                "sim_nllb_salamandra": mt_out.get("sim_nllb_salamandra", ""),
                "sim_madlad_salamandra": mt_out.get("sim_madlad_salamandra", ""),
                "agree_nllb_madlad": mt_out.get("agree_nllb_madlad", ""),
                "agree_nllb_salamandra": mt_out.get("agree_nllb_salamandra", ""),
                "agree_madlad_salamandra": mt_out.get("agree_madlad_salamandra", ""),
                "mt_voted": mt_out.get("voted", ""),
                "final_text": final_text,
                "final_source": final_source,
                "processed_at": datetime.now().isoformat(timespec="seconds"),
            })

    return final_map


# ============================================================
# 13: translate relations (3 MT + vote) + store all candidates
# ============================================================
def build_relation_translation_maps(
    relations: List[str],
    rel_csv_path: str,
    *,
    vote_threshold: float = 0.90,
) -> Dict[str, Dict[str, str]]:
    """
    returns: final_map[lang][source_token] = final_text (human text, spaces)
    source_token is the EN relation token from modifiedtripleset.
    """
    ensure_csv(rel_csv_path, REL_FIELDS)

    final_resume = load_final_map(rel_csv_path, kind="relation")
    final_map = {l: {} for l in LANG_PROFILES.keys()}

    for source_token in tqdm(relations, desc="Relations", unit="rel"):
        query_text = to_mt_source_text_relation(source_token)

        for lang, prof in LANG_PROFILES.items():
            if (source_token, lang) in final_resume:
                final_map[lang][source_token] = final_resume[(source_token, lang)]
                continue

            mt_out = translate_with_vote_one(
                query_text,
                src_nllb=SRC_NLLB,
                tgt_nllb=prof["nllb_tgt"],
                tgt_madlad_2letter=prof["madlad_tgt"],
                source_name="English",
                target_name=prof["name"],
                vote_threshold=vote_threshold,
                max_new_tokens_nllb=64,
                max_new_tokens_madlad=64,
                max_new_tokens_sala=120,
            )

            final_text = _norm(mt_out.get("voted", ""))
            final_map[lang][source_token] = final_text

            append_row(rel_csv_path, REL_FIELDS, {
                "kind": "relation",
                "lang": lang,
                "source_token": source_token,
                "source_query_text": query_text,
                "mt_nllb": mt_out.get("nllb", ""),
                "mt_madlad": mt_out.get("madlad", ""),
                "mt_salamandra": mt_out.get("salamandra", ""),
                "vote_threshold": str(vote_threshold),
                "sim_nllb_madlad": mt_out.get("sim_nllb_madlad", ""),
                "sim_nllb_salamandra": mt_out.get("sim_nllb_salamandra", ""),
                "sim_madlad_salamandra": mt_out.get("sim_madlad_salamandra", ""),
                "agree_nllb_madlad": mt_out.get("agree_nllb_madlad", ""),
                "agree_nllb_salamandra": mt_out.get("agree_nllb_salamandra", ""),
                "agree_madlad_salamandra": mt_out.get("agree_madlad_salamandra", ""),
                "mt_voted": mt_out.get("voted", ""),
                "final_text": final_text,
                "final_source": "mt_vote",
                "processed_at": datetime.now().isoformat(timespec="seconds"),
            })

    return final_map


# ============================================================
# 14: rebuild catalan triplesets in XMLs from EN modifiedtripleset
# ============================================================
def ensure_tripleset_node(entry: ET.Element, tag: str, insert_after: Optional[str] = None) -> ET.Element:
    existing = entry.find(tag)
    if existing is not None:
        return existing

    new_node = ET.Element(tag)

    if insert_after:
        after_node = entry.find(insert_after)
        if after_node is not None:
            children = list(entry)
            idx = children.index(after_node)
            entry.insert(idx + 1, new_node)
            return new_node

    # fallback: append near triplesets (after modifiedtripleset if present)
    en = entry.find("modifiedtripleset")
    if en is not None:
        children = list(entry)
        idx = children.index(en)
        entry.insert(idx + 1, new_node)
        return new_node

    entry.append(new_node)
    return new_node

def tripleset_has_triples(node: ET.Element) -> bool:
    return any((c.text or "").strip() for c in list(node))

def clear_children(node: ET.Element):
    for c in list(node):
        node.remove(c)

def rebuild_xml_triplesets(
    dataset_root: str,
    entity_map: Dict[str, Dict[str, str]],
    rel_map: Dict[str, Dict[str, str]],
    *,
    overwrite_existing: bool = False,
):
    """
    ONLY writes <catalantripleset> from <modifiedtripleset>.
    Does not modify other triplesets.
    """
    xml_files = sorted(iter_xml_files(dataset_root))

    for xp in tqdm(xml_files, desc="Writing Catalan triplesets", unit="file"):
        try:
            tree = ET.parse(xp)
        except Exception:
            continue

        root = tree.getroot()
        entries_parent = root.find("entries")
        if entries_parent is None:
            continue

        changed = False

        for entry in entries_parent.findall("entry"):
            en = entry.find("modifiedtripleset")
            if en is None:
                continue

            ca_node = ensure_tripleset_node(entry, "catalantripleset", insert_after="modifiedtripleset")

            if not overwrite_existing and tripleset_has_triples(ca_node):
                continue

            clear_children(ca_node)

            wrote_any = False
            for st in en.findall("mtriple"):
                line = (st.text or "").strip()
                if not line:
                    continue
                try:
                    s_en, p_en, o_en = parse_triple_line(line)
                except Exception:
                    continue

                s_t = entity_map["ca"].get(s_en, to_mt_source_text_entity(s_en))
                p_t = rel_map["ca"].get(p_en, to_mt_source_text_relation(p_en))
                o_t = entity_map["ca"].get(o_en, to_mt_source_text_entity(o_en))

                s_tok = to_triple_token(s_t)
                p_tok = to_triple_token(p_t)
                o_tok = to_triple_token(o_t)

                new_line = f"{s_tok}{TRIPLE_SPLIT}{p_tok}{TRIPLE_SPLIT}{o_tok}"
                el = ET.Element("ctriple")
                el.text = new_line
                ca_node.append(el)
                wrote_any = True

            if wrote_any:
                changed = True

        if changed:
            write_xml_atomic(tree, xp)


# ============================================================
# 15: RUN (set paths)
# ============================================================
DATASET_ROOT = "./WebNLG_ES"  # <-- change

REGISTRY_DIR = "./Registry_triples"
os.makedirs(REGISTRY_DIR, exist_ok=True)

ENTITY_CSV = os.path.join(REGISTRY_DIR, "entity_translations_ca.csv")
REL_CSV    = os.path.join(REGISTRY_DIR, "relation_translations_ca.csv")

# 1) Extract from EN modifiedtripleset
entities, relations = extract_en_entities_relations(DATASET_ROOT)
print("Unique entities:", len(entities))
print("Unique relations:", len(relations))

# 2) Translate + registry
entity_map = build_entity_translation_maps(
    entities,
    ENTITY_CSV,
    vote_threshold=0.90,
    do_mt_even_if_wikidata=True,
)

rel_map = build_relation_translation_maps(
    relations,
    REL_CSV,
    vote_threshold=0.90,
)

print("DONE")