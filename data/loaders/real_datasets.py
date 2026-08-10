"""
Real Dataset Loaders for VulMorph-Fed (plan.md §4).

Implements loaders for:
  - Devign   (HuggingFace: DetectVul/devign)
  - PrimeVul (HuggingFace: starsofchance/PrimeVul)
  - BigVul   (GitHub CSV: ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset)
  - DiverseVul (GitHub JSON: wagner-group/diversevul)

Graph construction
------------------
Full CPG extraction requires a heavyweight external tool (Joern). To keep the
framework self-contained and fully reproducible, we build a **lightweight
lexical dependence graph** per function:

  - nodes    : the first `max_tokens` lexical tokens of the function;
  - NCS edges: sequential next-token edges (natural code sequence);
  - DD edges : def-use proxy edges linking successive occurrences of the
               same identifier (a data-dependence approximation).

Each node receives a *morphological type* from the taxonomy in
`data/morphology.py` via deterministic, context-aware rules (documented
below in `classify_token`). This is the exact implementation of the
abstraction function phi_type reported in the manuscript.

The classification is done at the finest (|T|=32) granularity and projected
down to the 16- and 8-type taxonomies, so all taxonomy sizes share one rule
set and are strictly nested.
"""

import os
import re
import json
import hashlib
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch_geometric.data import Data, Dataset

from data.morphology import get_taxonomy

# ── Constants ────────────────────────────────────────────────────────────────

CACHE_DIR = Path(".cache/datasets_v2")

# API allow-lists for the fine-grained (|T|=32) classification.
from data.loaders.api_classes import API_CLASSES

TYPE_KEYWORDS = {
    "int", "char", "float", "double", "long", "short", "unsigned", "signed",
    "void", "size_t", "ssize_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "bool",
}

C_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "default", "break",
    "continue", "goto", "return", "sizeof", "struct", "union", "enum",
    "typedef", "static", "const", "extern", "register", "volatile", "inline",
} | TYPE_KEYWORDS

# Projection from the fine-grained 32-type taxonomy to 16 and 8 types.
PROJECT_32_TO_16 = {
    "MEMORY_ALLOC": "MEMORY_ALLOC", "MEMORY_REALLOC": "MEMORY_ALLOC",
    "MEMORY_FREE": "MEMORY_FREE",
    "MEMORY_COPY": "MEMORY_COPY", "MEMORY_SET": "MEMORY_COPY",
    "STRING_COPY": "STRING_OP", "STRING_CONCAT": "STRING_OP",
    "STRING_FORMAT": "STRING_OP", "STRING_LENGTH": "STRING_OP",
    "IO_CALL": "CALL_SITE",
    "ARRAY_INDEX": "ARRAY_INDEX",
    "PTR_DEREF": "PTR_DEREF", "ADDR_OF": "PTR_DEREF",
    "FIELD_ACCESS": "FIELD_ACCESS", "CAST": "PTR_DEREF",
    "LOOP_FOR": "LOOP", "LOOP_WHILE": "LOOP",
    "BRANCH_IF": "BRANCH", "BRANCH_SWITCH": "BRANCH",
    "JUMP_BREAK": "JUMP", "JUMP_GOTO": "JUMP", "RETURN": "JUMP",
    "ARITH_ADD": "ARITH_OP", "ARITH_MUL": "ARITH_OP", "ARITH_MOD": "ARITH_OP",
    "BITWISE_OP": "BITWISE_OP", "SHIFT_OP": "BITWISE_OP",
    "COMPARISON_EQ": "COMPARISON", "COMPARISON_REL": "COMPARISON",
    "LOGICAL_OP": "LOGICAL_OP",
    "CALL_SITE": "CALL_SITE", "ASSIGN": "ASSIGN",
    "UNKNOWN": "UNKNOWN",
}

PROJECT_16_TO_8 = {
    "MEMORY_ALLOC": "MEMORY_ACCESS", "MEMORY_FREE": "MEMORY_ACCESS",
    "MEMORY_COPY": "MEMORY_ACCESS", "STRING_OP": "MEMORY_ACCESS",
    "ARRAY_INDEX": "ARRAY_INDEX",
    "PTR_DEREF": "PTR_DEREF", "FIELD_ACCESS": "PTR_DEREF",
    "LOOP": "CONTROL_BRANCH", "BRANCH": "CONTROL_BRANCH", "JUMP": "CONTROL_BRANCH",
    "ARITH_OP": "ARITH_OP", "BITWISE_OP": "ARITH_OP",
    "COMPARISON": "COMPARISON", "LOGICAL_OP": "COMPARISON",
    "CALL_SITE": "CALL_SITE", "ASSIGN": "ASSIGN",
    "UNKNOWN": "UNKNOWN",
}

_TOKEN_RE = re.compile(
    r"\w+|->|\+\+|--|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||"
    r"\+=|-=|\*=|/=|%=|&=|\|=|\^=|[^\s\w]"
)

_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")
_NUM_RE = re.compile(r"^\d")


def _tokenize(code: str, max_tokens: int = 100) -> List[str]:
    """Lightweight C/C++ tokenizer with multi-character operator merging."""
    return _TOKEN_RE.findall(code)[:max_tokens]


def _is_value_token(tok: str) -> bool:
    """True if `tok` can terminate a value expression (identifier, number, ) or ])."""
    return bool(_IDENT_RE.match(tok)) or bool(_NUM_RE.match(tok)) or tok in (")", "]")


def classify_token(tokens: List[str], i: int) -> str:
    """
    Deterministic, context-aware mapping phi_type from a token (in its local
    context) to the fine-grained 32-type taxonomy. Rules, in priority order:

      1. identifier followed by '('  → API class if in an allow-list;
         C keyword class if a control keyword; else CALL_SITE.
         (User-defined functions, macros and unresolved calls all fall into
         CALL_SITE — the framework never needs to resolve them.)
      2. identifier followed by '['  → ARRAY_INDEX.
      3. control keywords            → LOOP_* / BRANCH_* / JUMP_* / RETURN.
      4. '->' , '.' between values   → FIELD_ACCESS.
      5. unary '*' / '&'             → PTR_DEREF / ADDR_OF (binary uses are
                                       ARITH_MUL / BITWISE_OP).
      6. type keyword inside '( .. )'→ CAST.
      7. operators                   → ARITH_* / BITWISE_OP / SHIFT_OP /
                                       COMPARISON_* / LOGICAL_OP / ASSIGN.
      8. everything else             → UNKNOWN.
    """
    tok = tokens[i]
    prev = tokens[i - 1] if i > 0 else ""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""

    if _IDENT_RE.match(tok):
        if nxt == "(":
            for cls, names in API_CLASSES.items():
                if tok in names:
                    return cls
            if tok == "if":
                return "BRANCH_IF"
            if tok == "switch":
                return "BRANCH_SWITCH"
            if tok == "for":
                return "LOOP_FOR"
            if tok == "while":
                return "LOOP_WHILE"
            if tok == "return":
                return "RETURN"
            if tok in C_KEYWORDS:
                return "UNKNOWN"
            return "CALL_SITE"
        if nxt == "[":
            return "ARRAY_INDEX"
        if tok == "for":
            return "LOOP_FOR"
        if tok in ("while", "do"):
            return "LOOP_WHILE"
        if tok in ("if", "else"):
            return "BRANCH_IF"
        if tok in ("switch", "case", "default"):
            return "BRANCH_SWITCH"
        if tok in ("break", "continue"):
            return "JUMP_BREAK"
        if tok == "goto":
            return "JUMP_GOTO"
        if tok == "return":
            return "RETURN"
        if tok in TYPE_KEYWORDS and prev == "(" and nxt in (")", "*"):
            return "CAST"
        return "UNKNOWN"

    # Operators / punctuation
    if tok == "->":
        return "FIELD_ACCESS"
    if tok == "." and _is_value_token(prev) and _IDENT_RE.match(nxt or ""):
        return "FIELD_ACCESS"
    if tok == "*":
        return "ARITH_MUL" if _is_value_token(prev) else "PTR_DEREF"
    if tok == "&":
        return "BITWISE_OP" if _is_value_token(prev) else "ADDR_OF"
    if tok in ("==", "!="):
        return "COMPARISON_EQ"
    if tok in ("<", ">", "<=", ">="):
        return "COMPARISON_REL"
    if tok in ("&&", "||", "!"):
        return "LOGICAL_OP"
    if tok in ("+", "-", "++", "--"):
        return "ARITH_ADD"
    if tok == "/":
        return "ARITH_MUL"
    if tok == "%":
        return "ARITH_MOD"
    if tok in ("|", "^", "~"):
        return "BITWISE_OP"
    if tok in ("<<", ">>"):
        return "SHIFT_OP"
    if tok in ("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="):
        return "ASSIGN"
    if tok == "?":
        return "BRANCH_IF"   # ternary conditional
    return "UNKNOWN"


def _project_type(fine: str, taxonomy_size: int) -> str:
    t16 = PROJECT_32_TO_16[fine]
    if taxonomy_size == 32:
        return fine
    if taxonomy_size == 16:
        return t16
    return PROJECT_16_TO_8[t16]


def _build_vocab(all_tokens: List[List[str]], max_vocab: int = 10000) -> Dict[str, int]:
    """Build vocabulary from list-of-token-lists."""
    from collections import Counter
    counter = Counter(t for toks in all_tokens for t in toks)
    vocab = {"<UNK>": 0, "<PAD>": 1}
    for tok, _ in counter.most_common(max_vocab - 2):
        vocab[tok] = len(vocab)
    return vocab


def _code_to_graph(code: str, vocab: Dict[str, int], max_tokens: int = 256,
                   taxonomy_size: int = 8) -> Optional[Data]:
    """
    Convert a C/C++ function string to a PyG graph.

    Primary path: a real tree-sitter AST graph (see ast_graphs.py) with
    grammar-kind and morphology node features. Fallback (parse failure or
    tree-sitter unavailable): the lexical dependence token graph below, whose
    nodes carry the reserved FALLBACK kind id.
    """
    from data.loaders.ast_graphs import (code_to_ast_graph, TS_AVAILABLE,
                                         FALLBACK_KIND)
    if TS_AVAILABLE:
        g = code_to_ast_graph(code, vocab, max_nodes=200,
                              taxonomy_size=taxonomy_size)
        if g is not None:
            return g

    tokens = _tokenize(code, max_tokens)
    if len(tokens) < 3:
        return None

    _, morph_map = get_taxonomy(taxonomy_size)

    n = len(tokens)
    x_lex = torch.tensor([vocab.get(t, 0) for t in tokens], dtype=torch.long)
    x_morph = torch.tensor(
        [morph_map[_project_type(classify_token(tokens, i), taxonomy_size)]
         for i in range(n)],
        dtype=torch.long,
    )

    # NCS edges: sequential next-token edges
    src = list(range(n - 1))
    dst = list(range(1, n))

    # DD edges: def-use proxy connecting successive same-identifier occurrences
    tok_positions: Dict[str, List[int]] = {}
    for i, t in enumerate(tokens):
        if _IDENT_RE.match(t) and t not in C_KEYWORDS:
            tok_positions.setdefault(t, []).append(i)

    for positions in tok_positions.values():
        for i in range(len(positions) - 1):
            src.append(positions[i])
            dst.append(positions[i + 1])

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    unknown_id = morph_map["UNKNOWN"]
    data = Data(x_lex=x_lex, x_morph=x_morph, edge_index=edge_index, num_nodes=n)
    data.x_kind = torch.full((n,), FALLBACK_KIND, dtype=torch.long)
    data.morph_known = int((x_morph != unknown_id).sum())
    return data


def abstraction_stats(data_list: List[Data]) -> Dict[str, float]:
    """
    Report how much the morphological abstraction actually compresses:
      - typed_node_ratio: fraction of nodes with a non-UNKNOWN semantic type
      - avg_nodes / avg_edges: raw graph size statistics
    """
    if not data_list:
        return {}
    total_nodes = sum(d.num_nodes for d in data_list)
    total_known = sum(int(getattr(d, "morph_known", 0)) for d in data_list)
    total_edges = sum(d.edge_index.size(1) for d in data_list)
    return {
        "typed_node_ratio": total_known / max(1, total_nodes),
        "avg_nodes": total_nodes / len(data_list),
        "avg_edges": total_edges / len(data_list),
        "num_graphs": len(data_list),
    }


# ── Generic row → graph list conversion ─────────────────────────────────────

def _rows_to_graphs(rows, code_key_candidates, label_fn, cwe_fn, project_fn,
                    max_samples: int, taxonomy_size: int, name: str) -> List[Data]:
    def get_code(row):
        for k in code_key_candidates:
            if row.get(k):
                return row[k]
        return ""

    all_tokens = [_tokenize(get_code(r), 100) for r in rows]
    vocab = _build_vocab(all_tokens)

    data_list = []
    for row in rows:
        if len(data_list) >= max_samples:
            break
        code = get_code(row)
        label = label_fn(row)
        graph = _code_to_graph(code, vocab, taxonomy_size=taxonomy_size)
        if graph is None:
            continue
        graph.y = torch.tensor([float(label)])
        graph.cwe = torch.tensor([cwe_fn(row) if label == 1 else -1], dtype=torch.long)
        graph.project = project_fn(row)
        data_list.append(graph)

    print(f"{name}: loaded {len(data_list)} samples, "
          f"vuln={sum(1 for d in data_list if d.y[0] == 1)}")
    return data_list


def _parse_cwe(raw) -> int:
    """
    Parse a raw CWE annotation to its integer id.

    Label model: functions may carry multiple CWE annotations (one CVE can
    span several functions and one function several CVEs); we take the FIRST
    listed CWE as the primary weakness type. The CWE id conditions only the
    prototype construction — the detection head is strictly binary.
    """
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else "-1"
    try:
        return int(re.search(r"\d+", str(raw)).group())
    except Exception:
        return -1


def bucket_cwes(data_list: List[Data], num_cwes: int) -> Dict[int, int]:
    """
    Map raw CWE ids to a fixed vocabulary of `num_cwes` buckets:
    the (num_cwes - 1) most frequent CWE types in the training corpus get
    dedicated buckets; every remaining type maps to a shared OTHER bucket
    (index num_cwes - 1). Benign functions (cwe = -1) never enter any bucket.

    Mutates `data_list` in place and returns the {raw_cwe: bucket} mapping.
    """
    from collections import Counter
    counts = Counter(
        int(d.cwe[0]) for d in data_list
        if float(d.y[0]) == 1.0 and int(d.cwe[0]) >= 0
    )
    top = [cwe for cwe, _ in counts.most_common(max(1, num_cwes - 1))]
    mapping = {cwe: i for i, cwe in enumerate(top)}
    other = num_cwes - 1

    for d in data_list:
        raw = int(d.cwe[0])
        if float(d.y[0]) == 1.0 and raw >= 0:
            d.cwe = torch.tensor([mapping.get(raw, other)], dtype=torch.long)
        else:
            d.cwe = torch.tensor([-1], dtype=torch.long)

    print(f"CWE bucketing: {len(counts)} distinct CWEs → {num_cwes} buckets "
          f"(top {len(top)} dedicated + OTHER); "
          f"top CWEs: {[f'CWE-{c}' for c in top[:10]]}")
    return mapping


# ── Devign Loader ────────────────────────────────────────────────────────────

def load_devign(max_samples: int = 5000, cache: bool = True,
                taxonomy_size: int = 8) -> List[Data]:
    """
    Load Devign dataset from HuggingFace (DetectVul/devign).
    Reference: plan.md §4.1, Zhou et al. NeurIPS 2019.
    """
    cache_path = CACHE_DIR / f"devign_{max_samples}_t{taxonomy_size}.pt"
    if cache and cache_path.exists():
        print(f"Loading Devign from cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    try:
        from datasets import load_dataset
        print("Downloading Devign from HuggingFace (DetectVul/devign)...")
        hf_ds = load_dataset("DetectVul/devign", split="train")
    except Exception as e:
        print(f"Could not load Devign from HuggingFace: {e}")
        return []

    rows = list(hf_ds)
    data_list = _rows_to_graphs(
        rows,
        code_key_candidates=["func"],
        label_fn=lambda r: int(bool(r["target"])) if not isinstance(r["target"], str)
                           else int(r["target"].strip().lower() in ("1", "true")),
        cwe_fn=lambda r: 0,   # Devign has no per-sample CWE; single bucket
        project_fn=lambda r: r.get("project", "devign"),
        max_samples=max_samples, taxonomy_size=taxonomy_size, name="Devign",
    )

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, cache_path)
    return data_list


# ── PrimeVul Loader ──────────────────────────────────────────────────────────

def load_primevul(split: str = "train", max_samples: int = 5000, cache: bool = True,
                  taxonomy_size: int = 8) -> List[Data]:
    """
    Load PrimeVul from HuggingFace (starsofchance/PrimeVul).
    Reference: plan.md §4.1, Ding et al. ICSE 2025.
    """
    cache_path = CACHE_DIR / f"primevul_{split}_{max_samples}_t{taxonomy_size}.pt"
    if cache and cache_path.exists():
        print(f"Loading PrimeVul ({split}) from cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    try:
        from datasets import load_dataset
        print(f"Downloading PrimeVul ({split}) from HuggingFace...")
        hf_ds = load_dataset("starsofchance/PrimeVul", split=split)
    except Exception as e:
        print(f"Could not load PrimeVul from HuggingFace: {e}")
        return []

    rows = list(hf_ds)
    data_list = _rows_to_graphs(
        rows,
        code_key_candidates=["func", "code"],
        label_fn=lambda r: int(r.get("target", r.get("label", 0)) or 0),
        cwe_fn=lambda r: _parse_cwe(r.get("cwe", "-1")),
        project_fn=lambda r: r.get("project", "primevul"),
        max_samples=max_samples, taxonomy_size=taxonomy_size,
        name=f"PrimeVul ({split})",
    )

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, cache_path)
    return data_list


# ── BigVul Loader ────────────────────────────────────────────────────────────

def load_bigvul(csv_path: str, max_samples: int = 10000, cache: bool = True,
                taxonomy_size: int = 8) -> List[Data]:
    """
    Load BigVul from a local CSV file.
    Download from: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
    Reference: plan.md §4.1, Fan et al. MSR 2020.
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"BigVul CSV not found at {csv_path}. Skipping.")
        return []

    cache_key = hashlib.md5(csv_path.encode()).hexdigest()[:8]
    cache_path = CACHE_DIR / f"bigvul_{cache_key}_{max_samples}_t{taxonomy_size}.pt"
    if cache and cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    import csv
    rows = []
    with open(csv_path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= max_samples * 2:
                break

    data_list = _rows_to_graphs(
        rows,
        code_key_candidates=["func_before"],
        label_fn=lambda r: int(r.get("vul", 0) or 0),
        cwe_fn=lambda r: _parse_cwe(r.get("CWE ID", "-1")),
        project_fn=lambda r: r.get("project", "bigvul"),
        max_samples=max_samples, taxonomy_size=taxonomy_size, name="BigVul",
    )

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, cache_path)
    return data_list


# ── DiverseVul Loader ─────────────────────────────────────────────────────────

def load_diversevul(json_path: str, max_samples: int = 10000, cache: bool = True,
                    taxonomy_size: int = 8) -> List[Data]:
    """
    Load DiverseVul from a local JSONL file.
    Download from: https://github.com/wagner-group/diversevul
    Reference: plan.md §4.1, Chen et al. RAID 2023.
    """
    if not json_path or not os.path.exists(json_path):
        print(f"DiverseVul JSON not found at {json_path}. Skipping.")
        return []

    cache_key = hashlib.md5(json_path.encode()).hexdigest()[:8]
    cache_path = CACHE_DIR / f"diversevul_{cache_key}_{max_samples}_t{taxonomy_size}.pt"
    if cache and cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    rows = []
    with open(json_path) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_samples * 2:
                break

    data_list = _rows_to_graphs(
        rows,
        code_key_candidates=["func"],
        label_fn=lambda r: int(r.get("target", 0) or 0),
        cwe_fn=lambda r: _parse_cwe(r.get("cwe", "-1")),
        project_fn=lambda r: r.get("project", r.get("repo", "diversevul")),
        max_samples=max_samples, taxonomy_size=taxonomy_size, name="DiverseVul",
    )

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, cache_path)
    return data_list


# ── HF-mirror loaders (BigVul / DiverseVul) ──────────────────────────────────

def _load_hf_generic(hf_name: str, split: str, code_keys, label_fn, cwe_fn,
                     project_fn, max_samples: int, taxonomy_size: int,
                     cache_tag: str, cache: bool = True) -> List[Data]:
    """Shared loader for HuggingFace-hosted vulnerability corpora."""
    cache_path = CACHE_DIR / f"{cache_tag}_{max_samples}_t{taxonomy_size}.pt"
    if cache and cache_path.exists():
        print(f"Loading {cache_tag} from cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    try:
        from datasets import load_dataset
        print(f"Streaming {hf_name} ({split}) from HuggingFace...")
        hf_ds = load_dataset(hf_name, split=split, streaming=True)
        rows = []
        for row in hf_ds:
            rows.append(row)
            if len(rows) >= max_samples * 2:
                break
    except Exception as e:
        print(f"Could not load {hf_name}: {e}")
        return []

    data_list = _rows_to_graphs(
        rows, code_key_candidates=code_keys, label_fn=label_fn, cwe_fn=cwe_fn,
        project_fn=project_fn, max_samples=max_samples,
        taxonomy_size=taxonomy_size, name=cache_tag,
    )

    if cache and data_list:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, cache_path)
    return data_list


def load_bigvul_hf(max_samples: int = 10000, taxonomy_size: int = 8,
                   cache: bool = True) -> List[Data]:
    """BigVul via the bstee615/bigvul HuggingFace mirror (has project + CWE)."""
    return _load_hf_generic(
        "bstee615/bigvul", "train",
        code_keys=["func_before"],
        label_fn=lambda r: int(r.get("vul", 0) or 0),
        cwe_fn=lambda r: _parse_cwe(r.get("CWE ID", "-1")),
        project_fn=lambda r: r.get("project", "bigvul"),
        max_samples=max_samples, taxonomy_size=taxonomy_size,
        cache_tag="bigvul_hf", cache=cache,
    )


def load_primevul_hf(max_samples: int = 10000, taxonomy_size: int = 8,
                     cache: bool = True) -> List[Data]:
    """PrimeVul via the ASSERT-KTH/PrimeVul mirror (train_unpaired split;
    has project + CWE list; the first CWE is taken as primary)."""
    return _load_hf_generic(
        "ASSERT-KTH/PrimeVul", "train_unpaired",
        code_keys=["func"],
        label_fn=lambda r: int(r.get("is_vulnerable", 0) or 0),
        cwe_fn=lambda r: _parse_cwe(r.get("cwe", "-1")),
        project_fn=lambda r: r.get("project", "primevul"),
        max_samples=max_samples, taxonomy_size=taxonomy_size,
        cache_tag="primevul_hf", cache=cache,
    )


def load_diversevul_hf(max_samples: int = 10000, taxonomy_size: int = 8,
                       cache: bool = True) -> List[Data]:
    """DiverseVul via the bstee615/diversevul HuggingFace mirror
    (has project + CWE list; the first CWE is taken as primary)."""
    return _load_hf_generic(
        "bstee615/diversevul", "train",
        code_keys=["func"],
        label_fn=lambda r: int(r.get("target", 0) or 0),
        cwe_fn=lambda r: _parse_cwe(r.get("cwe", "-1")),
        project_fn=lambda r: r.get("project", "diversevul"),
        max_samples=max_samples, taxonomy_size=taxonomy_size,
        cache_tag="diversevul_hf", cache=cache,
    )


def downsample_benign(bucket: List[Data], ratio: float = 4.0,
                      seed: int = 0) -> List[Data]:
    """
    Cap the benign:vulnerable ratio of a TRAINING bucket at `ratio`
    (ReVeal-style rebalancing). Test and calibration splits are never
    downsampled, so all reported metrics reflect the true prevalence.
    """
    rng = random.Random(seed)
    pos = [d for d in bucket if float(d.y[0]) == 1.0]
    neg = [d for d in bucket if float(d.y[0]) != 1.0]
    cap = int(len(pos) * ratio)
    if pos and len(neg) > cap:
        rng.shuffle(neg)
        neg = neg[:cap]
    out = pos + neg
    rng.shuffle(out)
    return out


def carve_calibration(client_buckets: List[List[Data]], seed: int,
                      cal_fraction: float = 0.1,
                      downsample_ratio: float = 4.0):
    """
    Split off a calibration set (cal_fraction of each client's samples,
    at true prevalence) BEFORE benign downsampling, then downsample the
    remaining training samples. The calibration set is used only to choose
    the decision threshold — it comes from training projects, never from
    held-out test projects.
    """
    rng = random.Random(seed)
    cal, new_buckets = [], []
    for b in client_buckets:
        b = b[:]
        rng.shuffle(b)
        n_cal = max(1, int(cal_fraction * len(b)))
        cal.extend(b[:n_cal])
        new_buckets.append(downsample_benign(b[n_cal:], downsample_ratio, seed))
    return new_buckets, cal


# ── Cross-Project Federated Split ─────────────────────────────────────────────

def split_by_project(
    data_list: List[Data],
    num_clients: int,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[List[Data]], List[Data]]:
    """
    Partition a dataset by project for cross-project federated evaluation.

    - Groups samples by `data.project`.
    - Holds out whole projects (~test_fraction of samples) as the
      cross-project test set: no function from a test project is ever seen
      by any federated client.
    - Distributes the remaining projects across K clients (round-robin by
      project when enough projects exist; otherwise samples of the remaining
      training projects are sharded randomly across clients, keeping the
      test set strictly project-disjoint).
    - Only when a dataset has a single project label does it fall back to a
      random sample split (and says so loudly) — this split is NOT
      cross-project and is excluded from cross-project claims.

    Returns:
        client_datasets: List of K lists of Data objects (train split).
        test_dataset:    List of Data objects from held-out projects.
    """
    rng = random.Random(seed)

    # Group by project
    by_project: Dict[str, List[Data]] = {}
    for d in data_list:
        proj = getattr(d, 'project', 'unknown')
        by_project.setdefault(proj, []).append(d)

    # Deterministic holdout: smallest projects first (seed only breaks ties),
    # so the held-out cross-project test set totals ~test_fraction of samples
    # while the training pool keeps the larger repositories.
    projects = sorted(by_project.keys())
    rng.shuffle(projects)
    projects.sort(key=lambda p: len(by_project[p]))

    if len(projects) == 1:
        print("WARNING: single-project dataset — falling back to a random "
              "sample split. This split is NOT cross-project.")
        shuffled = data_list[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_fraction))
        test_raw, train_raw = shuffled[:n_test], shuffled[n_test:]
    else:
        # Hold out whole projects totalling ~test_fraction of the samples.
        target = test_fraction * len(data_list)
        test_projects, count = [], 0
        for p in projects:
            if count >= target or len(test_projects) >= max(1, len(projects) - 1):
                break
            test_projects.append(p)
            count += len(by_project[p])
        train_projects = [p for p in projects if p not in set(test_projects)]

        test_raw = [d for p in test_projects for d in by_project[p]]

        if len(train_projects) >= num_clients:
            # Round-robin whole projects to clients (each client = disjoint
            # set of projects, simulating per-organisation codebases).
            client_buckets: List[List[Data]] = [[] for _ in range(num_clients)]
            for i, proj in enumerate(train_projects):
                client_buckets[i % num_clients].extend(by_project[proj])
            client_buckets = [b for b in client_buckets if b]
            print(f"Cross-project split: {len(train_projects)} train projects "
                  f"across {len(client_buckets)} clients, "
                  f"{len(test_projects)} held-out test projects "
                  f"({len(test_raw)} samples).")
            return client_buckets, test_raw

        # Fewer training projects than clients: shard training samples
        # randomly; the test set remains strictly project-disjoint.
        train_raw = [d for p in train_projects for d in by_project[p]]
        rng.shuffle(train_raw)
        print(f"Cross-project split (few projects): train projects "
              f"{train_projects} sharded across {num_clients} clients; "
              f"held-out test projects {test_projects} ({len(test_raw)} samples).")

    chunk = max(1, len(train_raw) // num_clients)
    client_buckets = [
        train_raw[i * chunk: (i + 1) * chunk] for i in range(num_clients)
    ]
    client_buckets[-1].extend(train_raw[num_clients * chunk:])
    client_buckets = [b for b in client_buckets if b]
    return client_buckets, test_raw


class ListDataset(Dataset):
    """Thin wrapper to expose a plain List[Data] as a PyG Dataset."""
    def __init__(self, data_list: List[Data]):
        super().__init__(root=None)
        self._data = data_list

    def len(self): return len(self._data)
    def get(self, idx): return self._data[idx]
