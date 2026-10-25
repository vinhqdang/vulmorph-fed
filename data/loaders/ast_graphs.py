"""
AST-based graph construction for VulMorph-Fed (tree-sitter).

Replaces the token-proxy graphs with real C/C++ abstract syntax trees:

  - nodes    : named AST nodes (pre-order, capped at `max_nodes`);
  - AST edges: parent -> child (both directions added downstream by the GNN
               normalisation; we store one direction each way);
  - SIB edges: consecutive named siblings (source-order sequencing);
  - DU edges : def-use proxy linking successive occurrences of the same
               identifier leaf.

Node features (all project-invariant):
  - x_kind : the tree-sitter grammar node kind (fixed grammar vocabulary,
             identical for every project and client by construction);
  - x_morph: the morphological taxonomy type, assigned by phi_type rules
             evaluated on the AST (call names against API allow-lists,
             operator text of binary expressions, statement kinds, ...).

x_lex (node-text token id) is retained solely for the lexical baselines and
the w/o-morphology ablation; the VulMorph model never consumes it.
"""

from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from data.morphology import get_taxonomy

try:
    import tree_sitter_c as _tsc
    from tree_sitter import Language as _Language, Parser as _Parser
    C_LANGUAGE = _Language(_tsc.language())
    _PARSER = _Parser(C_LANGUAGE)
    NUM_AST_KINDS = C_LANGUAGE.node_kind_count + 1   # +1 for FALLBACK_TOKEN
    FALLBACK_KIND = C_LANGUAGE.node_kind_count       # id used by token-graph fallback
    TS_AVAILABLE = True
except Exception:                                     # pragma: no cover
    C_LANGUAGE = _PARSER = None
    NUM_AST_KINDS, FALLBACK_KIND, TS_AVAILABLE = 364, 363, False


# API allow-lists shared with the token-level rules.
from data.loaders.api_classes import API_CLASSES, classify_api

_BINOP_TO_FINE = {
    "+": "ARITH_ADD", "-": "ARITH_ADD",
    "*": "ARITH_MUL", "/": "ARITH_MUL", "%": "ARITH_MOD",
    "&": "BITWISE_OP", "|": "BITWISE_OP", "^": "BITWISE_OP",
    "<<": "SHIFT_OP", ">>": "SHIFT_OP",
    "==": "COMPARISON_EQ", "!=": "COMPARISON_EQ",
    "<": "COMPARISON_REL", ">": "COMPARISON_REL",
    "<=": "COMPARISON_REL", ">=": "COMPARISON_REL",
    "&&": "LOGICAL_OP", "||": "LOGICAL_OP",
}

_KIND_TO_FINE = {
    "subscript_expression": "ARRAY_INDEX",
    "field_expression": "FIELD_ACCESS",
    "cast_expression": "CAST",
    "if_statement": "BRANCH_IF",
    "conditional_expression": "BRANCH_IF",
    "switch_statement": "BRANCH_SWITCH",
    "case_statement": "BRANCH_SWITCH",
    "for_statement": "LOOP_FOR",
    "while_statement": "LOOP_WHILE",
    "do_statement": "LOOP_WHILE",
    "break_statement": "JUMP_BREAK",
    "continue_statement": "JUMP_BREAK",
    "goto_statement": "JUMP_GOTO",
    "return_statement": "RETURN",
    "assignment_expression": "ASSIGN",
    "init_declarator": "ASSIGN",
    "update_expression": "ARITH_ADD",
}


def _classify_ast_node(node, src: bytes) -> str:
    """phi_type evaluated on an AST node → fine-grained (|T|=32) label."""
    kind = node.type

    if kind == "call_expression":
        fn = node.child_by_field_name("function")
        name = src[fn.start_byte:fn.end_byte].decode("utf8", "replace") if fn else ""
        # strip member access:  obj->free / ns.malloc  → last component
        name = name.split("->")[-1].split(".")[-1].strip()
        cls = classify_api(name)
        if cls is not None:
            return cls
        return "CALL_SITE"

    if kind in _KIND_TO_FINE:
        return _KIND_TO_FINE[kind]

    if kind == "binary_expression":
        op = node.child_by_field_name("operator")
        text = (src[op.start_byte:op.end_byte].decode("utf8", "replace")
                if op else "")
        return _BINOP_TO_FINE.get(text, "ARITH_ADD")

    if kind == "pointer_expression":
        op = src[node.start_byte:node.start_byte + 1].decode("utf8", "replace")
        return "ADDR_OF" if op == "&" else "PTR_DEREF"

    if kind == "unary_expression":
        op = src[node.start_byte:node.start_byte + 1].decode("utf8", "replace")
        return "LOGICAL_OP" if op == "!" else "ARITH_ADD"

    return "UNKNOWN"


# Projection tables live in real_datasets to keep one source of truth.
def _project(fine: str, taxonomy_size: int) -> str:
    from data.loaders.real_datasets import PROJECT_32_TO_16, PROJECT_16_TO_8
    t16 = PROJECT_32_TO_16[fine]
    if taxonomy_size == 32:
        return fine
    if taxonomy_size == 16:
        return t16
    return PROJECT_16_TO_8[t16]


# ── Lexical token definitions ────────────────────────────────────────────────
#
# Three ways a node can be turned into a token. They differ only in how much
# project-specific text survives, and the experiment grid crosses them with
# the grammar kind so that "structure labels vs no structure labels" is never
# confounded with "lexical vs invariant".
#
#   lexical : a LEAF carries its source text (<= 30 chars, else <LONG>);
#             every INTERNAL node carries the single token <INTERNAL>. The
#             lexical mode therefore holds no node-type information at all.
#   anon    : identifiers are alpha-renamed by first occurrence within the
#             function (VAR_i / FUN_i / FIELD_i / TYPE_i), numeric and string
#             literals collapse to NUM / STR, everything else keeps its text.
#             Equality of occurrences survives, spelling does not.
#
# Unseen tokens: `lexical` ids index a corpus-wide token dictionary; the
# experiment runner maps every token not seen in the TRAINING folds to <UNK>
# (id 0) per fold, so test-time out-of-vocabulary behaviour is measured.

_LITERAL_LEAVES = {"string_literal", "char_literal", "concatenated_string",
                   "number_literal", "system_lib_string"}
_RENAMED = {"identifier": "VAR", "field_identifier": "FIELD",
            "type_identifier": "TYPE", "statement_identifier": "LABEL"}
ANON_CAP = 15
# Leaves whose text is free-form and therefore project-specific.
_FREE_TEXT = {"comment", "preproc_arg", "string_content", "character",
              "escape_sequence", "preproc_directive", "ms_call_modifier",
              "attribute", "identifier_like"}


def _is_leaf_like(n) -> bool:
    return n.child_count == 0 or n.type in _LITERAL_LEAVES


def code_to_ast_graph(code: str, vocab: Dict[str, int],
                      max_nodes: int = 200,
                      taxonomy_size: int = 8,
                      anon_vocab: Optional[Dict[str, int]] = None,
                      grow_vocab: bool = False) -> Optional[Data]:
    """Parse a C/C++ function into a PyG graph; None if unusable.

    `vocab` maps lexical tokens to ids. With `grow_vocab` the dictionary is
    extended in place (ids are corpus-wide; the per-fold train-only cut is
    applied by the experiment runner). `anon_vocab` does the same for the
    alpha-renamed tokens, which form a small, project-invariant alphabet.
    """
    if not TS_AVAILABLE:
        return None
    src = code.encode("utf8", "replace")
    try:
        tree = _PARSER.parse(src)
    except Exception:
        return None
    root = tree.root_node
    if root is None or root.named_child_count == 0:
        return None
    if anon_vocab is None:
        anon_vocab = {"<UNK>": 0}

    _, morph_map = get_taxonomy(taxonomy_size)

    # Pre-order traversal over named nodes; the first `max_nodes` are kept
    # but the traversal continues so truncation can be measured.
    nodes, parent_of = [], []
    n_full = 0
    stack = [(root, -1)]
    while stack:
        node, parent_idx = stack.pop()
        n_full += 1
        idx = len(nodes)
        if idx < max_nodes:
            nodes.append(node)
            parent_of.append(parent_idx)
        else:
            idx = -2          # descendants of a dropped node are dropped
        for c in reversed([c for c in node.children if c.is_named]):
            stack.append((c, idx if idx >= 0 else -2))
    # children of dropped nodes carry parent -2 and fall outside the cap
    if len(nodes) < 3:
        return None

    def tok_id(d, tok):
        i = d.get(tok)
        if i is None:
            if not grow_vocab:
                return 0
            i = d[tok] = len(d)
        return i

    kind_ids, morph_ids, lex_ids, anon_ids = [], [], [], []
    ident_positions: Dict[str, List[int]] = {}
    rename: Dict[str, Dict[str, int]] = {}
    for i, n in enumerate(nodes):
        kind_ids.append(min(n.kind_id, NUM_AST_KINDS - 1))
        morph_ids.append(morph_map[_project(_classify_ast_node(n, src),
                                            taxonomy_size)])
        text = src[n.start_byte:n.end_byte].decode("utf8", "replace")
        if _is_leaf_like(n):
            lex_tok = text if len(text) <= 30 else "<LONG>"
            if n.type in _RENAMED:
                tag = _RENAMED[n.type]
                par = nodes[parent_of[i]] if parent_of[i] >= 0 else None
                if (tag == "VAR" and par is not None
                        and par.type == "call_expression"
                        and par.child_by_field_name("function") == n):
                    tag = "FUN"
                table = rename.setdefault(tag, {})
                k = table.setdefault(text, len(table))
                anon_tok = f"{tag}_{min(k, ANON_CAP)}"
            elif n.type == "number_literal":
                anon_tok = "NUM"
            elif n.type in _LITERAL_LEAVES:
                anon_tok = "STR"
            elif n.type in _FREE_TEXT:
                anon_tok = n.type.upper()
            else:
                anon_tok = lex_tok
        else:
            lex_tok = anon_tok = "<INTERNAL>"
        lex_ids.append(tok_id(vocab, lex_tok))
        anon_ids.append(tok_id(anon_vocab, anon_tok))
        if n.type == "identifier":
            ident_positions.setdefault(text, []).append(i)

    srcs, dsts, etype = [], [], []
    # AST edges (parent -> child = 0, child -> parent = 1)
    for i, p in enumerate(parent_of):
        if p >= 0:
            srcs += [p, i]
            dsts += [i, p]
            etype += [0, 1]
    # Sibling order edges (2)
    by_parent: Dict[int, List[int]] = {}
    for i, p in enumerate(parent_of):
        by_parent.setdefault(p, []).append(i)
    for sibs in by_parent.values():
        for a, b in zip(sibs, sibs[1:]):
            srcs.append(a)
            dsts.append(b)
            etype.append(2)
    # Def-use proxy edges (3)
    for positions in ident_positions.values():
        for a, b in zip(positions, positions[1:]):
            srcs.append(a)
            dsts.append(b)
            etype.append(3)

    unknown_id = morph_map["UNKNOWN"]
    data = Data(
        x_lex=torch.tensor(lex_ids, dtype=torch.long),
        x_anon=torch.tensor(anon_ids, dtype=torch.long),
        x_morph=torch.tensor(morph_ids, dtype=torch.long),
        x_kind=torch.tensor(kind_ids, dtype=torch.long),
        edge_index=torch.tensor([srcs, dsts], dtype=torch.long),
        edge_type=torch.tensor(etype, dtype=torch.long),
        num_nodes=len(nodes),
    )
    data.morph_known = int((data.x_morph != unknown_id).sum())
    data.n_full = int(n_full)
    data.truncated = int(n_full > max_nodes)
    return data
