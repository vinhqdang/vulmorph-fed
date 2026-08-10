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
from data.loaders.api_classes import API_CLASSES

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
        for cls, names in API_CLASSES.items():
            if name in names:
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


def code_to_ast_graph(code: str, vocab: Dict[str, int],
                      max_nodes: int = 200,
                      taxonomy_size: int = 8) -> Optional[Data]:
    """Parse a C/C++ function into a PyG graph; None if unusable."""
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

    _, morph_map = get_taxonomy(taxonomy_size)

    # Pre-order traversal over named nodes, capped.
    nodes, parent_of = [], []
    stack = [(root, -1)]
    while stack and len(nodes) < max_nodes:
        node, parent_idx = stack.pop()
        idx = len(nodes)
        nodes.append(node)
        parent_of.append(parent_idx)
        named = [c for c in node.children if c.is_named]
        for c in reversed(named):
            stack.append((c, idx))
    if len(nodes) < 3:
        return None

    kind_ids, morph_ids, lex_ids = [], [], []
    ident_positions: Dict[str, List[int]] = {}
    for i, n in enumerate(nodes):
        kind_ids.append(min(n.kind_id, NUM_AST_KINDS - 1))
        morph_ids.append(morph_map[_project(_classify_ast_node(n, src),
                                            taxonomy_size)])
        text = src[n.start_byte:n.end_byte].decode("utf8", "replace")
        token = text if len(text) <= 30 and n.child_count == 0 else n.type
        lex_ids.append(vocab.get(token, 0))
        if n.type == "identifier":
            ident_positions.setdefault(text, []).append(i)

    srcs, dsts = [], []
    # AST edges (child -> parent and parent -> child)
    for i, p in enumerate(parent_of):
        if p >= 0:
            srcs += [p, i]
            dsts += [i, p]
    # Sibling order edges
    by_parent: Dict[int, List[int]] = {}
    for i, p in enumerate(parent_of):
        by_parent.setdefault(p, []).append(i)
    for sibs in by_parent.values():
        for a, b in zip(sibs, sibs[1:]):
            srcs.append(a)
            dsts.append(b)
    # Def-use proxy edges
    for positions in ident_positions.values():
        for a, b in zip(positions, positions[1:]):
            srcs.append(a)
            dsts.append(b)

    unknown_id = morph_map["UNKNOWN"]
    data = Data(
        x_lex=torch.tensor(lex_ids, dtype=torch.long),
        x_morph=torch.tensor(morph_ids, dtype=torch.long),
        x_kind=torch.tensor(kind_ids, dtype=torch.long),
        edge_index=torch.tensor([srcs, dsts], dtype=torch.long),
        num_nodes=len(nodes),
    )
    data.morph_known = int((data.x_morph != unknown_id).sum())
    return data
