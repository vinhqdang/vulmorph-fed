"""
Morphological abstraction taxonomies for VulMorph-Fed.

Defines three nested semantic taxonomies (|T| = 8, 16, 32) used to map
concrete code tokens / CPG node labels onto project-invariant operation
types. The 8-type taxonomy is the default used in the paper; the 16- and
32-type variants support the taxonomy-size sensitivity analysis (RQ2b).

Every taxonomy additionally reserves one UNKNOWN type as a fallback, so a
taxonomy of size |T| yields |T| + 1 node categories and one-hot features
of dimension |T| + 1.
"""

import torch
import torch.nn as nn

# ── Taxonomy definitions ─────────────────────────────────────────────────────

# |T| = 8 (default, used throughout the paper)
TAXONOMY_8 = [
    "MEMORY_ACCESS",   # allocation/free/copy/set APIs (malloc, free, memcpy, ...)
    "ARRAY_INDEX",     # array subscript operations  a[i]
    "PTR_DEREF",       # pointer dereference / member access  *p, p->f
    "CONTROL_BRANCH",  # if/else/switch/loops/goto
    "ARITH_OP",        # arithmetic and bitwise operators
    "COMPARISON",      # relational operators
    "CALL_SITE",       # any function/method call
    "ASSIGN",          # assignment statements
]

# |T| = 16 (splits each coarse class into finer operation families)
TAXONOMY_16 = [
    "MEMORY_ALLOC",    # malloc, calloc, realloc, new
    "MEMORY_FREE",     # free, delete
    "MEMORY_COPY",     # memcpy, memmove, memset, bcopy
    "STRING_OP",       # strcpy, strcat, sprintf, strlen, ...
    "ARRAY_INDEX",
    "PTR_DEREF",       # *p
    "FIELD_ACCESS",    # p->f, s.f
    "LOOP",            # for, while, do
    "BRANCH",          # if, else, switch, case, ternary
    "JUMP",            # goto, break, continue, return
    "ARITH_OP",        # + - * / %
    "BITWISE_OP",      # & | ^ ~ << >>
    "COMPARISON",      # == != < > <= >=
    "LOGICAL_OP",      # && || !
    "CALL_SITE",
    "ASSIGN",
]

# |T| = 32 (finest granularity)
TAXONOMY_32 = [
    "MEMORY_ALLOC", "MEMORY_REALLOC", "MEMORY_FREE",
    "MEMORY_COPY", "MEMORY_SET",
    "STRING_COPY", "STRING_CONCAT", "STRING_FORMAT", "STRING_LENGTH",
    "IO_CALL",
    "ARRAY_INDEX",
    "PTR_DEREF", "ADDR_OF", "FIELD_ACCESS", "CAST",
    "LOOP_FOR", "LOOP_WHILE",
    "BRANCH_IF", "BRANCH_SWITCH",
    "JUMP_BREAK", "JUMP_GOTO", "RETURN",
    "ARITH_ADD", "ARITH_MUL", "ARITH_MOD",
    "BITWISE_OP", "SHIFT_OP",
    "COMPARISON_EQ", "COMPARISON_REL",
    "LOGICAL_OP",
    "CALL_SITE", "ASSIGN",
]

TAXONOMIES = {8: TAXONOMY_8, 16: TAXONOMY_16, 32: TAXONOMY_32}


def get_taxonomy(size: int = 8):
    """Return (type_list_with_unknown, {name: id}) for taxonomy of |T| = size."""
    if size not in TAXONOMIES:
        raise ValueError(f"Unsupported taxonomy size {size}; choose from {sorted(TAXONOMIES)}")
    types = TAXONOMIES[size] + ["UNKNOWN"]
    return types, {name: i for i, name in enumerate(types)}


# Default (|T| = 8) globals kept for backwards compatibility.
MORPHOLOGY_TYPES, MORPHOLOGY_MAP = get_taxonomy(8)
NUM_MORPHOLOGY_TYPES = len(MORPHOLOGY_TYPES)   # = |T| + 1 = 9 (incl. UNKNOWN)


class MorphologyEmbedding(nn.Module):
    """
    Embedding layer for the abstract morphology types (|T| + 1 categories,
    including the reserved UNKNOWN type). Replaces project-specific lexical
    token embeddings to enable cross-project transfer.
    """

    def __init__(self, embedding_dim: int, num_types: int = NUM_MORPHOLOGY_TYPES):
        super().__init__()
        self.num_types = num_types
        self.embedding = nn.Embedding(num_types, embedding_dim)

    def forward(self, abstract_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            abstract_type_ids: (num_nodes,) morphological type ids in [0, num_types).
        Returns:
            (num_nodes, embedding_dim)
        """
        return self.embedding(abstract_type_ids)
