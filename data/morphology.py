import torch
import torch.nn as nn

# The 8 abstract morphological types from the research plan
MORPHOLOGY_TYPES = [
    "MEMORY_ACCESS",  # malloc, free, memcpy, buffer reads/writes
    "ARRAY_INDEX",    # array/pointer indexing operations
    "PTR_DEREF",      # pointer dereferences (*p, ->)
    "CONTROL_BRANCH", # if/else, switch, ternary conditionals
    "ARITH_OP",       # arithmetic operations (+, -, *, /, %)
    "COMPARISON",     # relational operators (<, >, ==, !=)
    "CALL_SITE",      # function/method calls
    "ASSIGN",         # assignment statements
    "UNKNOWN"         # fallback for non-critical nodes
]

MORPHOLOGY_MAP = {k: i for i, k in enumerate(MORPHOLOGY_TYPES)}
NUM_MORPHOLOGY_TYPES = len(MORPHOLOGY_TYPES)

# A simplified mock mapping from C/C++ AST node types to the 8 morphological types
AST_TO_MORPHOLOGY = {
    "CallExpression": "CALL_SITE",
    "AssignmentExpression": "ASSIGN",
    "BinaryExpression": "ARITH_OP", # General arithmetic
    "Identifier": "UNKNOWN",
    "IfStatement": "CONTROL_BRANCH",
    "SwitchStatement": "CONTROL_BRANCH",
    "ForStatement": "CONTROL_BRANCH",
    "WhileStatement": "CONTROL_BRANCH",
    "ArraySubscriptExpression": "ARRAY_INDEX",
    "PointerDereference": "PTR_DEREF",
    "MemberExpression": "PTR_DEREF", # often -> in C/C++
    "ReturnStatement": "UNKNOWN",
    "FunctionDecl": "UNKNOWN",
    # Mocks for direct memory APIs
    "malloc": "MEMORY_ACCESS",
    "free": "MEMORY_ACCESS",
    "memcpy": "MEMORY_ACCESS",
    "memset": "MEMORY_ACCESS",
    # Comparators
    "<": "COMPARISON",
    ">": "COMPARISON",
    "==": "COMPARISON",
    "!=": "COMPARISON",
    "<=": "COMPARISON",
    ">=": "COMPARISON",
}

def get_morphology_id(ast_node_label: str) -> int:
    """Map a raw AST/CPG node label to its abstract morphology ID."""
    abstract_type = AST_TO_MORPHOLOGY.get(ast_node_label, "UNKNOWN")
    return MORPHOLOGY_MAP[abstract_type]

class MorphologyEmbedding(nn.Module):
    """
    Embedding layer for the abstract morphology types.
    This replaces project-specific lexical token embeddings to enable cross-project transfer.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(NUM_MORPHOLOGY_TYPES, embedding_dim)

    def forward(self, abstract_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            abstract_type_ids: Tensor of shape (num_nodes,) containing morphological IDs.
        Returns:
            Tensor of shape (num_nodes, embedding_dim)
        """
        return self.embedding(abstract_type_ids)
