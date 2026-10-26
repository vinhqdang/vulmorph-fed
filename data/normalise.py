"""
Format-normalised source: a canonical token stream.

Every leaf token of the tree-sitter parse (comments dropped) is emitted
separated by single spaces. Indentation, line breaks, blank lines, trailing
whitespace, tabs vs spaces and comments are all removed, while the tokens
themselves (identifiers, literals, operators) are untouched. A model that reads
this text can still use identifiers; it can no longer use formatting.
"""

from data.loaders.ast_graphs import _PARSER, TS_AVAILABLE

_ATOMIC = {"string_literal", "char_literal", "concatenated_string",
           "system_lib_string"}


def normalise_code(code: str) -> str:
    if not TS_AVAILABLE:
        return " ".join(code.split())
    src = code.encode("utf8", "replace")
    try:
        root = _PARSER.parse(src).root_node
    except Exception:
        return " ".join(code.split())
    toks = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "comment":
            continue
        if n.child_count == 0 or n.type in _ATOMIC:
            t = src[n.start_byte:n.end_byte].decode("utf8", "replace")
            t = " ".join(t.split()) if n.type not in _ATOMIC else t
            if t:
                toks.append((n.start_byte, t))
        else:
            stack.extend(n.children)
    toks.sort()
    return " ".join(t for _, t in toks)
