import re

"""API allow-lists for the phi_type mapping (single source of truth
shared by the token-level and AST-level classifiers)."""

API_CLASSES = {
    "MEMORY_ALLOC":   {"malloc", "calloc", "alloca", "valloc", "new"},
    "MEMORY_REALLOC": {"realloc", "reallocarray"},
    "MEMORY_FREE":    {"free", "cfree", "delete"},
    "MEMORY_COPY":    {"memcpy", "memmove", "bcopy"},
    "MEMORY_SET":     {"memset", "bzero", "explicit_bzero"},
    "STRING_COPY":    {"strcpy", "strncpy", "strlcpy", "wcscpy", "strdup", "strndup"},
    "STRING_CONCAT":  {"strcat", "strncat", "strlcat", "wcscat"},
    "STRING_FORMAT":  {"sprintf", "snprintf", "vsprintf", "vsnprintf",
                       "scanf", "sscanf", "fscanf", "printf", "fprintf", "vprintf"},
    "STRING_LENGTH":  {"strlen", "strnlen", "wcslen"},
    "IO_CALL":        {"read", "write", "pread", "pwrite", "fread", "fwrite",
                       "recv", "recvfrom", "send", "sendto", "fgets", "gets",
                       "fopen", "open", "close", "fclose"}
}


# ── Wrapped-allocator recognition ────────────────────────────────────────────
# Real codebases rarely call libc allocators directly: FFmpeg uses av_malloc,
# GLib g_malloc, the Linux kernel kmalloc/vmalloc, OpenSSL OPENSSL_malloc,
# QEMU qemu_memalign. An exact-match allow-list therefore routes almost every
# real allocation site to CALL_SITE, erasing the memory-safety semantics the
# morphology is meant to capture.
#
# Recognition is by IDENTIFIER COMPONENT, not substring. A callee name is split
# on underscores and camelCase boundaries and a rule fires only when a whole
# component matches. Substring matching was measurably wrong: it classified
# `__put_user` and `skb_put` (writes) as MEMORY_FREE via "_put", `gen_new_label`
# as MEMORY_ALLOC via "_new", `float64_is_zero` (a predicate) as MEMORY_SET via
# "_zero", and `reallocate_view` as MEMORY_REALLOC — while missing
# `copy_to_user`, a genuine bounded copy. Only the matched *class* is retained,
# never the identifier, so project invariance is preserved.
_COMPONENT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+")


def identifier_components(name: str):
    """Split an identifier into lowercase components (snake_case + camelCase)."""
    parts = []
    for chunk in name.split("_"):
        parts.extend(m.group(0).lower() for m in _COMPONENT_RE.finditer(chunk))
    return parts


# Whole-component needles. Deliberately conservative: a component must BE the
# operation word. Ambiguous words that caused false positives ("put", "new",
# "zero", "release", "destroy", "unref") are excluded — they describe ownership
# or lifecycle, not the memory operations the taxonomy models.
COMPONENT_RULES = [
    # dealloc/realloc are tested before alloc so they are not captured by it.
    (("free", "freep", "kfree", "vfree", "dealloc", "deallocate"), "MEMORY_FREE"),
    (("realloc", "reallocz"), "MEMORY_REALLOC"),
    (("malloc", "kmalloc", "vmalloc", "kzalloc", "zalloc", "calloc",
      "alloc", "memalign", "alloca"), "MEMORY_ALLOC"),
    (("memcpy", "memmove", "memdup", "bcopy", "copy", "strdup"), "MEMORY_COPY"),
    (("memset", "bzero"), "MEMORY_SET"),
    (("strcpy", "strncpy", "strlcpy", "wcscpy"), "STRING_COPY"),
    (("strcat", "strncat", "strlcat", "wcscat"), "STRING_CONCAT"),
    # Only the buffer-writing printf family is memory-relevant. Plain
    # printf/fprintf and project logging macros are I/O, and IO_CALL projects
    # to CALL_SITE rather than MEMORY_ACCESS.
    (("sprintf", "snprintf", "vsprintf", "vsnprintf", "sscanf"), "STRING_FORMAT"),
    (("printf", "fprintf", "puts", "scanf", "fscanf"), "IO_CALL"),
    (("strlen", "strnlen", "wcslen"), "STRING_LENGTH"),
]


def classify_api(name: str):
    """
    Map a callee name to an API class, or None.

    Two stages: exact match against the libc allow-lists, then whole-component
    matching for project-wrapped equivalents. Unmatched names (including all
    user-defined functions and macros) return None and become CALL_SITE.
    """
    if not name:
        return None
    for cls, names in API_CLASSES.items():
        if name in names:
            return cls
    comps = set(identifier_components(name))
    if not comps:
        return None
    for needles, cls in COMPONENT_RULES:
        if comps.intersection(needles):
            return cls
    return None
