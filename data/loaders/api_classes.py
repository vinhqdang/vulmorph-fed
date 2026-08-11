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
# real allocation site to CALL_SITE, which erases exactly the memory-safety
# semantics the morphology is meant to capture. These affix rules recover them
# without reintroducing project-specific lexical information: only the matched
# *class* is kept, never the identifier.
AFFIX_RULES = [
    # Order is significant: deallocation and reallocation are tested before
    # allocation so that `dealloc`/`realloc` are not captured by the broader
    # "alloc" needle.
    (("_free", "free_", "kfree", "vfree", "dealloc", "_release", "_destroy",
      "_unref", "_put"), "MEMORY_FREE"),
    (("realloc",), "MEMORY_REALLOC"),
    (("malloc", "alloc", "memalign", "_new"), "MEMORY_ALLOC"),
    (("memcpy", "memmove", "memdup", "_copy", "bcopy"), "MEMORY_COPY"),
    (("memset", "bzero", "_zero"), "MEMORY_SET"),
    (("strcpy", "strncpy", "strlcpy", "strdup", "wcscpy"), "STRING_COPY"),
    (("strcat", "strncat", "strlcat", "wcscat"), "STRING_CONCAT"),
    (("sprintf", "snprintf", "printf", "scanf"), "STRING_FORMAT"),
    (("strlen", "strnlen", "wcslen"), "STRING_LENGTH"),
]


def classify_api(name: str):
    """
    Map a callee name to an API class, or None. Exact libc match first, then
    affix rules for project-wrapped allocators/copies. Order matters: the
    realloc/free tests run before the generic alloc test so that e.g.
    `av_realloc` is not captured by the `_alloc` rule.
    """
    for cls, names in API_CLASSES.items():
        if name in names:
            return cls
    low = name.lower()
    for needles, cls in AFFIX_RULES:
        if any(nd in low for nd in needles):
            return cls
    return None
