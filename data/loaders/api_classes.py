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
