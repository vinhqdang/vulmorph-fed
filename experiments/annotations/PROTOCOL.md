# Blind annotation protocol for the callee-classification validation (RQ1)

Each annotator labels every row of three sheets (`blind_devign.tsv`,
`blind_bigvul.tsv`, `blind_diversevul.tsv`) independently, without seeing the
analysis' predicted class, without seeing any other annotator's labels, and
without reading the analysis source code. Sheets contain a callee name, how
often it is called in the sample, and one example call expression.

Label each callee with exactly one class, judging from the documented or
evident semantics of the callee (use the example call and your knowledge of
C libraries and the projects concerned -- FFmpeg, QEMU, Linux kernel, Chrome,
Android, OpenSSL, GLib, ...):

| label | criterion |
|---|---|
| MEMORY_ALLOC | its purpose is to obtain a heap allocation (malloc, calloc, alloca-like, `*_malloc`, `*_alloc`, `kmalloc`, ...) that the CALLER then owns |
| MEMORY_REALLOC | resizes such an allocation |
| MEMORY_FREE | releases such an allocation (free, kfree, `*_free`, `*_release` of owned memory) |
| MEMORY_COPY | writes a caller-supplied destination buffer of caller-specified length from a source (memcpy, memmove, copy_from_user, `*_copy` of raw bytes) |
| MEMORY_SET | fills a caller-supplied destination buffer with a value (memset, bzero, `*_memset`, `*_zero` of a buffer) |
| STRING_COPY | copies a NUL-terminated string into a destination (strcpy, strncpy, strlcpy, strdup-like) |
| STRING_CONCAT | appends a string (strcat, strncat, strlcat) |
| STRING_FORMAT | formats or scans INTO or FROM a caller-supplied buffer (sprintf, snprintf, vsnprintf, sscanf, scanf family). printf/fprintf write to a STREAM and are IO_CALL, not STRING_FORMAT |
| STRING_LENGTH | returns a string's length without writing (strlen, strnlen, wcslen) |
| IO_CALL | transfers bytes to or from a file descriptor, stream, socket, device or monitor (read, write, fopen, fprintf, printf, send, logging/print wrappers) |
| NONE | everything else: accessors, predicates, arithmetic helpers, constructors that do not return raw memory, register or IR helpers, tracing macros, ambiguous local helpers |

Rules of thumb. A callee that both allocates and does something else takes
the allocation label. When you genuinely cannot tell what a name does and the
example does not resolve it, label NONE and write "unsure" in the note column.
Do not look anything up in this repository other than your own sheets.

Output: for each sheet write `annotator_<ID>_<corpus>.tsv` with header
`id<TAB>label<TAB>note` and one row per sheet row (all ids, same order).
