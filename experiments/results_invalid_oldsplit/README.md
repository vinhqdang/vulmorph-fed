# INVALID — do not use in the manuscript

These results were produced before two corrections:

1. `split_by_project` sorted projects by size after shuffling, so the
   cross-project split was byte-identical for every seed. Reported standard
   deviations therefore captured only initialisation/DP-noise variance, not
   split variance.
2. `classify_api` matched libc allocator names exactly, so project-wrapped
   allocators (`av_malloc`, `kmalloc`, `OPENSSL_malloc`, ...) were classified
   as generic `CALL_SITE`. `MEMORY_ACCESS` fired on only 0.23% of AST nodes,
   leaving the framework's central mechanism effectively untested.

Retained only for provenance.
