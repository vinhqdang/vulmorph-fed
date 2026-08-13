"""
E2, step 2 - annotate the phi_type gold set and score it as a static analysis.

Ground truth is assigned from documented API semantics under criteria stated
below, so that a reader can disagree with a specific label rather than with an
opaque number. The criteria matter more than any individual call:

  MEMORY_ALLOC/REALLOC/FREE  the callee's documented purpose is to obtain,
                             resize or release a heap allocation whose
                             lifetime the CALLER then owns.
  MEMORY_COPY/SET            the callee writes a caller-supplied destination
                             buffer of caller-specified length.
  STRING_COPY/CONCAT/FORMAT  as above, for NUL-terminated strings; note that
                             printf/fprintf write to a STREAM and are IO_CALL.
  STRING_LENGTH              reads a string's length without writing.
  IO_CALL                    transfers bytes to or from a file descriptor,
                             stream or socket.
  NONE                       everything else, including accessors, predicates,
                             arithmetic helpers, object construction that does
                             not return raw memory, and register/IR helpers.

Judgement calls are recorded explicitly in JUDGEMENT below with the reasoning,
because they are the cases a reviewer will probe. Deliberately conservative:
where a call both allocates and does something else, the allocation label is
taken, since that is the memory effect the taxonomy exists to capture.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
from collections import defaultdict

from data.loaders.api_classes import classify_api

# Names whose ground truth differs from a naive reading, with the reason.
JUDGEMENT = {
    # QEMU frees an IR temporary in its own allocator discipline. It IS a
    # release of a managed resource, and the taxonomy models release, so we
    # label it MEMORY_FREE and flag it as the boundary case it is.
    "tcg_temp_free": ("MEMORY_FREE", "releases an IR temporary, not heap"),
    "tcg_temp_free_i32": ("MEMORY_FREE", "releases an IR temporary"),
    "tcg_temp_free_i64": ("MEMORY_FREE", "releases an IR temporary"),
    "tcg_temp_free_internal": ("MEMORY_FREE", "releases an IR temporary"),
    # Allocates a formatted string: both allocation and formatting. Single
    # label taken as allocation, the stronger memory effect.
    "g_strdup_printf": ("MEMORY_ALLOC", "allocates and formats; alloc taken"),
    # Frees a struct, not a raw buffer, but the caller owned it.
    "error_free": ("MEMORY_FREE", "frees a caller-owned Error object"),
    "v9fs_string_free": ("MEMORY_FREE", "frees a caller-owned string struct"),
    "av_free_packet": ("MEMORY_FREE", "frees packet-owned buffers"),
    "av_frame_free": ("MEMORY_FREE", "frees a frame and its buffers"),
    "scsi_req_free": ("MEMORY_FREE", "frees a request object"),
    "guest_phys_blocks_free": ("MEMORY_FREE", "frees a list's memory"),
    "frame_thread_free": ("MEMORY_FREE", "frees per-thread frame state"),
    "ff_nut_free_sp": ("MEMORY_FREE", "frees syncpoint storage"),
    "qapi_free_AltStrBool": ("MEMORY_FREE", "generated deallocator"),
    "qapi_free_MemoryDeviceInfoList": ("MEMORY_FREE", "generated deallocator"),
    # NOT a free: registers a callback that will free elements later.
    "g_ptr_array_new_with_free_func": (
        "MEMORY_ALLOC", "allocates an array; the free_func is a callback"),
    # Copy-family judgements.
    "av_dict_copy": ("MEMORY_COPY", "deep-copies a dictionary"),
    "avfilter_copy_frame_props": ("MEMORY_COPY", "copies frame metadata"),
    "copy_from_user_timeval": ("MEMORY_COPY", "bounded copy across boundary"),
    "memcpy_to_target": ("MEMORY_COPY", "bounded copy to target memory"),
    "init_thread_copy": ("NONE", "initialises thread state; not a buffer copy"),
    "copy_chapters": ("NONE", "copies chapter records, not raw memory"),
    "copy_bits": ("NONE", "bit-level helper, no caller buffer"),
    # SIMD/lane extraction macros: read a lane, do not write a caller buffer.
    "__msa_copy_u_w": ("NONE", "SIMD lane extraction"),
    "__msa_copy_u_h": ("NONE", "SIMD lane extraction"),
    "__msa_copy_s_w": ("NONE", "SIMD lane extraction"),
    "AV_COPY32": ("MEMORY_COPY", "copies 32 bits between buffers"),
    "COPY": ("NONE", "context-dependent macro"),
    "COPY16TO8": ("NONE", "pixel format conversion macro"),
    "COPY16TO9_OR_10": ("NONE", "pixel format conversion macro"),
    "copy": ("NONE", "ambiguous local helper"),
    # Allocation-family judgements.
    "trace_xics_alloc_block": ("NONE", "tracepoint, not an allocation"),
    "bios_linker_loader_alloc": ("MEMORY_ALLOC", "allocates a loader entry"),
    "gencb_alloc": ("MEMORY_ALLOC", "allocates a callback object"),
    "qemu_chr_alloc": ("MEMORY_ALLOC", "allocates a chardev"),
    "ff_alloc_packet": ("MEMORY_ALLOC", "allocates packet payload"),
    "avcodec_alloc_frame": ("MEMORY_ALLOC", "allocates a frame"),
    "av_hwdevice_ctx_alloc": ("MEMORY_ALLOC", "allocates a device context"),
    "realloc_texture": ("MEMORY_REALLOC", "resizes a texture allocation"),
    "swri_realloc_audio": ("MEMORY_REALLOC", "resizes an audio buffer"),
    "av_audio_fifo_realloc": ("MEMORY_REALLOC", "resizes a FIFO"),
    "av_buffer_realloc": ("MEMORY_REALLOC", "resizes a buffer"),
    "av_realloc_array": ("MEMORY_REALLOC", "resizes an array allocation"),
    "alloc": ("MEMORY_ALLOC", "generic allocator wrapper"),
    # Formatting / IO judgements.
    "v9fs_string_sprintf": ("STRING_FORMAT", "writes into a caller string"),
    "target_strlen": ("STRING_LENGTH", "length of a target string"),
    "PCIE_DEV_PRINTF": ("IO_CALL", "logging macro"),
    "DB_PRINT_L": ("IO_CALL", "logging macro"),
    "error_printf": ("IO_CALL", "writes to the error stream"),
    "mon_printf": ("IO_CALL", "writes to the monitor"),
    "monitor_printf": ("IO_CALL", "writes to the monitor"),
    "xen_be_printf": ("IO_CALL", "logging"),
    "avio_printf": ("IO_CALL", "writes to an AVIO stream"),
    "cpu_fprintf": ("IO_CALL", "writes to a stream"),
    "bdrv_pread": ("IO_CALL", "block-device positioned read"),
    "qemu_fflush": ("IO_CALL", "flushes a stream"),
    "get_user_u64": ("MEMORY_COPY", "bounded copy across the user boundary"),
}

# Exact libc semantics; no judgement required.
LIBC = {
    "malloc": "MEMORY_ALLOC", "calloc": "MEMORY_ALLOC", "alloca": "MEMORY_ALLOC",
    "realloc": "MEMORY_REALLOC", "free": "MEMORY_FREE",
    "memcpy": "MEMORY_COPY", "memmove": "MEMORY_COPY", "memset": "MEMORY_SET",
    "strcpy": "STRING_COPY", "strncpy": "STRING_COPY", "strdup": "STRING_COPY",
    "strcat": "STRING_CONCAT", "strncat": "STRING_CONCAT",
    "sprintf": "STRING_FORMAT", "snprintf": "STRING_FORMAT",
    "vsnprintf": "STRING_FORMAT", "sscanf": "STRING_FORMAT",
    "scanf": "STRING_FORMAT", "fscanf": "STRING_FORMAT",
    "strlen": "STRING_LENGTH",
    "printf": "IO_CALL", "fprintf": "IO_CALL",
    "read": "IO_CALL", "write": "IO_CALL", "pread": "IO_CALL",
    "pwrite": "IO_CALL", "fread": "IO_CALL", "fwrite": "IO_CALL",
    "open": "IO_CALL", "close": "IO_CALL", "fopen": "IO_CALL",
    "fclose": "IO_CALL", "fgets": "IO_CALL", "send": "IO_CALL",
    "sendto": "IO_CALL",
}

# Project wrappers whose semantics follow directly from the wrapped call.
WRAPPER_PREFIXES = {
    "av_malloc": "MEMORY_ALLOC", "av_fast_malloc": "MEMORY_ALLOC",
    "av_fifo_alloc": "MEMORY_ALLOC", "av_frame_alloc": "MEMORY_ALLOC",
    "g_malloc": "MEMORY_ALLOC", "qemu_malloc": "MEMORY_ALLOC",
    "qemu_ram_alloc": "MEMORY_ALLOC", "png_malloc": "MEMORY_ALLOC",
    "checkasm_malloc": "MEMORY_ALLOC",
    "av_realloc": "MEMORY_REALLOC", "g_realloc": "MEMORY_REALLOC",
    "qemu_realloc": "MEMORY_REALLOC", "av_fast_realloc": "MEMORY_REALLOC",
    "av_free": "MEMORY_FREE", "av_freep": "MEMORY_FREE",
    "g_free": "MEMORY_FREE", "qemu_free": "MEMORY_FREE",
    "g_strdup": "MEMORY_COPY", "av_strdup": "MEMORY_COPY",
    "qemu_iovec_memset": "MEMORY_SET",
    "av_strlcpy": "STRING_COPY", "g_strlcpy": "STRING_COPY",
    "av_strlcat": "STRING_CONCAT", "g_strlcat": "STRING_CONCAT",
}


def ground_truth(name: str):
    if name in JUDGEMENT:
        return JUDGEMENT[name]
    if name in LIBC:
        return LIBC[name], ""
    if name in WRAPPER_PREFIXES:
        return WRAPPER_PREFIXES[name], ""
    return "NONE", ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goldset", default="experiments/goldset_devign.tsv")
    p.add_argument("--output", default="experiments/results/phitype_eval.json")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.goldset), delimiter="\t"))
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    errors = []
    weighted_tp = weighted_total = 0

    for r in rows:
        name = r["name"]
        occ = int(r["occurrences"])
        pred = classify_api(name) or "NONE"
        truth, note = ground_truth(name)
        r["true"] = truth
        r["note"] = note
        weighted_total += occ
        if pred == truth:
            if truth != "NONE":
                tp[truth] += 1
            weighted_tp += occ
        else:
            if pred != "NONE":
                fp[pred] += 1
            if truth != "NONE":
                fn[truth] += 1
            errors.append({"name": name, "occurrences": occ,
                           "predicted": pred, "true": truth, "note": note})

    classes = sorted(set(list(tp) + list(fp) + list(fn)))
    per_class = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else float("nan")
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec and (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "tp": tp[c], "fp": fp[c], "fn": fn[c]}

    macro_p = sum(v["precision"] for v in per_class.values()) / max(1, len(per_class))
    macro_r = sum(v["recall"] for v in per_class.values()) / max(1, len(per_class))

    print(f"{len(rows)} annotated names; "
          f"call-site-weighted accuracy {weighted_tp/max(1,weighted_total):.3f}")
    print(f"{'class':<18} {'prec':>6} {'rec':>6} {'F1':>6}  tp/fp/fn")
    for c, v in per_class.items():
        print(f"  {c:<16} {v['precision']:>6.3f} {v['recall']:>6.3f} "
              f"{v['f1']:>6.3f}  {v['tp']}/{v['fp']}/{v['fn']}")
    print(f"  {'MACRO':<16} {macro_p:>6.3f} {macro_r:>6.3f}")
    print(f"\n{len(errors)} disagreements; largest by call volume:")
    for e in sorted(errors, key=lambda x: -x["occurrences"])[:10]:
        print(f"   {e['name']:<30} x{e['occurrences']:<5} "
              f"pred={e['predicted']:<15} true={e['true']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_class": per_class, "macro_precision": macro_p,
               "macro_recall": macro_r, "n_names": len(rows),
               "weighted_accuracy": weighted_tp / max(1, weighted_total),
               "errors": errors}, open(out, "w"), indent=2)
    with open(args.goldset, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "occurrences", "predicted",
                                          "true", "note"], delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
