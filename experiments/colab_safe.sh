#!/bin/bash
# Crash-safe wrapper around the colab CLI.
#
# The CLI rewrites its session-state file on exit. If the process is killed
# mid-write (e.g. a background monitor reaped by the harness), the file is
# truncated to "[]" and the session's auth token is lost forever: the VM keeps
# running server-side but can no longer be addressed or stopped, holding an
# account slot until Colab reclaims it. This wrapper snapshots the state file
# before each call and restores it if the call leaves it empty or unparsable.
#
# Usage: colab_safe.sh <session> <colab args...>
set -u
VM=$1; shift
# COLAB_HOME lets several Google accounts be driven concurrently: the CLI keeps
# its OAuth token and session records under $HOME/.config/colab-cli, so pointing
# HOME at a per-account directory gives each account fully isolated state.
if [ -n "${COLAB_HOME:-}" ]; then export HOME="$COLAB_HOME"; fi
CFG=/tmp/colabcfg_${COLAB_ACCT:-a}_$VM.json
BAK=$CFG.bak

valid() { python -c "
import json,sys
try:
    d=json.load(open('$1'))
    sys.exit(0 if isinstance(d,dict) and d else 1)
except Exception: sys.exit(1)"; }

if valid "$CFG"; then cp "$CFG" "$BAK"; fi
colab --config "$CFG" "$@"
rc=$?
if ! valid "$CFG" && [ -f "$BAK" ]; then
  cp "$BAK" "$CFG"
  echo "[colab_safe] restored truncated session state for $VM" >&2
fi
exit $rc
