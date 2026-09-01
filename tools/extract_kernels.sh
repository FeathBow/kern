#!/usr/bin/env bash
# Put every module a manifest pins into one kernel directory, by content.
#
#   tools/extract_kernels.sh <manifest.json> <dump_dir>[:<dump_dir>...] [out_dir=kernels]
#
# Every module the manifest's `modules` table names is pinned by sha256 (the
# verifier insists). This script finds each sha among the capture dumps, the
# handwritten builds (tools/build_kernels.sh -> target/cubins) and the repo's
# kernels/, and lands it as `<module name>-<sha12>.cubin` — readable in `ls`,
# unique per version. The runtime resolves by hash, never by name, so the
# directory only ever grows: extract A, extract B, and `kern test` loads both
# from it — the runtime loads only what the manifest pins, so the rest of
# the directory is inert. Dump dirs are searched recursively (a capture
# root holding `pid*/module_*.cubin`).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${1:?manifest.json}"
IFS=: read -r -a dumps <<<"${2:?dump dir(s)}"
out="${3:-$repo/kernels}"
build="$repo/target/cubins"
mkdir -p "$out"
"$repo/tools/build_kernels.sh" "$build"

# (module source, sha, one entry) per local module the manifest pins
mapfile -t wanted < <(python3 - "$manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
entry_of = {}
for op in m["ops"].values():
    for l in op["impl"]["launches"]:
        if "module" in l:
            entry_of.setdefault(l["module"], l["entry"])
for name, md in m["modules"].items():
    if not md["source"].startswith("hf:"):
        print(md["source"], md["sha256"], entry_of.get(name, "-"))
PY
)

# content index over every candidate cubin
declare -A by_sha
while read -r sha path; do
  by_sha[$sha]="${by_sha[$sha]:-$path}"
done < <(for d in "${dumps[@]}" "$build" "$repo/kernels" "$out"; do
           find "$d" -name '*.cubin' -print0 2>/dev/null | xargs -0 -r sha256sum; done)

# what the out dir already holds, by content (the same bytes never land twice)
declare -A landed
while read -r sha path; do landed[$sha]=$path; done < <(ls "$out"/*.cubin >/dev/null 2>&1 && sha256sum "$out"/*.cubin)

land() {  # land <src> <display name> <sha>
  local src="$1" name="$2" sha="$3"
  [ -n "${landed[$sha]:-}" ] && return
  local dst="$out/${name%.cubin}-${sha:0:12}.cubin"
  cp "$src" "$dst"; landed[$sha]=$dst; echo "  + $(basename "$dst")   ($sym)" >&2
}

missing=0
for entry in "${wanted[@]}"; do
  read -r cubin sha sym <<<"$entry"
  src="${by_sha[$sha]:-}"
  if [ -z "$src" ]; then
    echo "MISSING $cubin @${sha:0:12} ($sym): no file with that sha256 in ${dumps[*]}, $build, $repo/kernels" >&2
    if [ -f "$build/$cubin" ]; then
      echo "        $build/$cubin exists but hashes $(sha256sum "$build/$cubin" | cut -c1-12): a different build" \
           "(other nvcc / flags / source) — regenerate the manifest to pin this build, or rebuild the one it pins" >&2
    fi
    missing=$((missing+1)); continue
  fi
  land "$src" "$cubin" "$sha"
done
[ "$missing" -eq 0 ] || { echo "$missing pinned cubin(s) missing" >&2; exit 1; }
echo "$out: $(ls "$out"/*.cubin | wc -l) cubins" >&2
