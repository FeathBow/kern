#!/usr/bin/env bash
# Put every cubin a manifest pins into one kernel directory, by content.
#
#   tools/extract_kernels.sh <manifest.json> <dump_dir>[:<dump_dir>...] [out_dir=kernels]
#
# Every step that names a cubin pins its sha256 (the verifier insists). This
# script finds each sha among the capture dumps, the handwritten builds
# (tools/build_kernels.sh -> target/cubins) and the repo's kernels/, and
# lands it as `<display name>-<sha12>.cubin` — readable in `ls`, unique per
# version. The runtime resolves by hash, never by name, so the directory
# only ever grows: extract A, extract B, and kern-test loads both from it.
# Steps without a cubin (symbol-only, disambiguated at load by param
# layout) bring every dump module that defines the symbol.
#
# Different models' dumps should still not share an out_dir when their
# symbol-only instances collide (same symbol, same ABI, different constexpr).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${1:?manifest.json}"
IFS=: read -r -a dumps <<<"${2:?dump dir(s)}"
out="${3:-$repo/kernels}"
build="$repo/target/cubins"
mkdir -p "$out"
"$repo/tools/build_kernels.sh" "$build"

# (display name, sha, symbol) per step; "-" when absent
mapfile -t wanted < <(python3 - "$manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
keys = []
for k in m["kernels"].values():
    for st in k["impl"]["steps"]:
        s = st["symbol"]
        if s.startswith("extern:") or (st.get("cubin") or "").startswith("hf:"):
            continue
        key = (st.get("cubin") or "-", st.get("sha256") or "-", s)
        if key not in keys:
            keys.append(key)
# pinned first: a module that is both pinned (by name) and swept in by a
# symbol-only step lands under its pinned name
for key in sorted(keys, key=lambda k: k[1] == "-"):
    print(*key)
PY
)

# content index over every candidate cubin
declare -A by_sha
while read -r sha path; do
  by_sha[$sha]="${by_sha[$sha]:-$path}"
done < <(for d in "${dumps[@]}" "$build" "$repo/kernels" "$out"; do
           ls "$d"/*.cubin >/dev/null 2>&1 && sha256sum "$d"/*.cubin; done)

# what the out dir already holds, by content (the same bytes never land twice)
declare -A landed
while read -r sha path; do landed[$sha]=$path; done < <(ls "$out"/*.cubin >/dev/null 2>&1 && sha256sum "$out"/*.cubin)

land() {  # land <src> <display name> <sha>
  local src="$1" name="$2" sha="$3"
  [ -n "${landed[$sha]:-}" ] && return
  local dst="$out/${name%.cubin}-${sha:0:12}.cubin"
  cp "$src" "$dst"; landed[$sha]=$dst; echo "  + $(basename "$dst")   ($sym)" >&2
}

cuobjdump="${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump"
missing=0
for entry in "${wanted[@]}"; do
  read -r cubin sha sym <<<"$entry"
  if [ "$sha" != "-" ]; then
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
  else
    for d in "${dumps[@]}"; do
      for mod in "$d"/module_*.cubin; do
        [ -f "$mod" ] || continue
        names="$("$cuobjdump" -symbols "$mod" 2>/dev/null | awk '{print $NF}')" || continue
        if grep -qxF "$sym" <<<"$names"; then
          land "$mod" "$(basename "$mod")" "$(sha256sum "$mod" | cut -d' ' -f1)"
        fi
      done
    done
  fi
done
[ "$missing" -eq 0 ] || { echo "$missing pinned cubin(s) missing" >&2; exit 1; }
echo "$out: $(ls "$out"/*.cubin | wc -l) cubins" >&2
