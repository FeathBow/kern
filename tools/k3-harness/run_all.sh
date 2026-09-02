#!/usr/bin/env bash
# run_all.sh — run every naive kernel through the harness at the required
# shapes and print a table.  Also runs the retired pegainfer K5 as the timing
# baseline.  Exit status is non-zero if anything FAILs.
#
#   ./run_all.sh                 # everything
#   ./run_all.sh kda_core rms    # only these kernels
#   BIG=1 ./run_all.sh           # add the expensive corners (K5 ctx=32768 B=64)
#
# The harness prints a `RESULT\t...` line per run; this script collects them.
set -uo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
REPS=${REPS:-50}
LOG=${LOG:-run_all.log}
: > "$LOG"
sel=("$@")

want() {
  [ ${#sel[@]} -eq 0 ] && return 0
  for p in "${sel[@]}"; do [ "$p" = "$1" ] && return 0; done
  return 1
}

run() {   # run <kernel> <cubin> [extra args...]
  local k=$1 cubin=$2; shift 2
  want "$k" || return 0
  ./harness --kernel "$k" --cubin "$cubin" --reps "$REPS" "$@" >>"$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 2 ]; then
    printf 'RESULT\t%s\t%s\tERROR (harness/driver)\n' "$k" "$*" >>"$LOG"
  fi
}

N=naive
for B in 1 2 8 64; do
  for nbsnap in "0 0" "1 1" "4 1" "8 0"; do
    set -- $nbsnap
    run attnres_rms          $N/attnres_rms.cubin          --B $B --nb $1 --snapshot $2
    run land_add_attnres_rms $N/land_add_attnres_rms.cubin --B $B --nb $1 --snapshot $2
  done
  run land_add2  $N/land_add2.cubin  --B $B --two 0
  run land_add2  $N/land_add2.cubin  --B $B --two 1
  run conv_silu  $N/conv_silu.cubin  --B $B
  run kda_core   $N/kda_core.cubin   --B $B
  run mla_prep   $N/mla_prep.cubin   --B $B
  run mla_prep   $N/mla_prep.cubin   --B $B --nmla 2 --layer 1
  run router_topk $N/router_topk.cubin --B $B
  run argmax_f32 $N/argmax_f32.cubin --B $B
  run rms        $N/rms.cubin        --B $B
  run land       $N/land.cubin       --B $B
  run land_situ  $N/land_situ.cubin  --B $B
done

# K5: the documented acceptance shapes.
for ctx in 1 64 65 2048 32768; do
  for B in 1 8; do
    run mla_paged_attn $N/mla_paged_attn.cubin --B $B --ctx $ctx
  done
done
run mla_paged_attn $N/mla_paged_attn.cubin --B 2  --ctx 2048
run mla_paged_attn $N/mla_paged_attn.cubin --B 64 --ctx 2048
run mla_paged_attn $N/mla_paged_attn.cubin --B 1  --ctx 2048 --nmla 2 --layer 1
if [ "${BIG:-0}" = "1" ]; then
  run mla_paged_attn $N/mla_paged_attn.cubin --B 64 --ctx 32768
fi

# the baseline K5 authors must beat by 3x at ctx=32768, B=1
for ctx in 1 64 65 2048 32768; do
  for B in 1 8; do
    run mla_paged_attn_old baseline/mla_paged_attn_old.cubin --B $B --ctx $ctx
  done
done

printf '\n%-22s %-4s %-7s %-4s %-5s %-4s %-6s %12s %10s\n' \
  KERNEL B CTX NB SNAP TWO STATUS "MEDIAN us" "GB/s"
printf '%s\n' "-------------------------------------------------------------------------------------------"
awk -F'\t' '/^RESULT/{
  gsub(/^B=/,"",$3); gsub(/^ctx=/,"",$4); gsub(/^nb=/,"",$5);
  gsub(/^snap=/,"",$6); gsub(/^two=/,"",$7);
  gsub(/ us$/,"",$9); gsub(/ GB\/s$/,"",$10);
  printf "%-22s %-4s %-7s %-4s %-5s %-4s %-6s %12s %10s\n",$2,$3,$4,$5,$6,$7,$8,$9,$10;
  if($8!="PASS") bad++
} END { if (bad) printf "\n%d FAILING RUN(S)\n", bad; else printf "\nall runs PASS\n" }' "$LOG"

if grep -q 'FAIL\|ERROR' "$LOG"; then
  echo "see $LOG"
  exit 1
fi
exit 0
