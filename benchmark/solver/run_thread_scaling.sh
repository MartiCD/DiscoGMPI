#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_RAW="benchmark/solver/thread_scaling_raw.txt"
OUT_CSV="benchmark/solver/thread_scaling.csv"

CSV_HEADER="timestamp,formulation,julia_version,threads,t_serial_min,t_serial_med,t_serial_mean,t_threaded_min,t_threaded_med,t_threaded_mean,serial_allocs,threaded_allocs,serial_memory,threaded_memory,speedup_min,speedup_med,speedup_mean,efficiency_med,abs_err,rel_err,correct"
THREAD_COUNTS="${THREAD_COUNTS:-1 2 4 8}"

rm -f "$OUT_RAW" "$OUT_CSV"

echo "$CSV_HEADER" > "$OUT_CSV"

for t in $THREAD_COUNTS; do
    echo "Running with $t threads..." | tee -a "$OUT_RAW"
    julia --project=. --threads="$t" benchmark/solver/benchmark_backends.jl "$@" | tee -a "$OUT_RAW"
done

grep '^CSV_RESULT,' "$OUT_RAW" | sed 's/^CSV_RESULT,//' >> "$OUT_CSV"

echo "Wrote raw output: $OUT_RAW"
echo "Wrote CSV:        $OUT_CSV"
