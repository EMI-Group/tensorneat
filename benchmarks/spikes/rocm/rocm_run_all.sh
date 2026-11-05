#!/usr/bin/env bash
# Drive every ROCm spike probe: ONE graph per subprocess, each under `timeout 900`,
# because a failed HIP device is poisoned for the rest of the process.
#
#   bash benchmarks/spikes/rocm/rocm_run_all.sh [LOG_DIR]
#
# Everything is printed as `PROBE <name> <OK|FAIL> <detail>`; grep `^PROBE .* OK`
# for the measured table. Requires the local evoxc checkout:
#   uv sync --all-extras && uv pip install -e /root/evoxc --no-deps
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../../.." || exit 1
LOG_DIR="${1:-$PWD/benchmarks/spikes/rocm/logs}"
mkdir -p "$LOG_DIR"
PY="uv run --no-sync python"
SPIKES="benchmarks/spikes/rocm"

run() {  # run <label> <cmd...>
  local label="$1"; shift
  local log="$LOG_DIR/$label.log"
  timeout 900 "$@" >"$log" 2>&1
  local code=$?
  grep -v nanobind "$log" | grep '^PROBE .* \(OK\|FAIL\|BUILT\|FAILED\)' | tail -2
  echo "PROBE run_$label DONE exit=$code log=$log"
}

# ---------------------------------------------------------------- A: placement
for case in a1 a1_infer a1_hop a1_numpy a2numpy a2iree a3; do
  run "A_$case" $PY "$SPIKES/rocm_a_pipeline.py" --case "$case"
done

# ------------------------------------------------- B: stage verdicts on rocm
for case in b_init b_transform b_sched_tables b_sched_levels b_sched_levels_nocond \
            b_forward b_forward_schedule b_update_species b_nextgen b_speciate b_tell; do
  run "B_rocm_$case" $PY "$SPIKES/rocm_b_stages.py" --case "$case"
done

# ------------------------------------- B/C: same-stage cost, rocm vs numpy
for case in b_init b_sched_tables b_sched_levels_nocond b_forward_schedule \
            b_update_species b_nextgen; do
  run "B_rocm_timed_$case" $PY "$SPIKES/rocm_b_stages.py" --case "$case" \
      --backend rocm --repeats 10
  run "B_numpy_timed_$case" $PY "$SPIKES/rocm_b_stages.py" --case "$case" \
      --backend numpy --repeats 10
done

# ------------------------------------------------------- C: kernel microbench
for case in xor_p100 xor_p500 xor3d_p500 custom_pagie_p500 cartpole_p100; do
  for backend in rocm numpy; do
    run "C_level_step_${case}_$backend" $PY "$SPIKES/rocm_c_kernels.py" \
        --kernel level_step --case "$case" --backend "$backend" --repeats 10
    run "C_distance_row_${case}_$backend" $PY "$SPIKES/rocm_c_kernels.py" \
        --kernel distance_row --case "$case" --backend "$backend" --repeats 10
  done
done
for case in xor_p500 custom_pagie_p500 cartpole_p100; do
  for backend in rocm numpy; do
    run "C_level_step_x3_${case}_$backend" $PY "$SPIKES/rocm_c_kernels.py" \
        --kernel level_step_x3 --case "$case" --backend "$backend" --repeats 10
  done
done
for case in xor_p500 cartpole_p100; do
  run "C_forward_unrolled_${case}_numpy" $PY "$SPIKES/rocm_c_kernels.py" \
      --kernel forward_unrolled --case "$case" --backend numpy --repeats 3
done

echo "PROBE all DONE logs=$LOG_DIR"
