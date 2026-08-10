#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/models/mi325_dsv4_perf_tuning_20260803_065439
TEST_ID="${1:?test id required}"
OUT="$RUN_ROOT/atom_benchmark/$TEST_ID"
mkdir -p "$OUT"
curl -fsS http://127.0.0.1:18000/v1/models > "$OUT/models_before.json"
docker run --rm --name "mi325_dsv4_${TEST_ID}_atom" --network host --device /dev/kfd --device /dev/dri --group-add video \
  -v /data/DeepSeek-V4-Flash-FP8:/data/DeepSeek-V4-Flash-FP8:ro \
  -v "$RUN_ROOT/atom_benchmark":/work \
  rocm/atom-dev:nightly_202607271535 bash -lc '
set -euo pipefail
cd /app/ATOM
python -m atom.benchmarks.benchmark_serving \
  --backend openai --base-url http://127.0.0.1:18000 --endpoint /v1/completions \
  --model /data/DeepSeek-V4-Flash-FP8 --served-model-name DeepSeek-V3.2 \
  --tokenizer /data/DeepSeek-V4-Flash-FP8 --dataset-name random \
  --random-input-len 6100 --random-output-len 1024 --num-prompts 140 --num-warmups 10 \
  --max-concurrency 70 --request-rate inf --ignore-eos --seed 20260802 --disable-tqdm \
  --save-result --save-detailed --result-dir /work/'"$TEST_ID"' --result-filename result.json' \
  | tee "$OUT/benchmark.stdout"
jq -e '.total_token_throughput and .mean_ttft_ms and .mean_tpot_ms' "$OUT/result.json" > "$OUT/summary.json"
curl -fsS http://127.0.0.1:18000/v1/models > "$OUT/models_after.json"
