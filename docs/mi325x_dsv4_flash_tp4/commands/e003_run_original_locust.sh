#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/models/mi325_dsv4_perf_tuning_20260803_065439
DATASET=/data/models/mi325_dsv4_reuse_tuning_20260801_103501/09_locust_router_baseline/locust_short_u32_20260801_130700/llm_test_datasets-prod
TEST_ID="${1:?test id required}"
USERS="${2:-70}"
DURATION="${3:-3m}"
OUT="$RUN_ROOT/locust/$TEST_ID"

test -d "$DATASET"
mkdir -p "$OUT"
sha256sum /data/DSV4-tune/locustfile.py > "$OUT/locustfile.sha256"
find "$DATASET" -maxdepth 1 -type f -name '*.json' | wc -l > "$OUT/dataset_file_count.txt"
curl -fsS http://127.0.0.1:18080/router/status > "$OUT/router_status_before.json"
docker run --rm --name "mi325_dsv4_${TEST_ID}_locust" --network host \
  -v "$RUN_ROOT/locust":/work \
  -v /data/DSV4-tune/locustfile.py:/work/locustfile.py:ro \
  -v "$DATASET":/work/dataset:ro \
  locust-awcloud:1.5 -lc \
  '/root/locust_test/.venv/bin/locust -f /work/locustfile.py --headless --host http://127.0.0.1:18080 -u '"$USERS"' -r '"$USERS"' -t '"$DURATION"' --stop-timeout 90 --csv /work/'"$TEST_ID"'/locust --csv-full-history --logfile /work/'"$TEST_ID"'/locust.log --loglevel INFO --exit-code-on-error 1 --model-name DeepSeek-V3.2 --max-tokens 1024 --include-usage true --dataset-dir /work/dataset --interval 50 --sla-interval 60' \
  | tee "$OUT/locust.stdout"
curl -fsS http://127.0.0.1:18080/router/status > "$OUT/router_status_after.json"
