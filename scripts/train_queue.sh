#!/usr/bin/env bash
# Train a list of jobs sequentially. Each line of the jobs file: <model> <variant> <seed>
# Jobs whose model file already exists are skipped, so an interrupted queue can be restarted.
# Usage: bash scripts/train_queue.sh jobs.txt
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$1"
mkdir -p "$REPO/results/logs"
while read -r model variant seed; do
  [ -z "$model" ] && continue
  case "$model" in \#*) continue ;; esac
  dir="$REPO/results/models/${model}-${variant}/seed${seed}"
  if [ -f "$dir/model.pt" ] || [ -f "$dir/model.txt" ]; then
    echo "skip $model $variant $seed"; continue
  fi
  echo "$(date '+%F %T') start $model $variant $seed"
  (cd /tmp && python3 -I -m vigil.train --model "$model" --variant "$variant" --seed "$seed") \
    > "$REPO/results/logs/train_${model}-${variant}_seed${seed}.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') end   $model $variant $seed (exit $rc)"
done < "$JOBS"
