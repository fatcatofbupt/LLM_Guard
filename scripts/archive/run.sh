#!/usr/bin/env bash
# Run Qwen3Guard-Gen inference on test_questions.json.
# Usage:
#   ./run.sh                        # defaults to 0.6B
#   ./run.sh --variant 8B           # use 8B model
#   ./run.sh --variant 0.6B --limit 10 --batch-size 4
#   ./run.sh --model ./models/Qwen/Qwen3Guard-Gen-8B --output ./out.json
set -euo pipefail
cd "$(dirname "$0")"

python3 answer_questions.py "$@"
