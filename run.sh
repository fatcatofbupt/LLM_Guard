#!/usr/bin/env bash
# Run answer_questions.py with default arguments.
# cd to script dir so the relative paths inside answer_questions.py resolve.

set -euo pipefail
cd "$(dirname "$0")"

python answer_questions.py
