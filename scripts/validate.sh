#!/usr/bin/env bash
# One-command wrapper around the categorizer-validation workflow.
#
#   ./scripts/validate.sh generate   # 1. make outputs + a worklist to label  (needs API keys)
#   #  -> edit worklist.json: set "label" on each item (cited / uncited-confident /
#   #     uncited-hedged / abstained). This human step is the whole point; it can't be
#   #     automated without making the result circular and meaningless.
#   ./scripts/validate.sh finish     # 2. assemble + score + validate-judge     (needs API keys)
#
# Override any of these via env vars before the command, e.g.:
#   MODEL=openai:gpt-4o JUDGE_A=anthropic:claude-sonnet-4-6 JUDGE_B=anthropic:claude-opus-4-1 \
#     ./scripts/validate.sh generate
set -euo pipefail
cd "$(dirname "$0")/.."

# The model whose categorizer you are validating (used by `coa score`).
MODEL="${MODEL:-anthropic:claude-sonnet-4-6}"
# Two labeling judges — MUST be a different family from MODEL (non-circularity).
JUDGE_A="${JUDGE_A:-openai:gpt-4o}"
JUDGE_B="${JUDGE_B:-openai:gpt-4o-mini}"
# Models that produce the outputs (diversity is good; does not affect circularity).
GENERATORS="${GENERATORS:-anthropic:claude-sonnet-4-6 openai:gpt-4o openai:gpt-4o-mini}"
CORPUS="${CORPUS:-examples/corpus}"          # grounds generation so citations are checkable
PROMPTS="${PROMPTS:-examples/validation_prompts.json}"
AUDIT_FRAC="${AUDIT_FRAC:-0.15}"

_require_keys() {
  : "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}" "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
}

_provider_model() { echo "${1%%:*} ${1#*:}"; }  # "openai:gpt-4o" -> "openai gpt-4o"

case "${1:-}" in
  ingest)
    # Path A: you already have outputs. INPUTS=myoutputs.json ./scripts/validate.sh ingest
    : "${INPUTS:?set INPUTS to a JSON/CSV of {id,prompt,output}}"
    python scripts/build_validation_set.py ingest --inputs "$INPUTS" \
      --out-candidates candidates.json --out-worklist worklist.json
    echo
    echo "NEXT: open worklist.json, set \"label\" on each item, then: ./scripts/validate.sh finish"
    ;;
  generate)
    _require_keys
    python scripts/build_validation_set.py generate \
      --prompts "$PROMPTS" \
      --generators $GENERATORS \
      --judge-a "$JUDGE_A" --judge-b "$JUDGE_B" \
      --corpus "$CORPUS" --audit-frac "$AUDIT_FRAC" \
      --out-candidates candidates.json --out-worklist worklist.json
    echo
    echo "NEXT: open worklist.json, set \"label\" on each item, then: ./scripts/validate.sh finish"
    ;;
  finish)
    _require_keys
    python scripts/build_validation_set.py assemble \
      --candidates candidates.json --labels worklist.json \
      --out examples/validation_set.json
    read -r provider model <<<"$(_provider_model "$MODEL")"
    coa score --cases examples/validation_set.json --corpus "$CORPUS" \
      --provider "$provider" --model "$model" -o report.json --html report.html
    coa validate-judge --report report.json
    echo
    echo "Fill the accuracy + Gwet AC1 (above) and coverage/conf-error (score table) into"
    echo "the README 'Validation results' table."
    ;;
  *)
    echo "usage: ./scripts/validate.sh {ingest|generate|finish}" >&2
    echo "  ingest    INPUTS=outputs.json ...   (you already have model outputs; no keys)" >&2
    echo "  generate                            (start fresh from the prompt bank; needs keys)" >&2
    echo "  finish                              (assemble + score + validate-judge; needs keys)" >&2
    exit 2
    ;;
esac
