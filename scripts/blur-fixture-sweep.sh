#!/usr/bin/env bash
# QA / regression sweep: run the blur CLI on every video in a samples
# directory and write blurred copies + manifests to an output tree.
#
# Port of run_blur_owl_batch.sh from the senior engineer's bundle. Used
# to regenerate the reference fixture set against senior's validated
# sample videos before PR 6 lands in staging.
#
# Usage:
#   SAMPLES_DIR=/path/to/samples OUT_DIR=/path/to/output \
#     scripts/blur-fixture-sweep.sh
#
#   # Optional overrides (forwarded to the CLI):
#   OWL_STRIDE=5 SCORE_THRESHOLD=0.35 \
#     scripts/blur-fixture-sweep.sh
#
# Exit code is non-zero if any video fails — safe for CI.

set -euo pipefail

SAMPLES_DIR="${SAMPLES_DIR:-./samples}"
OUT_DIR="${OUT_DIR:-./output_owl}"
OWL_STRIDE="${OWL_STRIDE:-5}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.35}"

VIDEO_EXTENSIONS="mp4|mov|avi|mkv|webm|flv|wmv"

if [ ! -d "$SAMPLES_DIR" ]; then
    echo "error: SAMPLES_DIR does not exist: $SAMPLES_DIR" >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

count=0
failed=0

for video in "$SAMPLES_DIR"/*; do
    [ -f "$video" ] || continue

    ext="${video##*.}"
    ext_lower=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
    if ! echo "$ext_lower" | grep -qE "^($VIDEO_EXTENSIONS)$"; then
        echo "[SKIP] Not a video file: $(basename "$video")"
        continue
    fi

    stem="$(basename "${video%.*}")"
    job_dir="$OUT_DIR/$stem"
    mkdir -p "$job_dir"
    out_mp4="$job_dir/${stem}_blurred_owl.mp4"
    out_manifest="$job_dir/${stem}_manifest.json"

    count=$((count + 1))
    echo ""
    echo "========================================================================"
    echo "[$count] Processing: $(basename "$video")"
    echo "        → $out_mp4"
    echo "========================================================================"

    if python -m heimdex_media_pipelines blur process \
            --video "$video" \
            --out "$out_mp4" \
            --manifest "$out_manifest" \
            --owl-stride "$OWL_STRIDE" \
            --score-threshold "$SCORE_THRESHOLD"; then
        echo "[OK] $(basename "$video")"
    else
        echo "[FAIL] $(basename "$video")"
        failed=$((failed + 1))
    fi
done

echo ""
echo "========================================================================"
echo "Batch complete: $count video(s) processed, $failed failed"
echo "========================================================================"

exit $failed
