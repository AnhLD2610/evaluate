#!/bin/bash
set -e
# GPUS=0,1,2,3 bash run_eval_all.sh
# ============================================================
# Config - Modify these as needed
# ============================================================
MODEL_NAME="${MODEL_NAME:-Zigeng/R1-VeriThinker-7B}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"

# k values to evaluate
K_VALUES=(1 2 4 8 16 32 64 128)

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_SHORT=$(basename "$MODEL_NAME")
OUT_DIR="results/${MODEL_SHORT}_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES="$GPUS"

echo "============================================================"
echo " Evaluation Config"
echo "============================================================"
echo " Model:       $MODEL_NAME"
echo " GPUs:        $GPUS (tp=$TP)"
echo " K values:    ${K_VALUES[*]}"
echo " Output dir:  $OUT_DIR"
echo "============================================================"

COMMON_ARGS="--model_name $MODEL_NAME --tp $TP --temperature $TEMPERATURE --top_p $TOP_P --max_tokens $MAX_TOKENS --max_model_len $MAX_MODEL_LEN"

for N in "${K_VALUES[@]}"; do
    echo ""
    echo "########################################################"
    echo "  n = $N"
    echo "########################################################"

    # --- MATH-500 ---
    echo ">>> [n=$N] MATH-500 ..."
    python eval_math500.py $COMMON_ARGS --n $N \
        --output "$OUT_DIR/math500_n${N}.json" \
        2>&1 | tee "$OUT_DIR/math500_n${N}.log"

    # --- AIME 2024 ---
    echo ">>> [n=$N] AIME 2024 ..."
    python eval_aime24.py $COMMON_ARGS --n $N \
        --output "$OUT_DIR/aime24_n${N}.json" \
        2>&1 | tee "$OUT_DIR/aime24_n${N}.log"

    # --- AIME 2025 ---
    echo ">>> [n=$N] AIME 2025 ..."
    python eval_aime25.py $COMMON_ARGS --n $N \
        --output "$OUT_DIR/aime25_n${N}.json" \
        2>&1 | tee "$OUT_DIR/aime25_n${N}.log"

done

# ============================================================
# Final Summary
# ============================================================
echo ""
echo "============================================================"
echo " SUMMARY - All pass@k results"
echo "============================================================"

for DATASET in math500 aime24 aime25; do
    echo ""
    echo "--- ${DATASET} ---"
    for N in "${K_VALUES[@]}"; do
        LOG="$OUT_DIR/${DATASET}_n${N}.log"
        if [ -f "$LOG" ]; then
            RESULT=$(grep "pass@" "$LOG" | tail -1)
            echo "  n=${N}: ${RESULT}"
        fi
    done
done

echo ""
echo "All results saved in: $OUT_DIR"
