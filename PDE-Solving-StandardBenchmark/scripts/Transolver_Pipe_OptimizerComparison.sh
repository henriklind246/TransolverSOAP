set -euo pipefail

DATA_PATH="${DATA_PATH:-/data/fno/pipe}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-500}"
SEED="${SEED:-0}"

COMMON_ARGS=(
  --gpu "${GPU}"
  --model Transolver_Structured_Mesh_2D
  --n-hidden 128
  --n-heads 8
  --n-layers 8
  --mlp_ratio 2
  --lr 0.001
  --max_grad_norm 0.1
  --batch-size 8
  --slice_num 64
  --unified_pos 0
  --ref 8
  --eval 0
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --data_path "${DATA_PATH}"
)

python exp_pipe.py \
  "${COMMON_ARGS[@]}" \
  --optimizer adamw \
  --save_name "pipe_adamw_seed${SEED}"

python exp_pipe.py \
  "${COMMON_ARGS[@]}" \
  --optimizer soap \
  --soap-precondition-frequency 10 \
  --save_name "pipe_soap_seed${SEED}"
