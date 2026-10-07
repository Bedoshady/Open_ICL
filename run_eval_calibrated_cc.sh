#!/bin/bash
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-03:00:00
#SBATCH --account=def-rsadve
#SBATCH --array=0-4
#SBATCH --output=evalcal-%A_%a.out

# ===========================================================================
#  Calibrated evaluation (EVALUATION ONLY - nothing is retrained)
#
#  Same model, same weights, same USB, same decision rule. The only change is
#  how the novelty threshold tau is computed:
#
#     evaluate.py  :  tau = mu + 0.5*sigma
#     this script  :  tau = (1-alpha) percentile of the novelty scores of the
#                          KNOWN TRAINING signals the model was trained on,
#                          i.e. tau is set so that alpha of the known training
#                          signals would be flagged unknown.
#
#  Calibration uses only training-split known signals, so there is NO overlap
#  with the evaluation set. The realised false-alarm rate on validation knowns
#  is reported next to every row so the calibration is fully transparent.
#
#  Three false-alarm rates are reported for every checkpoint (alpha = 1%, 5%,
#  10%) so nothing can be read as cherry-picking, plus the old mu+0.5sigma rule
#  for direct comparison.
#
# ---------------------------------------------------------------------------
#  Usage:
#     sbatch run_eval_calibrated_cc.sh
#     sbatch run_eval_calibrated_cc.sh rml2018 checkpoints 42,43,44
#
#  args:  1 = dataset type (rml2018 | rml2016)      default rml2018
#         2 = checkpoint root                       default checkpoints
#         3 = seeds to include                      default 42,43,44
#
#  Other locations:
#     CKPT_ROOT=$SCRATCH/open_set_modulation/checkpoints \
#     DATASET_PATH=$SCRATCH/GOLD_XYZ_OSC.0001_1024.hdf5 \
#     sbatch run_eval_calibrated_cc.sh
# ===========================================================================

module load StdEnv/2023
module load python/3.12
module load cuda/12.2
module load scipy-stack

if [ ! -d "myDRLenv" ]; then
    virtualenv --no-download myDRLenv
    source myDRLenv/bin/activate
    pip install --no-index torch torchvision torchtext torchaudio
    pip install scikit-learn numpy
else
    source myDRLenv/bin/activate
fi

DATASET_TYPE=${1:-rml2018}
CKPT_ROOT=${2:-${CKPT_ROOT:-checkpoints}}
SEEDS=${3:-42,43,44}

if [ "$DATASET_TYPE" = "rml2018" ]; then
    declare -a names=("set1_2018" "set2_2018" "set3_2018" "set4_2018" "set5_2018")
    DATASET_PATH=${DATASET_PATH:-GOLD_XYZ_OSC.0001_1024.hdf5}
else
    declare -a names=("set1_random" "set2_random" "set3_random" "set4_random" "set5_random")
    DATASET_PATH=${DATASET_PATH:-RML2016.10a_dict.pkl}
fi

i=${SLURM_ARRAY_TASK_ID:-0}
name="${names[$i]}"
CKDIR="${CKPT_ROOT}/${name}"

echo "========================================================="
echo " Calibrated evaluation (no retraining)"
echo "   split        : $name"
echo "   dataset type : $DATASET_TYPE"
echo "   dataset path : $DATASET_PATH"
echo "   checkpoints  : $CKDIR   (seeds: $SEEDS)"
echo "========================================================="

if [ ! -d "$CKDIR" ]; then
    echo "ERROR: '$CKDIR' does not exist. Contents of '${CKPT_ROOT}':"
    ls -1 "${CKPT_ROOT}" 2>/dev/null || echo "  ('${CKPT_ROOT}' missing)"
    exit 0
fi

echo ""
echo "--- checkpoints that will be evaluated ---"
python evaluate_calibrated.py \
    --root "$CKDIR" \
    --seeds "$SEEDS" \
    --list_only

echo ""
echo "--- calibrated evaluation ---"
python evaluate_calibrated.py \
    --root "$CKDIR" \
    --seeds "$SEEDS" \
    --dataset_type "$DATASET_TYPE" \
    --dataset_path "$DATASET_PATH" \
    --alphas 0.01,0.05,0.10 \
    --calib_from train \
    --calib_samples 200000

echo ""
echo "========================================================="
echo " Finished: $name"
echo "========================================================="
