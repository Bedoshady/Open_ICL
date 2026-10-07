#!/bin/bash
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=0-03:00:00
#SBATCH --account=def-rsadve
#SBATCH --array=0-4
#SBATCH --output=gate-%A_%a.out

# ===========================================================================
#  Gate ablation on the stored checkpoints  (read-only, no training)
#
#  Separates the two halves of the decision rule in evaluate.py:
#      Gate A : min contrast prob > tau          (DM novelty score)
#      Gate B : argmax(logits) >= K              (classifier picks a novel class)
#      reported prediction = (A or B)
#
#  and sweeps tau, so we can see:
#      A_only / B_only / OR_current / OR_best
#  plus which candidate novelty score actually separates unknowns:
#      dm_minprob, euclid_min, novel_maxprob, novel_margin, fused_rank,
#      novel_centroid
#
#  The VERDICT block answers: does the combination have headroom, or is the
#  OR rule already at its own ceiling?
#
# ---------------------------------------------------------------------------
#  Usage:
#     sbatch run_gate_ablation_cc.sh
#     sbatch run_gate_ablation_cc.sh rml2016
#     sbatch run_gate_ablation_cc.sh rml2018 /path/to/checkpoint/root
#     sbatch run_gate_ablation_cc.sh rml2018 checkpoints 0     # skip k-select
#
#  args:  1 = dataset type (rml2018 | rml2016)      default rml2018
#         2 = checkpoint root dir                   default checkpoints
#         3 = run the k-selection diagnostic 1/0    default 1
#
#  It expects the layout created by run_experiments.py:
#     <root>/<split>/seed_<42|43|44>/phase2_incremental_model.pth
#  (falls back to phase1_model.pth if there is no phase-2 file).
#
#  If your checkpoints live somewhere else, either pass it as arg 2 or set
#  CKPT_ROOT in the environment, e.g.
#     CKPT_ROOT=$SCRATCH/open_set_modulation/checkpoints sbatch run_gate_ablation_cc.sh
#  and set DATASET_PATH the same way if the dataset is not in the current dir.
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
RUN_KSELECT=${3:-1}

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
echo " Gate ablation / k-selection diagnostic"
echo "   split        : $name"
echo "   dataset type : $DATASET_TYPE"
echo "   dataset path : $DATASET_PATH"
echo "   checkpoints  : $CKDIR"
echo "   host         : $(hostname)"
echo "   python       : $(which python)"
echo "========================================================="

if [ ! -d "$CKDIR" ]; then
    echo "ERROR: '$CKDIR' does not exist."
    echo "Contents of '${CKPT_ROOT}':"
    ls -1 "${CKPT_ROOT}" 2>/dev/null || echo "  ('${CKPT_ROOT}' itself is missing)"
    exit 0
fi

echo ""
echo "--- checkpoints that will be evaluated ---"
python gate_ablation.py --root "$CKDIR" --list_only

echo ""
echo "--- GATE ABLATION ---"
python gate_ablation.py \
    --root "$CKDIR" \
    --dataset_type "$DATASET_TYPE" \
    --dataset_path "$DATASET_PATH" \
    --novel_k auto

# ---------------------------------------------------------------------------
#  K-selection diagnostic. Gives the USB's true-class coverage - how many of
#  the real unknown classes are present at all - which bounds what Gate B can
#  ever catch. Gap statistic and prediction strength are skipped here to keep
#  the wall time down; run it on its own CPU job without --skip_gap --skip_ps
#  if you want the full selector comparison.
# ---------------------------------------------------------------------------
if [ "$RUN_KSELECT" = "1" ]; then
    echo ""
    echo "--- K-SELECTION DIAGNOSTIC (USB composition + selector comparison) ---"
    python analyze_k_selection.py \
        --root "$CKDIR" \
        --dataset_type "$DATASET_TYPE" \
        --dataset_path "$DATASET_PATH" \
        --k_min 2 --k_max 15 \
        --sil_sample 6000 --fit_sample 10000 \
        --skip_gap --skip_ps
fi

echo ""
echo "========================================================="
echo " Finished: $name"
echo "========================================================="
