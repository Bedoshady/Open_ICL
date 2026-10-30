#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=0-04:00:00
#SBATCH --account=def-rsadve
#SBATCH --array=0-4
#SBATCH --output=clust-%A_%a.out

# ===========================================================================
#  Clustering purity analysis for all three algorithms, one job.
#
#  CPU only, no GPU needed, nothing is trained. For each split and seed it loads
#  the phase-1 checkpoint, takes the stored Unknown Signal Bank (USB) embeddings,
#  and applies the three implementations used in the paper:
#
#     kmeans   (openicl_kmeans)      Silhouette-guided k in [2,15]
#     dbscan   (openicl_clustering)  eps=0.5, min_samples=100
#     hdbscan  (openicl_hdbscan)     min_cluster_size grid, silhouette*(1-noise)
#
#  and reports for each: d_hat, % USB retained, purity against the true unseen
#  classes, ACC (Hungarian) and NMI. Because all three see the same USB, the
#  comparison is controlled.
#
# ---------------------------------------------------------------------------
#  Usage:
#     sbatch run_clustering_purity_cc.sh                    # rml2016
#     sbatch run_clustering_purity_cc.sh rml2018
#     sbatch run_clustering_purity_cc.sh rml2016 checkpoints 42,43,44
#
#  args:  1 = dataset type (rml2016 | rml2018)     default rml2016
#         2 = checkpoint root                      default checkpoints
#         3 = seeds                                default 42,43,44
#
#  Other locations:
#     CKPT_ROOT=$SCRATCH/open_set_modulation/checkpoints \
#     DATASET_PATH=$SCRATCH/RML2016.10a_dict.pkl \
#     sbatch run_clustering_purity_cc.sh
#
#  If a job runs long, add --sil_sample 10000 on the python line to subsample the
#  Silhouette computation (exact is the default and matches the branch code).
# ===========================================================================

module load StdEnv/2023
module load python/3.12
module load scipy-stack

if [ ! -d "myDRLenv" ]; then
    virtualenv --no-download myDRLenv
    source myDRLenv/bin/activate
    pip install --no-index torch scikit-learn numpy
    pip install scikit-learn numpy
else
    source myDRLenv/bin/activate
fi

DATASET_TYPE=${1:-rml2016}
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
echo " Clustering purity analysis (no training)"
echo "   split        : $name"
echo "   dataset type : $DATASET_TYPE"
echo "   dataset path : $DATASET_PATH"
echo "   checkpoints  : $CKDIR   (seeds: $SEEDS)"
echo "   host         : $(hostname)"
echo "========================================================="

if [ ! -d "$CKDIR" ]; then
    echo "ERROR: '$CKDIR' does not exist. Contents of '${CKPT_ROOT}':"
    ls -1 "${CKPT_ROOT}" 2>/dev/null || echo "  ('${CKPT_ROOT}' missing)"
    exit 0
fi

echo ""
echo "--- checkpoints that will be evaluated ---"
python analyze_clustering_all.py --split_dir "$CKDIR" --list_only

echo ""
echo "--- analysis ---"
python analyze_clustering_all.py \
    --split_dir "$CKDIR" \
    --dataset_type "$DATASET_TYPE" \
    --dataset_path "$DATASET_PATH" \
    --seeds "$SEEDS" \
    --methods kmeans,dbscan,hdbscan

echo ""
echo "========================================================="
echo " Finished: $name"
echo "========================================================="
