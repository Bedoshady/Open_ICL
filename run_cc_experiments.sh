#!/bin/bash
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1 
#SBATCH --cpus-per-task=6 # Cores proportional to GPUs
#SBATCH --mem=32000M # Memory proportional to GPUs
#SBATCH --time=0-03:00:00 # DD-HH:MM:SS
#SBATCH --account=def-rsadve
#SBATCH --array=0-4
#SBATCH --output=slurm-%A_%a.out

# Load modules and activate environment
module load StdEnv/2023
module load python/3.12
module load cuda/12.2
module load scipy-stack

# Prevent race conditions by checking if env exists
if [ ! -d "myDRLenv" ]; then
    virtualenv --no-download myDRLenv
    source myDRLenv/bin/activate
    pip install --no-index torch torchvision torchtext torchaudio
    pip install scikit-learn numpy
else
    source myDRLenv/bin/activate
fi

# Define arrays for the sub-datasets
# Define arrays for the sub-datasets (Randomized)
declare -a names=("set1_random" "set2_random" "set3_random" "set4_random" "set5_random")
declare -a known_classes=(
    "AM-SSB,BPSK,GFSK,QAM64,WBFM,8PSK"
    "AM-DSB,CPFSK,PAM4,QAM16,QPSK,AM-SSB"
    "QPSK,8PSK,GFSK,AM-DSB,QAM64,BPSK"
    "WBFM,PAM4,QAM16,AM-SSB,CPFSK,8PSK"
    "GFSK,QAM16,QAM64,BPSK,AM-DSB,PAM4"
)

# SLURM_ARRAY_TASK_ID goes from 0 to 4
i=$SLURM_ARRAY_TASK_ID
name="${names[$i]}"
classes="${known_classes[$i]}"
ckpt_dir="checkpoints/${name}"

echo "========================================================="
echo "Running Experiment: $name"
echo "Known Classes: $classes"
echo "Checkpoint Directory: $ckpt_dir"
echo "========================================================="

# 1. Phase 1 Training
echo "Starting Phase 1 Training..."
python train.py --known_classes "$classes" --checkpoint_dir "$ckpt_dir"

# 2. Phase 2 Incremental Training
echo "Starting Phase 2 Incremental Training..."
python incremental_train.py --checkpoint_dir "$ckpt_dir"

# 3. Evaluation
echo "Starting Evaluation..."
python evaluate.py --checkpoint_dir "$ckpt_dir"

echo "Finished Experiment: $name"
echo "========================================================="
