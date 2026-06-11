#!/bin/bash
#SBATCH --gpus-per-node=a100:1 
#SBATCH --cpus-per-task=6 # Cores proportional to GPUs
#SBATCH --mem=32000M # Memory proportional to GPUs
#SBATCH --time=0-03:00:00 # DD-HH:MM:SS
#SBATCH --account=def-rsadve

# Load modules and activate environment
module load StdEnv/2023
module load python/3.12
module load cuda/12.2
module load scipy-stack

virtualenv --no-download myDRLenv
source myDRLenv/bin/activate
pip install --no-index torch torchvision torchtext torchaudio
pip install scikit-learn numpy

# Define arrays for the sub-datasets
declare -a names=("4_known_core" "6_known_analog" "6_known_fsk" "6_known_qam_pam" "8_known_mixed")
declare -a known_classes=(
    "BPSK,QPSK,QAM16,8PSK"
    "BPSK,QPSK,QAM16,8PSK,AM-DSB,AM-SSB"
    "BPSK,QPSK,QAM16,8PSK,CPFSK,GFSK"
    "BPSK,QPSK,QAM16,8PSK,QAM64,PAM4"
    "BPSK,QPSK,QAM16,8PSK,QAM64,PAM4,WBFM,GFSK"
)

# Iterate over each sub-dataset configuration
for i in "${!names[@]}"; do
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
done

echo "All experiments completed!"
