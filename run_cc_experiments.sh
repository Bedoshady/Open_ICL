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

# Read dataset type from arguments (default to rml2016)
DATASET_TYPE=${1:-rml2016}

if [ "$DATASET_TYPE" = "rml2018" ]; then
    declare -a names=("set1_2018" "set2_2018" "set3_2018" "set4_2018" "set5_2018")
    declare -a known_classes=(
        "OOK,BPSK,PSK16,APSK32,QAM32,QAM256,AM_SSB_WC,AM_DSB_WC,FM,ASK8,PSK32,QAM64"
        "ASK4,QPSK,PSK8,APSK16,APSK64,QAM16,QAM128,AM_SSB_SC,AM_DSB_SC,GMSK,OQPS,ASK8"
        "BPSK,PSK16,APSK32,APSK128,QAM32,QAM128,AM_SSB_WC,AM_DSB_SC,FM,OQPS,OOK,QAM64"
        "ASK4,QPSK,PSK32,APSK16,APSK64,QAM16,QAM256,AM_SSB_SC,AM_DSB_WC,GMSK,ASK8,PSK8"
        "BPSK,PSK8,PSK16,APSK32,APSK128,QAM16,QAM64,AM_SSB_WC,AM_DSB_SC,FM,OQPS,QAM32"
    )
    DATASET_PATH="GOLD_XYZ_OSC.0001_1024.hdf5"
else
    declare -a names=("set1_random" "set2_random" "set3_random" "set4_random" "set5_random")
    declare -a known_classes=(
        "AM-SSB,BPSK,GFSK,QAM64,WBFM,8PSK"
        "AM-DSB,CPFSK,PAM4,QAM16,QPSK,AM-SSB"
        "QPSK,8PSK,GFSK,AM-DSB,QAM64,BPSK"
        "WBFM,PAM4,QAM16,AM-SSB,CPFSK,8PSK"
        "GFSK,QAM16,QAM64,BPSK,AM-DSB,PAM4"
    )
    DATASET_PATH="RML2016.10a_dict.pkl"
fi

# SLURM_ARRAY_TASK_ID goes from 0 to 4
i=$SLURM_ARRAY_TASK_ID
name="${names[$i]}"
classes="${known_classes[$i]}"
ckpt_dir="checkpoints/${name}"

echo "========================================================="
echo "Running Experiment: $name"
echo "Dataset Type: $DATASET_TYPE"
echo "Dataset Path: $DATASET_PATH"
echo "Known Classes: $classes"
echo "Checkpoint Directory: $ckpt_dir"
echo "========================================================="

# 1. Phase 1 Training
echo "Starting Phase 1 Training..."
python train.py --known_classes "$classes" --checkpoint_dir "$ckpt_dir" --dataset_type "$DATASET_TYPE" --dataset_path "$DATASET_PATH"

# 2. Phase 2 Incremental Training
echo "Starting Phase 2 Incremental Training..."
python incremental_train.py --checkpoint_dir "$ckpt_dir" --dataset_path "$DATASET_PATH"

# 3. Evaluation
echo "Starting Evaluation..."
python evaluate.py --checkpoint_dir "$ckpt_dir" --dataset_path "$DATASET_PATH"

echo "Finished Experiment: $name"
echo "========================================================="
