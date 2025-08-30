#!/bin/bash
#slurm options
#SBATCH -p amd-ep2,intel-sc3,amd-ep2-short
#SBATCH -q normal
#SBATCH -J rust
#SBATCH -c 1
#SBATCH -n 20
#SBATCH --mem 150G

# Load module
module load gcc

# Original data path
ORIGINAL_PATH="/storage/guotiannanLab/wangshuaiyao/002.LiuZW/777.DIABERT_TimsTOF/00.Training100data_guomics_ai/K20200620yuel_TPHP_DIA_001_Slot2-1_1_741.d"

# Check if on Lustre/network storage and copy to local if needed
if df -T "$ORIGINAL_PATH" | grep -q "lustre\|nfs"; then
    echo "Network storage detected, copying to /tmp..."
    LOCAL_PATH="/tmp/$(basename $ORIGINAL_PATH)_$$"
    cp -r "$ORIGINAL_PATH" "$LOCAL_PATH"
    DATA_PATH="$LOCAL_PATH"
else
    DATA_PATH="$ORIGINAL_PATH"
fi

# Run the program
/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/indexOpt_readOpt_cacheRemoved/timstof_diagnostic/target/release/timstof_diagnostic "$DATA_PATH" -v

# Cleanup if we made a local copy
[ -n "$LOCAL_PATH" ] && rm -rf "$LOCAL_PATH"