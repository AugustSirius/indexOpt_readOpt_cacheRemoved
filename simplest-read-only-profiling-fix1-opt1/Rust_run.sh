#!/bin/bash
#slurm options
#SBATCH -p amd-ep2,intel-sc3,amd-ep2-short
#SBATCH -q normal
#SBATCH -J rust
#SBATCH -c 1
#SBATCH -n 64
#SBATCH --mem 300G
########################## MSConvert run #####################
# module
module load gcc
cd /storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/indexOpt_readOpt_cacheRemoved/simplest-read-only-profiling-fix1
cargo run --release