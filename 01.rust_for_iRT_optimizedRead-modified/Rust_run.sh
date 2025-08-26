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
cd /storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/indexOpt_readOpt_cacheRemoved/01.rust_for_iRT_optimizedRead-modified
cargo run --release -- \
  -d "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/test_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d" \
  -r "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/test_data/report.parquet" \
  -l "/storage/guotiannanLab/wangshuaiyao/777.library/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv" \
  -o "output" \
  -t 64 \
  -n 1000