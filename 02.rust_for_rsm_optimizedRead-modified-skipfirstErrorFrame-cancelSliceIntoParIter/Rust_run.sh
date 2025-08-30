#!/bin/bash
#slurm options
#SBATCH -p amd-ep2,intel-sc3,amd-ep2-short
#SBATCH -q normal
#SBATCH -J rust
#SBATCH -c 1
#SBATCH -n 64
#SBATCH --mem 300G

module load gcc

# Original data
D="/storage/guotiannanLab/wangshuaiyao/002.LiuZW/777.DIABERT_TimsTOF/00.Training100data_guomics_ai/K20200620yuel_TPHP_DIA_001_Slot2-1_1_741.d"

# Copy to local if needed
[ -d "$D" ] && cp -r "$D" /tmp/ && D="/tmp/$(basename $D)"

cd /storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/indexOpt_readOpt_cacheRemoved/02.rust_for_rsm_optimizedRead-modified-skipfirstErrorFrame-cancelSliceIntoParIter

cargo run --release -- -d "$D" \
  -r "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/test_data/report.parquet" \
  -l "/storage/guotiannanLab/wangshuaiyao/777.library/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv" \
  -t 64 -n 1000

rm -rf /tmp/$(basename $D) 2>/dev/null