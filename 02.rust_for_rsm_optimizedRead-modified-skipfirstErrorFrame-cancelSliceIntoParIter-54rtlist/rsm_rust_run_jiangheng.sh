#!/bin/bash
#slurm options
#SBATCH -p amd-ep2,intel-sc3,amd-ep2-short
#SBATCH -q huge
#SBATCH -J rust
#SBATCH -c 1
#SBATCH -a 1-2
#SBATCH -n 64
#SBATCH --mem 350G

module load gcc

ROOT_DIR="/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/indexOpt_readOpt_cacheRemoved/02.rust_for_rsm_optimizedRead-modified-skipfirstErrorFrame-cancelSliceIntoParIter-54rtlist/"
OUTPUT_BASE_DIR="/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/indexOpt_readOpt_cacheRemoved/02.rust_for_rsm_optimizedRead-modified-skipfirstErrorFrame-cancelSliceIntoParIter/20250908test"

cd $ROOT_DIR
INPUT_FILE_LIST="/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/02.rust_for_rsm_RTshift1_new/tims_file_failed.txt"
DIANN_FILE_LIST="/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/02.rust_for_rsm_RTshift1_new/diann_result_failed.txt"
DATA_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $INPUT_FILE_LIST)
DIANN_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $DIANN_FILE_LIST)

# Copy to local if needed
[ -d "$DATA_PATH" ] && cp -r "$DATA_PATH" /tmp/ && DATA_PATH="/tmp/$(basename $DATA_PATH)"

BASENAME=$(basename "$DIANN_PATH")

/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/indexOpt_readOpt_cacheRemoved/02.rust_for_rsm_optimizedRead-modified-skipfirstErrorFrame-cancelSliceIntoParIter-54rtlist/target/release/Parsed_RSM \
-d "$DATA_PATH" \
-r "$DIANN_PATH/report.parquet" \
-l /storage/guotiannanLab/wangshuaiyao/777.library/TPHPlib_frag1025_swissprot_final_all_from_Yueliang_with_decoy_sort.tsv \
-o "$OUTPUT_BASE_DIR/$BASENAME" \
-t 64 \
-b 1000

rm -rf /tmp/$(basename $DATA_PATH) 2>/dev/null
