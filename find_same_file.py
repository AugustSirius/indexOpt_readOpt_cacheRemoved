import os
import pandas as pd
from tqdm import tqdm  # 进度条

# 两个文件夹路径
# folder1 = "/Users/augustsirius/Desktop/indexOpt_readOpt_cacheRemoved/02.rust_for_rsm_optimizedRead-modified-extractPeakGroup/output"
# folder2 = "/Users/augustsirius/Desktop/dia-bert-release-version/ORIGINAL-UNTOUCHED-VERSION/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/02.rust_for_rsm-加入finaldataframe输出csv/output"

folder1 = "/wangshuaiyao/jiangheng/indexOpt_readOpt_cacheRemoved/02.rust_for_rsm_optimizedRead-modified-extractPeakGroup/output"
folder2 = "/wangshuaiyao/jiangheng/indexOpt_readOpt_cacheRemoved/shuaiyao-yuanban-extractPeakGroup/output"

# 输出文件路径
output_file = "comparison_result.txt"

# 获取两个文件夹下的文件名（只取 csv）
files1 = {f for f in os.listdir(folder1) if f.endswith(".csv")}
files2 = {f for f in os.listdir(folder2) if f.endswith(".csv")}

# 找出同名的 csv 文件
common_files = files1.intersection(files2)

# 保存结果内容
results = []
all_equal = True
compared_files = 0

# tqdm 进度条
for fname in tqdm(sorted(common_files), desc="Comparing files", unit="file"):
    path1 = os.path.join(folder1, fname)
    path2 = os.path.join(folder2, fname)
    
    try:
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        # 去掉索引影响
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

        # 转换为集合（每一行作为 tuple）
        set1 = set([tuple(row) for row in df1.to_numpy()])
        set2 = set([tuple(row) for row in df2.to_numpy()])

        only1 = set1 - set2
        only2 = set2 - set1
        common = set1 & set2

        if len(only1) == 0 and len(only2) == 0:
            results.append(fname)
        else:
            results.append(f"\n[比较] {fname}")
            results.append(f"  folder1 独有行数: {len(only1)}")
            results.append(f"  folder2 独有行数: {len(only2)}")
            results.append(f"  共同的行数: {len(common)}")
            results.append(f"  folder1 总行数: {len(df1)}, folder2 总行数: {len(df2)}")
            all_equal = False
        compared_files += 1

    except Exception as e:
        results.append(f"[错误] {fname} 无法比较: {e}")
        all_equal = False


# 若两个文件夹的 csv 文件名完全一致，且所有同名文件内容一致，则额外提示
if (
    len(common_files) > 0
    and len(files1) == len(files2) == len(common_files)
    and compared_files == len(common_files)
    and all_equal
):
    results.append("\n两个文件夹的最终结果完全一致")

# 写入文件
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\n✅ 结果已保存到 {output_file}")
