import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def compare_single_file(fname, folder1, folder2):
    """比较单个文件的函数，用于多进程"""
    path1 = os.path.join(folder1, fname)
    path2 = os.path.join(folder2, fname)
    
    try:
        # 使用更高效的读取参数
        df1 = pd.read_csv(path1, engine='c', low_memory=False)
        df2 = pd.read_csv(path2, engine='c', low_memory=False)
        
        # 快速检查：如果形状不同，直接判定不同
        if df1.shape != df2.shape:
            return {
                'fname': fname,
                'equal': False,
                'only1': df1.shape[0],
                'only2': df2.shape[0],
                'common': 0,
                'total1': df1.shape[0],
                'total2': df2.shape[0],
                'error': None
            }
        
        # 对于大文件，使用哈希比较而不是集合
        if len(df1) > 10000:  # 大文件阈值
            # 使用哈希值进行快速比较
            df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
            df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
            
            # 使用pandas的equals方法进行快速比较
            if df1_sorted.equals(df2_sorted):
                return {
                    'fname': fname,
                    'equal': True,
                    'only1': 0,
                    'only2': 0,
                    'common': len(df1),
                    'total1': len(df1),
                    'total2': len(df2),
                    'error': None
                }
            
            # 如果不相等，计算差异（对大文件使用采样）
            # 创建行的哈希值
            df1_hashes = pd.util.hash_pandas_object(df1, index=False)
            df2_hashes = pd.util.hash_pandas_object(df2, index=False)
            
            set1 = set(df1_hashes)
            set2 = set(df2_hashes)
            
            only1 = len(set1 - set2)
            only2 = len(set2 - set1)
            common = len(set1 & set2)
        else:
            # 小文件使用原方法
            set1 = set([tuple(row) for row in df1.to_numpy()])
            set2 = set([tuple(row) for row in df2.to_numpy()])
            
            only1 = len(set1 - set2)
            only2 = len(set2 - set1)
            common = len(set1 & set2)
        
        return {
            'fname': fname,
            'equal': (only1 == 0 and only2 == 0),
            'only1': only1,
            'only2': only2,
            'common': common,
            'total1': len(df1),
            'total2': len(df2),
            'error': None
        }
        
    except Exception as e:
        return {
            'fname': fname,
            'equal': False,
            'only1': 0,
            'only2': 0,
            'common': 0,
            'total1': 0,
            'total2': 0,
            'error': str(e)
        }

def batch_compare_files(file_batch, folder1, folder2):
    """批量比较文件，减少进程创建开销"""
    results = []
    for fname in file_batch:
        results.append(compare_single_file(fname, folder1, folder2))
    return results

def main():
    # 文件夹路径
    # folder1 = "/wangshuaiyao/jiangheng/indexOpt_readOpt_cacheRemoved/02.rust_for_rsm_optimizedRead-modified-extractPeakGroup/output"
    # folder2 = "/wangshuaiyao/jiangheng/indexOpt_readOpt_cacheRemoved/shuaiyao-yuanban-extractPeakGroup/output"

    folder1 = "/Users/augustsirius/Desktop/indexOpt_readOpt_cacheRemoved/02.rust_for_rsm_optimizedRead-modified-extractPeakGroup/output"
    folder2 = "/Users/augustsirius/Desktop/dia-bert-release-version/ORIGINAL-UNTOUCHED-VERSION/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/02.rust_for_rsm-加入finaldataframe输出csv/output"

    
    # 输出文件路径
    output_file = "comparison_result.txt"
    
    # 获取两个文件夹下的文件名（只取 csv）
    print("正在扫描文件夹...")
    files1 = {f for f in os.listdir(folder1) if f.endswith(".csv")}
    files2 = {f for f in os.listdir(folder2) if f.endswith(".csv")}
    
    # 找出同名的 csv 文件
    common_files = sorted(files1.intersection(files2))
    print(f"找到 {len(common_files)} 个同名文件")
    
    # 决定使用的进程数（留一个核心给系统）
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"使用 {num_workers} 个进程进行并行处理")
    
    # 将文件列表分批，每批的大小根据总文件数和进程数动态调整
    batch_size = max(1, len(common_files) // (num_workers * 4))
    file_batches = [common_files[i:i+batch_size] 
                    for i in range(0, len(common_files), batch_size)]
    
    # 使用进程池并行处理
    results = []
    all_equal = True
    compared_files = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有批次任务
        future_to_batch = {
            executor.submit(batch_compare_files, batch, folder1, folder2): batch 
            for batch in file_batches
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(common_files), desc="比较文件", unit="file") as pbar:
            for future in as_completed(future_to_batch):
                batch_results = future.result()
                for result in batch_results:
                    compared_files += 1
                    pbar.update(1)
                    
                    if result['error']:
                        results.append(f"[错误] {result['fname']} 无法比较: {result['error']}")
                        all_equal = False
                    elif result['equal']:
                        results.append(result['fname'])
                    else:
                        results.append(f"\n[比较] {result['fname']}")
                        results.append(f"  folder1 独有行数: {result['only1']}")
                        results.append(f"  folder2 独有行数: {result['only2']}")
                        results.append(f"  共同的行数: {result['common']}")
                        results.append(f"  folder1 总行数: {result['total1']}, folder2 总行数: {result['total2']}")
                        all_equal = False
    
    # 添加总结信息
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
    print(f"📊 共比较了 {compared_files} 个文件")

if __name__ == "__main__":
    main()