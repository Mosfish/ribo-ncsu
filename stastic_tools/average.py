import numpy as np
import os
from typing import List

def analyze_mean_stability(filenames: List[str]):
    """
    针对多个独立实验（每个文件代表一次实验），执行以下操作：
    1. 对每个文件，分别计算出三个长度区间的 Recovery 和 SC Score 的平均数。
    2. 收集所有实验的平均数结果。
    3. 对这些平均数，计算最终的平均值和标准差。
    
    这可以衡量模型性能在不同训练运行中的稳定性。

    Args:
        filenames (List[str]): 包含结果数据的文件名列表。
    """
    for filename in filenames:
        if not os.path.exists(filename):
            print(f"错误：文件 '{filename}' 不存在。请检查文件名和路径。")
            return

    # --- MODIFICATION: 创建列表来存储每次运行的平均数 ---
    # 每个列表将包含 N 个值 (N=文件数量)
    small_recovery_means, medium_recovery_means, large_recovery_means = [], [], []
    small_scc_means, medium_scc_means, large_scc_means = [], [], []

    # --- MODIFICATION: 逐个文件处理，而不是混合数据 ---
    for filename in filenames:
        # 为当前文件创建临时数据桶
        range_small, range_medium, range_large = [], [], []
        
        with open(filename, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        length = int(parts[0])
                        recovery = float(parts[2])
                        scc = float(parts[4])
                        
                        if length <= 100:
                            range_small.append([recovery, scc])
                        elif 100 < length <= 200:
                            range_medium.append([recovery, scc])
                        else:
                            range_large.append([recovery, scc])
                except (ValueError, IndexError):
                    continue
        
        # --- MODIFICATION: 对当前文件的每个范围计算平均数，并存储 ---
        if range_small:
            small_recovery_means.append(np.mean(np.array(range_small)[:, 0]))
            small_scc_means.append(np.mean(np.array(range_small)[:, 1]))
        if range_medium:
            medium_recovery_means.append(np.mean(np.array(range_medium)[:, 0]))
            medium_scc_means.append(np.mean(np.array(range_medium)[:, 1]))
        if range_large:
            large_recovery_means.append(np.mean(np.array(range_large)[:, 0]))
            large_scc_means.append(np.mean(np.array(range_large)[:, 1]))

    # --- MODIFICATION: 对收集到的平均数列表计算最终的均值和标准差 ---
    def calculate_stats_of_means(means_list):
        if not means_list:
            return 0.0, 0.0
        return np.mean(means_list), np.std(means_list)

    mean_rec_small, std_rec_small = calculate_stats_of_means(small_recovery_means)
    mean_rec_medium, std_rec_medium = calculate_stats_of_means(medium_recovery_means)
    mean_rec_large, std_rec_large = calculate_stats_of_means(large_recovery_means)
    
    mean_scc_small, std_scc_small = calculate_stats_of_means(small_scc_means)
    mean_scc_medium, std_scc_medium = calculate_stats_of_means(medium_scc_means)
    mean_scc_large, std_scc_large = calculate_stats_of_means(large_scc_means)

    # --- MODIFICATION: 更新输出格式和说明 ---
    print("\n--- 多种子实验【平均数】稳定性分析 ---")
    print(f"数据来源文件: {', '.join(filenames)}")
    print("计算逻辑: 先对每个文件取平均数，再对得到的平均数序列求 均值±标准差。\n")

    result_line_recovery = (
        f"Recovery (Mean): & "
        f"{mean_rec_small:.3f} ± {std_rec_small:.3f} & "
        f"{mean_rec_medium:.3f} ± {std_rec_medium:.3f} & "
        f"{mean_rec_large:.3f} ± {std_rec_large:.3f}"
    )
    result_line_scc = (
        f"SC Score (Mean): & "
        f"{mean_scc_small:.3f} ± {std_scc_small:.3f} & "
        f"{mean_scc_medium:.3f} ± {std_scc_medium:.3f} & "
        f"{mean_scc_large:.3f} ± {std_scc_large:.3f}"
    )

    print("指标              & 0-100 nt                & 100-200 nt              & >200 nt")
    print("-" * 90)
    print(result_line_recovery)
    print(result_line_scc)

def analyze_median_stability(filenames: List[str]):
    """
    针对多个独立实验（每个文件代表一次实验），执行以下操作：
    1. 对每个文件，分别计算出三个长度区间的 Recovery 和 SC Score 的中位数。
    2. 收集所有实验的中位数结果。
    3. 对这些中位数，计算最终的平均值和标准差。
    
    这可以衡量模型“典型性能”在不同训练运行中的稳定性。

    Args:
        filenames (List[str]): 包含结果数据的文件名列表。
    """
    for filename in filenames:
        if not os.path.exists(filename):
            print(f"错误：文件 '{filename}' 不存在。请检查文件名和路径。")
            return

    # --- MODIFICATION: 创建列表来存储每次运行的中位数 ---
    # 每个列表将包含 N 个值 (N=文件数量)
    small_recovery_medians, medium_recovery_medians, large_recovery_medians = [], [], []
    small_scc_medians, medium_scc_medians, large_scc_medians = [], [], []

    # --- MODIFICATION: 逐个文件处理，而不是混合数据 ---
    for filename in filenames:
        # 为当前文件创建临时数据桶
        range_small, range_medium, range_large = [], [], []
        
        with open(filename, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        length = int(parts[0])
                        recovery = float(parts[2])
                        scc = float(parts[4])
                        
                        if length <= 100:
                            range_small.append([recovery, scc])
                        elif 100 < length <= 200:
                            range_medium.append([recovery, scc])
                        else:
                            range_large.append([recovery, scc])
                except (ValueError, IndexError):
                    continue
        
        # --- MODIFICATION: 对当前文件的每个范围计算中位数，并存储 ---
        if range_small:
            small_recovery_medians.append(np.median(np.array(range_small)[:, 0]))
            small_scc_medians.append(np.median(np.array(range_small)[:, 1]))
        if range_medium:
            medium_recovery_medians.append(np.median(np.array(range_medium)[:, 0]))
            medium_scc_medians.append(np.median(np.array(range_medium)[:, 1]))
        if range_large:
            large_recovery_medians.append(np.median(np.array(range_large)[:, 0]))
            large_scc_medians.append(np.median(np.array(range_large)[:, 1]))

    # --- MODIFICATION: 对收集到的中位数列表计算最终的均值和标准差 ---
    def calculate_stats_of_medians(medians_list):
        if not medians_list:
            return 0.0, 0.0
        return np.mean(medians_list), np.std(medians_list)

    mean_rec_small, std_rec_small = calculate_stats_of_medians(small_recovery_medians)
    mean_rec_medium, std_rec_medium = calculate_stats_of_medians(medium_recovery_medians)
    mean_rec_large, std_rec_large = calculate_stats_of_medians(large_recovery_medians)
    
    mean_scc_small, std_scc_small = calculate_stats_of_medians(small_scc_medians)
    mean_scc_medium, std_scc_medium = calculate_stats_of_medians(medium_scc_medians)
    mean_scc_large, std_scc_large = calculate_stats_of_medians(large_scc_medians)

    # --- MODIFICATION: 更新输出格式和说明 ---
    print("\n--- 多种子实验【中位数】稳定性分析 ---")
    print(f"数据来源文件: {', '.join(filenames)}")
    print("计算逻辑: 先对每个文件取中位数，再对得到的中位数序列求 均值±标准差。\n")

    result_line_recovery = (
        f"Recovery (Median): & "
        f"{mean_rec_small:.3f} ± {std_rec_small:.3f} & "
        f"{mean_rec_medium:.3f} ± {std_rec_medium:.3f} & "
        f"{mean_rec_large:.3f} ± {std_rec_large:.3f}"
    )
    result_line_scc = (
        f"SC Score (Median): & "
        f"{mean_scc_small:.3f} ± {std_scc_small:.3f} & "
        f"{mean_scc_medium:.3f} ± {std_scc_medium:.3f} & "
        f"{mean_scc_large:.3f} ± {std_scc_large:.3f}"
    )

    print("指标              & 0-100 nt                & 100-200 nt              & >200 nt")
    print("-" * 90)
    print(result_line_recovery)
    print(result_line_scc)


if __name__ == "__main__":
    files_to_analyze = [
        "gr_1_seed0.txt", 
        "gr_1_seed1.txt", 
        "gr_1_seed2.txt"
    ]
    
    analyze_mean_stability(files_to_analyze)
    analyze_median_stability(files_to_analyze)