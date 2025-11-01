import numpy as np
import os

def analyze_rna_data(filename="gatmix_203_s0.txt"):
    """
    读取并分析RNA设计结果文件。

    文件格式应为：
    第一列: 长度 (int)
    第三列: recovery (float)
    第五列: scc (float)
    """
    if not os.path.exists(filename):
        print(f"错误：文件 '{filename}' 不在当前目录中。")
        return

    # 为三个长度范围创建数据桶
    range_small = []  # 0-100 nt
    range_medium = [] # 101-200 nt
    range_large = []  # >200 nt

    with open(filename, 'r') as f:
        for line in f:
            # 尝试解析每一行
            try:
                parts = line.strip().split()
                # 确保行有足够的数据列
                if len(parts) >= 5:
                    length = int(parts[0])
                    recovery = float(parts[2])
                    scc = float(parts[4])
                    
                    # 根据长度将数据放入对应的桶中
                    if length <= 100:
                        range_small.append([recovery, scc])
                    elif 100 < length <= 200:
                        range_medium.append([recovery, scc])
                    else: # length > 200
                        range_large.append([recovery, scc])
            except (ValueError, IndexError):
                # 如果某一行格式不正确（例如，无法转换为数字或列数不够），则跳过
                continue

    # 定义一个函数来计算统计指标并返回一个字典
    def calculate_statistics(data):
        if not data:
            return {
                "mean_recovery": 0.0, "median_recovery": 0.0,
                "mean_scc": 0.0, "median_scc": 0.0
            }

        data_np = np.array(data)
        recoveries = data_np[:, 0]
        sccs = data_np[:, 1]
        
        return {
            "mean_recovery": np.mean(recoveries),
            "median_recovery": np.median(recoveries),
            "mean_scc": np.mean(sccs),
            "median_scc": np.median(sccs)
        }

    # 计算每个范围的统计数据
    stats_small = calculate_statistics(range_small)
    stats_medium = calculate_statistics(range_medium)
    stats_large = calculate_statistics(range_large)

    # 按照要求的格式构建并打印输出字符串
    # 打印表头以供参考
    print(filename)
    print("           & Recovery (0-100) & Recovery (100-200) & Recovery (>200) & SC Score (0-100) & SC Score (100-200) & SC Score (>200)")
    
    # 构建平均值行
    mean_line = (
        f"平均值:    & {stats_small['mean_recovery']:.3f} & {stats_medium['mean_recovery']:.3f} & {stats_large['mean_recovery']:.3f} & "
        f"{stats_small['mean_scc']:.3f} & {stats_medium['mean_scc']:.3f} & {stats_large['mean_scc']:.3f}"
    )

    # 构建中位数行
    median_line = (
        f"中位数:    & {stats_small['median_recovery']:.3f} & {stats_medium['median_recovery']:.3f} & {stats_large['median_recovery']:.3f} & "
        f"{stats_small['median_scc']:.3f} & {stats_medium['median_scc']:.3f} & {stats_large['median_scc']:.3f}"
    )

    print(mean_line)
    print(median_line)


if __name__ == "__main__":
    analyze_rna_data()