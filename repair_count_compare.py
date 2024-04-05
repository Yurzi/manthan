from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

XLSX_PATH = [
    "/home/yurzi/Documents/Research/manthan/log-manthan2-random.xlsx",
    "/home/yurzi/Documents/Research/manthan/log-manthan2-fullsample.xlsx",
    "/home/yurzi/Documents/Research/manthan/log-manthan2-original.xlsx",
]

LINE_LABLES = {
    0: "random",
    1: "fullsample",
    2: "original",
}


def compare_sequences(A, B):
    # 确保A和B的长度相同
    if len(A) != len(B):
        raise ValueError("The sequences must have the same length.")

    # 初始化计数器和差值列表
    count = 0
    differences = []

    # 遍历序列，进行比较
    for a, b in zip(A, B):
        if a < b:
            count += 1
            differences.append(b - a)
            print(a, b)

    # 计算总差值和平均差值
    total_difference = sum(differences)
    avg_difference = total_difference / count if count > 0 else 0

    return count, total_difference, avg_difference


if __name__ == "__main__":
    data: Dict[str, Dict] = defaultdict(dict)
    for idx, xlsx in enumerate(XLSX_PATH):
        table = pd.read_excel(xlsx)
        for index, row in table.iterrows():
            if not bool(row["有解"]):
                continue

            instance = str(row["实例名称"])
            repair_count = int(row["修复次数"])

            data[instance][idx] = repair_count

    repair_count_group: List = list()

    # draw line chart
    plt.figure(figsize=(10, 6))

    n_instance = len(data)
    n_line = 3

    for key in range(n_line):
        y_values: List = [data[instance].get(key, 5001) for instance in data]
        repair_count_group.append(y_values)

    count, difference, _ = compare_sequences(
        repair_count_group[0], repair_count_group[2]
    )
    plt.scatter(
        repair_count_group[0],
        repair_count_group[2],
        label="Random(x) VS Original(Y)",
        marker="^",
        facecolors="none",
        edgecolors="orange",
    )
    print(
        f"compare random with original: better {count}, total_difference {difference}"
    )
    count, difference, _ = compare_sequences(
        repair_count_group[1], repair_count_group[2]
    )
    plt.scatter(
        repair_count_group[1],
        repair_count_group[2],
        label="Fullsample(x) VS Original(Y)",
        marker="x",
    )
    # plt.scatter(
    #    repair_count_group[1],
    #    repair_count_group[0],
    #    label="Fullsample(x) VS Random(Y)",
    #    marker="x",
    # )
    print(
        f"compare fullsample with original: better {count}, total_difference {difference}"
    )

    # 获取当前坐标轴的限制，并基于此绘制对角线y=x
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # 散点图数据的最小值
        np.max([plt.xlim(), plt.ylim()]),  # 散点图数据的最大值
    ]

    # 绘制对角线
    plt.plot(lims, lims, "k-", alpha=0.75, zorder=0, label="y=x")
    plt.fill_between(lims, 0, lims, alpha=0.2, color="gray", zorder=-1)

    plt.xlabel("Repair Count")  # X-axis Label
    plt.ylabel("Repair Count")  # Y-axis Label
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Repari Count")  # Chart title
    plt.legend(title="Experment")  # Legend
    plt.savefig("result-pic/repari-count-compare.png")
    plt.show()
