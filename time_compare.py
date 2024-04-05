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

if __name__ == "__main__":
    data: Dict[str, Dict] = defaultdict(dict)
    for idx, xlsx in enumerate(XLSX_PATH):
        table = pd.read_excel(xlsx)
        for index, row in table.iterrows():
            if not bool(row["有解"]):
                continue

            instance = str(row["实例名称"])
            time_cost = float(row["Preprocess时间"])
            time_cost += float(row["LearnSkf时间"])
            time_cost += float(row["Refine时间"])

            data[instance][idx] = time_cost

    repair_count_group: List = list()

    # draw line chart
    plt.figure(figsize=(10, 6))

    n_instance = len(data)
    n_line = 3

    for key in range(n_line):
        y_values: List = [data[instance].get(key, 10800) for instance in data]
        repair_count_group.append(y_values)

    plt.scatter(
        repair_count_group[0],
        repair_count_group[2],
        label="Random(x) VS Original(Y)",
        marker="o",
    )
    plt.scatter(
        repair_count_group[1],
        repair_count_group[2],
        label="Fullsample(x) VS Original(Y)",
        marker="o",
    )
    plt.scatter(
        repair_count_group[1],
        repair_count_group[0],
        label="Fullsample(x) VS Random(Y)",
        marker="o",
    )

    # 获取当前坐标轴的限制，并基于此绘制对角线y=x
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # 散点图数据的最小值
        np.max([plt.xlim(), plt.ylim()]),  # 散点图数据的最大值
    ]

    # 绘制对角线
    plt.plot(lims, lims, "k-", alpha=0.75, zorder=0, label="y=x")
    plt.fill_between(lims, 0, lims, alpha=0.2, color="gray", zorder=-1)

    plt.xlabel("Time")  # X-axis Label
    plt.ylabel("Time")  # Y-axis Label
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Time Cost")  # Chart title
    plt.legend(title="Experment")  # Legend
    plt.savefig("result-pic/time-cost-compare.png")
    plt.show()
