import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

XLSX_PATH: str = "/home/yurzi/Documents/Research/manthan/log-manthan2-fullsample.xlsx"

# read data from excel without header
data = pd.read_excel(XLSX_PATH)

dt_score: list = list()
repair_count: list = list()

for index, row in data.iterrows():
    if row["决策树得分"] == -1 or row["备注"] != "无错误" or row["有解"] == "False":
        continue
    dt_score.append(row["决策树得分"])
    repair_count.append(row["修复次数"])

# calculate the pearson correlation
pearson_corr, pearson_p = stats.pearsonr(dt_score, repair_count)
print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p}")

# calculate the spearman correlation
spearman_corr, spearman_p = stats.spearmanr(dt_score, repair_count)
print(f"Spearman correlation: {spearman_corr}, p-value: {spearman_p}")

# draw the scatter plot

plt.scatter(dt_score, repair_count)
plt.yscale("log")
plt.xlabel("Decision Tree Score")
plt.ylabel("Repair Count")
plt.title("Decision Tree Score vs Repair Count")
plt.savefig("result-pic/fullsample-repair-count.png")
plt.show()
