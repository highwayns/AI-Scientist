
# templates/bmad_cross_role_ai/plot.py
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main(run_dir='.'):
    df = pd.read_csv(os.path.join(run_dir, "per_run_metrics.csv"))

    # fig_main：交付核心指标散点
    plt.figure()
    subset = df[["meta/treatment","metric/lead_time_median","metric/deploy_freq_per_week"]]
    for t in subset["meta/treatment"].unique():
        part = subset[subset["meta/treatment"]==t]
        plt.scatter(part["metric/lead_time_median"], part["metric/deploy_freq_per_week"], label=t)
    plt.xlabel("Lead time (median, hours)")
    plt.ylabel("Deploy frequency (per week)")
    plt.legend()
    plt.title("Delivery Tradeoff: lead time vs deploy freq")
    plt.savefig("fig_main.png", dpi=200)

    # fig_ablation：协作 vs 失败率
    plt.figure()
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["metric/cross_discipline_review_ratio"], width, label="cross-discipline")
    plt.bar(x + width/2, df["metric/change_failure_rate"], width, label="change-failure")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Collaboration vs Failure Rate")
    plt.savefig("fig_ablation.png", dpi=200)

    # fig_network：占位（观测模式可替换为评审网络结构图）
    plt.figure()
    plt.plot(np.load(os.path.join(run_dir, "series.npy")))
    plt.title("Series placeholder")
    plt.savefig("fig_network.png", dpi=200)

if __name__ == "__main__":
    main(".")
