
# templates/bmad_agent_agile/plot.py
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main(run_dir='.'):
    df = pd.read_csv(os.path.join(run_dir, "per_run_metrics.csv"))
    # fig_main：核心交付指标对比
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

    # fig_ablation：协作与缺陷指标
    plt.figure()
    ax = plt.gca()
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df["metric/cross_role_review_ratio"], width, label="cross-role")
    ax.bar(x + width/2, df["metric/change_failure_rate"], width, label="change-failure")
    ax.set_xticks(x, df["meta/treatment"])
    ax.legend(); ax.set_title("Collaboration vs Failure Rate")
    plt.savefig("fig_ablation.png", dpi=200)

    # fig_network：占位（若有观测图，可替换为评审网络图/度分布）
    plt.figure()
    plt.plot(np.load(os.path.join(run_dir, "series.npy")))
    plt.title("Series placeholder")
    plt.savefig("fig_network.png", dpi=200)

if __name__ == "__main__":
    main(".")
