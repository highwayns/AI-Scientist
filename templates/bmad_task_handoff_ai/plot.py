
# templates/bmad_task_handoff_ai/plot.py
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

    # fig_ablation：交接损失/缺陷/质量
    plt.figure()
    x = np.arange(len(df))
    width = 0.25
    plt.bar(x - width, df["metric/handoff_loss_rate"], width, label="handoff_loss")
    plt.bar(x, df["metric/defects_per_handoff"], width, label="defects/handoff")
    plt.bar(x + width, df["metric/change_failure_rate"], width, label="change-failure")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Handoff Loss & Defects & CFR")
    plt.savefig("fig_ablation.png", dpi=200)

    # fig_handoff：上下文复用率与Token成本
    plt.figure()
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["metric/context_reuse_ratio"], width, label="context_reuse")
    plt.bar(x + width/2, df["metric/token_cost_per_task"], width, label="token_cost")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Context Reuse vs Token Cost")
    plt.savefig("fig_handoff.png", dpi=200)

if __name__ == "__main__":
    main(".")
