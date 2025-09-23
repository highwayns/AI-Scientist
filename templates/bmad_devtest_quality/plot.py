
# templates/bmad_devtest_quality/plot.py
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

    # fig_quality：质量相关柱图
    plt.figure()
    x = np.arange(len(df))
    width = 0.18
    plt.bar(x - 2*width, df["metric/defect_leakage_ppk"], width, label="leakage_ppk")
    plt.bar(x - width, df["metric/test_coverage"], width, label="coverage")
    plt.bar(x, df["metric/mutation_score"], width, label="mutation")
    plt.bar(x + width, df["metric/test_pass_rate"], width, label="pass_rate")
    plt.bar(x + 2*width, df["metric/flaky_fail_share"], width, label="flaky_share")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Quality Metrics")
    plt.savefig("fig_quality.png", dpi=200)

    # fig_cost：管线时长与成本
    plt.figure()
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["metric/pipeline_duration_min"], width, label="pipeline_min")
    plt.bar(x + width/2, df["metric/pipeline_cost_unit"], width, label="pipeline_cost")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Pipeline Duration & Cost")
    plt.savefig("fig_cost.png", dpi=200)

if __name__ == "__main__":
    import numpy as np
    main(".")
