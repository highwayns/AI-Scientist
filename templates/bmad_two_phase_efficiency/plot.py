
# templates/two_phase_efficiency/plot.py
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main(run_dir='.'):
    df = pd.read_csv(os.path.join(run_dir, "per_run_metrics.csv"))

    # fig_efficiency：效率-成本
    plt.figure()
    plt.scatter(df["metric/lead_time_median"], df["metric/cost_per_feature_unit"])
    for i, t in enumerate(df["meta/treatment"]):
        plt.annotate(t, (df["metric/lead_time_median"][i], df["metric/cost_per_feature_unit"][i]))
    plt.xlabel("Lead time (median, hours)")
    plt.ylabel("Cost per feature (unit)")
    plt.title("Efficiency vs Cost")
    plt.savefig("fig_efficiency.png", dpi=200)

    # fig_quality：质量指标
    plt.figure()
    x = np.arange(len(df))
    width = 0.25
    plt.bar(x - width, df["metric/rework_rate"], width, label="rework_rate")
    plt.bar(x, df["metric/defect_leakage_ppk"], width, label="defect_leakage")
    plt.bar(x + width, df["metric/test_pass_rate"], width, label="test_pass_rate")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Quality Metrics")
    plt.savefig("fig_quality.png", dpi=200)

    # fig_sensitivity：变更失败率 vs 恢复时间
    plt.figure()
    plt.scatter(df["metric/change_failure_rate"], df["metric/mttr_hours"])
    for i, t in enumerate(df["meta/treatment"]):
        plt.annotate(t, (df["metric/change_failure_rate"][i], df["metric/mttr_hours"][i]))
    plt.xlabel("Change Failure Rate")
    plt.ylabel("MTTR (hours)")
    plt.title("Stability Sensitivity")
    plt.savefig("fig_sensitivity.png", dpi=200)

if __name__ == "__main__":
    import numpy as np
    main(".")
