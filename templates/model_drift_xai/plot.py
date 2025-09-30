
# templates/model_drift_xai/plot.py
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main(run_dir='.'):
    df = pd.read_csv(os.path.join(run_dir, "per_run_metrics.csv"))

    # fig_perf：pre/post性能与校准
    plt.figure()
    x = np.arange(len(df))
    width = 0.2
    plt.bar(x - 1.5*width, df["metric/auroc_pre"], width, label="AUROC_pre")
    plt.bar(x - 0.5*width, df["metric/auroc_post"], width, label="AUROC_post")
    plt.bar(x + 0.5*width, df["metric/calib_ece_pre"], width, label="ECE_pre")
    plt.bar(x + 1.5*width, df["metric/calib_ece_post"], width, label="ECE_post")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Pre/Post Performance & Calibration")
    plt.savefig("fig_perf.png", dpi=200)

    # fig_drift：PSI/KL
    plt.figure()
    x = np.arange(len(df))
    width = 0.3
    plt.bar(x - width/2, df["metric/psi"], width, label="PSI")
    plt.bar(x + width/2, df["metric/kl_div"], width, label="KL")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Distribution Drift (PSI/KL)")
    plt.savefig("fig_drift.png", dpi=200)

    # fig_xai：一致性与告警
    plt.figure()
    plt.scatter(df["metric/xai_consistency"], df["metric/alert_precision"])
    for i, t in enumerate(df["meta/treatment"]):
        plt.annotate(t, (df["metric/xai_consistency"][i], df["metric/alert_precision"][i]))
    plt.xlabel("XAI Consistency@K")
    plt.ylabel("Alert Precision")
    plt.title("XAI Consistency vs Alert Precision")
    plt.savefig("fig_xai.png", dpi=200)

if __name__ == "__main__":
    main(".")
