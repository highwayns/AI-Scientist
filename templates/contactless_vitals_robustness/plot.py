
# templates/contactless_vitals_robustness/plot.py
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main(run_dir='.'):
    df = pd.read_csv(os.path.join(run_dir, "per_run_metrics.csv"))

    # fig_acc：HR/RR RMSE 与 MAE
    plt.figure()
    x = np.arange(len(df))
    width = 0.25
    plt.bar(x - width, df["metric/hr_rmse_bpm"], width, label="HR RMSE")
    plt.bar(x, df["metric/rr_rmse_bpm"], width, label="RR RMSE")
    plt.bar(x + width, df["metric/hr_mae_bpm"], width, label="HR MAE")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Accuracy (bpm)")
    plt.savefig("fig_acc.png", dpi=200)

    # fig_robust：覆盖率/失效率/SNR
    plt.figure()
    x = np.arange(len(df))
    width = 0.3
    plt.bar(x - width/2, df["metric/coverage_rate"], width, label="coverage")
    plt.bar(x + width/2, df["metric/failure_rate"], width, label="failure")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Robustness")
    plt.savefig("fig_robust.png", dpi=200)

    # fig_ba：Bland–Altman界限（HR）
    plt.figure()
    low = df["metric/bland_altman_loA_hr_low"]
    high = df["metric/bland_altman_loA_hr_high"]
    mid = (low + high)/2
    x = np.arange(len(df))
    plt.errorbar(x, mid, yerr=[mid-low, high-mid], fmt='o')
    plt.xticks(x, df["meta/treatment"])
    plt.title("Bland–Altman LoA (HR)")
    plt.savefig("fig_ba.png", dpi=200)

if __name__ == "__main__":
    main(".")
