
# templates/blockchain_audit_immutable/plot.py
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main(run_dir='.'):
    df = pd.read_csv(os.path.join(run_dir, "per_run_metrics.csv"))

    # fig_security：检测与完整性
    plt.figure()
    x = np.arange(len(df))
    width = 0.25
    plt.bar(x - width, df["metric/detection_rate"], width, label="detection")
    plt.bar(x, df["metric/false_positive_rate"], width, label="false_positive")
    plt.bar(x + width, df["metric/audit_integrity_score"], width, label="integrity")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Security & Integrity Metrics")
    plt.savefig("fig_security.png", dpi=200)

    # fig_cost_latency：成本与延迟
    plt.figure()
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["metric/anchoring_latency_sec"], width, label="latency_sec")
    plt.bar(x + width/2, df["metric/anchoring_cost_unit"], width, label="cost_unit")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Anchoring Latency & Cost")
    plt.savefig("fig_cost_latency.png", dpi=200)

    # fig_tradeoff：检测率 vs 成本
    plt.figure()
    plt.scatter(df["metric/anchoring_cost_unit"], df["metric/detection_rate"])
    for i, t in enumerate(df["meta/treatment"]):
        plt.annotate(t, (df["metric/anchoring_cost_unit"][i], df["metric/detection_rate"][i]))
    plt.xlabel("Anchoring Cost (unit/day)")
    plt.ylabel("Detection Rate")
    plt.title("Cost–Detection Tradeoff")
    plt.savefig("fig_tradeoff.png", dpi=200)

if __name__ == "__main__":
    main(".")
