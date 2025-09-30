
# templates/edge_realtime_pipeline/plot.py
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main(run_dir='.'):
    df = pd.read_csv(os.path.join(run_dir, "per_run_metrics.csv"))

    # fig_latency：p50/p95/p99
    plt.figure()
    x = np.arange(len(df))
    width = 0.2
    plt.bar(x - width, df["metric/latency_p50_ms"], width, label="p50")
    plt.bar(x, df["metric/latency_p95_ms"], width, label="p95")
    plt.bar(x + width, df["metric/latency_p99_ms"], width, label="p99")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("End-to-End Latency (ms)")
    plt.savefig("fig_latency.png", dpi=200)

    # fig_throughput：吞吐与掉帧
    plt.figure()
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["metric/throughput_fps"], width, label="throughput_fps")
    plt.bar(x + width/2, df["metric/drop_rate"], width, label="drop_rate")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Throughput & Drop Rate")
    plt.savefig("fig_throughput.png", dpi=200)

    # fig_resource：CPU与能耗
    plt.figure()
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["metric/cpu_util"], width, label="cpu_util")
    plt.bar(x + width/2, df["metric/energy_per_inf_mj"], width, label="energy_mJ")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Resource & Energy")
    plt.savefig("fig_resource.png", dpi=200)

if __name__ == "__main__":
    main(".")
