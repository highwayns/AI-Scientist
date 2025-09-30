
# templates/customizable_ui_ux/plot.py
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main(run_dir='.'):
    df = pd.read_csv(os.path.join(run_dir, "per_run_metrics.csv"))

    # fig_main：任务成功率 vs 完成时长
    plt.figure()
    plt.scatter(df["metric/task_success_rate"], df["metric/avg_completion_time_s"])
    for i, t in enumerate(df["meta/treatment"]):
        plt.annotate(t, (df["metric/task_success_rate"][i], df["metric/avg_completion_time_s"][i]))
    plt.xlabel("Task Success Rate")
    plt.ylabel("Avg Completion Time (s)")
    plt.title("Success vs Time")
    plt.savefig("fig_main.png", dpi=200)

    # fig_quality：SUS/TLX/错误率
    plt.figure()
    x = np.arange(len(df))
    width = 0.25
    plt.bar(x - width, df["metric/sus_score"], width, label="SUS")
    plt.bar(x, df["metric/nasa_tlx"], width, label="NASA-TLX")
    plt.bar(x + width, df["metric/error_rate"], width, label="Error Rate")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Usability & Cognitive Load")
    plt.savefig("fig_quality.png", dpi=200)

    # fig_latency：交互延迟与视口利用
    plt.figure()
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["metric/interaction_latency_ms"], width, label="Latency (ms)")
    plt.bar(x + width/2, df["metric/viewport_utilization"], width, label="Viewport Util.")
    plt.xticks(x, df["meta/treatment"])
    plt.legend()
    plt.title("Latency & Viewport Utilization")
    plt.savefig("fig_latency.png", dpi=200)

if __name__ == "__main__":
    main(".")
