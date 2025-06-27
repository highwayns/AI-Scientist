import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

# ====================================================
# Plotting Utilities for ASO Experiment
# ====================================================

def plot_loss_curve(history_csv: str, output_path: str):
    """
    从 CSV 文件中读取训练与验证损失，并绘制学习曲线。
    CSV 文件应包含三列：epoch, train_loss, val_loss
    """
    df = pd.read_csv(history_csv)
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metrics(metrics_txt: str, output_path: str):
    """
    从保存的 metrics 文本文件中读取评估指标，并绘制条形图。
    metrics 文件格式：Python 字典字符串，如 {'R2': 0.85, 'MSE': 2.3}
    """
    with open(metrics_txt, 'r') as f:
        metrics = ast.literal_eval(f.read())
    names = list(metrics.keys())
    values = [metrics[k] for k in names]

    plt.figure()
    plt.bar(names, values)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Evaluation Metrics')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')

    # Plot loss curves (train/val)
    history_csv = os.path.join(results_dir, 'train_val_history.csv')  # TODO: 确保实验脚本写出了此文件
    loss_curve_png = os.path.join(results_dir, 'loss_curve.png')
    if os.path.exists(history_csv):
        plot_loss_curve(history_csv, loss_curve_png)
        print(f'Loss curve saved to: {loss_curve_png}')
    else:
        print(f'History CSV not found: {history_csv}')

    # Plot evaluation metrics
    metrics_txt = os.path.join(results_dir, 'evaluation.txt')
    metrics_png = os.path.join(results_dir, 'metrics_bar.png')
    if os.path.exists(metrics_txt):
        plot_metrics(metrics_txt, metrics_png)
        print(f'Metrics bar chart saved to: {metrics_png}')
    else:
        print(f'Metrics file not found: {metrics_txt}')
