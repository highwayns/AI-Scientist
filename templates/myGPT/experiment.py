import os
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================
# Project Metadata
# ====================================================
project_name = "基于dify生成反义寡核苷酸（ASO）在血友病患者中的应用与优化研究"
author = "Zheng Jun"
date = datetime.date.today().isoformat()

# ====================================================
# Directory Structure Setup
# ====================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
dirs = [
    os.path.join(base_dir, 'data'),
    os.path.join(base_dir, 'src'),
    os.path.join(base_dir, 'notebooks'),
    os.path.join(base_dir, 'models'),
    os.path.join(base_dir, 'results'),
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

# ====================================================
# Environment Requirements (requirements.txt sample)
# ====================================================
# pandas
# numpy
# torch
# sakana_ai
# scikit-learn
# matplotlib

# ====================================================
# Data Loading & Preprocessing
# ====================================================
def load_data(path: str) -> pd.DataFrame:
    """
    加载实验数据
    :param path: 数据文件路径
    :return: pandas DataFrame
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗与特征工程示例函数
    """
    df = df.dropna()
    # 示例：将 delivery_method 转为类别编码
    df['delivery_code'] = df['delivery_method'].map({
        'intravenous': 0,
        'subcutaneous': 1,
        'intramuscular': 2
    })
    return df

# ====================================================
# Model Definition
# ====================================================
class ASOPredictor(nn.Module):
    """
    基于寡核苷酸序列和临床特征的改善预测模型
    """
    def __init__(self, seq_len=15, base_embed_dim=8, lstm_hidden=16, method_embed_dim=4, mlp_hidden=64):
        super().__init__()
        self.base2idx = {'A':0,'T':1,'C':2,'G':3}
        self.embedding = nn.Embedding(num_embeddings=4, embedding_dim=base_embed_dim)
        self.lstm = nn.LSTM(input_size=base_embed_dim, hidden_size=lstm_hidden, batch_first=True)
        self.method_emb = nn.Embedding(num_embeddings=3, embedding_dim=method_embed_dim)
        # 输入维度：LSTM 输出 + 方法 embedding + 年龄、剂量、基线水平（3）
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + method_embed_dim + 3, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)  # 输出改善百分比预测
        )

    def forward(self, sequences: list, methods: torch.Tensor, numeric_feats: torch.Tensor):
        # 序列编码
        batch_size = len(sequences)
        seq_idx = torch.zeros((batch_size, len(sequences[0])), dtype=torch.long)
        for i, seq in enumerate(sequences):
            seq_idx[i] = torch.tensor([self.base2idx[b] for b in seq], dtype=torch.long)
        x_embed = self.embedding(seq_idx)  # [B, L, E]
        _, (h_n, _) = self.lstm(x_embed)  # h_n: [1, B, H]
        seq_feat = h_n.squeeze(0)        # [B, H]
        # 方法 embedding
        method_feat = self.method_emb(methods)  # [B, M]
        # 拼接所有特征
        x = torch.cat([seq_feat, method_feat, numeric_feats], dim=1)
        out = self.fc(x)
        return out


def build_model() -> ASOPredictor:
    """
    实例化 ASOPredictor 模型
    """
    model = ASOPredictor(seq_len=15,
                         base_embed_dim=8,
                         lstm_hidden=16,
                         method_embed_dim=4,
                         mlp_hidden=64)
    return model

# ====================================================
# Training & Validation
# ====================================================
def train(model, train_loader, optimizer, criterion, device='cpu'):
    model.train()
    for epoch in range(1, 11):
        epoch_loss = 0.0
        for batch in train_loader:
            seqs, methods, nums, labels = batch
            optimizer.zero_grad()
            preds = model(seqs, methods.to(device), nums.to(device))
            loss = criterion(preds.squeeze(), labels.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch:02d}: Loss={epoch_loss/len(train_loader):.4f}")


def validate(model, val_loader, criterion, device='cpu'):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            seqs, methods, nums, labels = batch
            preds = model(seqs, methods.to(device), nums.to(device))
            loss = criterion(preds.squeeze(), labels.to(device))
            val_loss += loss.item()
    print(f"Validation Loss={val_loss/len(val_loader):.4f}")

# ====================================================
# Evaluation & Results
# ====================================================
def evaluate(model, test_loader, device='cpu'):
    from sklearn.metrics import r2_score, mean_squared_error
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, methods, nums, labels in test_loader:
            preds = model(seqs, methods.to(device), nums.to(device)).squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    metrics = {
        'R2': r2_score(all_labels, all_preds),
        'MSE': mean_squared_error(all_labels, all_preds)
    }
    with open(os.path.join('results', 'evaluation.txt'), 'w') as f:
        f.write(str(metrics))
    return metrics

# ====================================================
# Visualization
# ====================================================
def visualize(metrics):
    import matplotlib.pyplot as plt
    names, values = zip(*metrics.items())
    plt.bar(names, values)
    plt.title('Evaluation Metrics')
    plt.savefig(os.path.join('results', 'metrics_bar.png'))

# ====================================================
# Main Execution
# ====================================================
if __name__ == "__main__":
    print(f"Starting experiment: {project_name}")
    # TODO: 数据加载、DataLoader 构建、训练配置代码
    pass
