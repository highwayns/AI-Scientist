import os
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

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
# Data Loading & Preprocessing
# ====================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df['delivery_code'] = df['delivery_method'].map({
        'intravenous': 0,
        'subcutaneous': 1,
        'intramuscular': 2
    })
    return df

# ====================================================
# Custom Dataset
# ====================================================
class ASODataset(Dataset):
    def __init__(self, df):
        self.sequences = df['aso_sequence'].tolist()
        self.methods = torch.tensor(df['delivery_code'].values, dtype=torch.long)
        self.numeric = torch.tensor(
            df[['age','dosage_mg','baseline_factor_level_pct']].values,
            dtype=torch.float32
        )
        self.labels = torch.tensor(df['improvement_pct'].values, dtype=torch.float32)
        self.base2idx = {'A':0,'T':1,'C':2,'G':3}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_idx = torch.tensor([self.base2idx[b] for b in seq], dtype=torch.long)
        return seq_idx, self.methods[idx], self.numeric[idx], self.labels[idx]

# ====================================================
# Model Definition
# ====================================================
class ASOPredictor(nn.Module):
    def __init__(self, seq_len=15, base_embed_dim=8, lstm_hidden=16, method_embed_dim=4, mlp_hidden=64):
        super().__init__()
        self.base2idx = {'A':0,'T':1,'C':2,'G':3}
        self.embedding = nn.Embedding(num_embeddings=4, embedding_dim=base_embed_dim)
        self.lstm = nn.LSTM(input_size=base_embed_dim, hidden_size=lstm_hidden, batch_first=True)
        self.method_emb = nn.Embedding(num_embeddings=3, embedding_dim=method_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + method_embed_dim + 3, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, seq_idx: torch.Tensor, methods: torch.Tensor, numeric_feats: torch.Tensor):
        device = seq_idx.device
        x_embed = self.embedding(seq_idx)      # seq_idx 已经在 device 上
        _, (h_n, _) = self.lstm(x_embed)
        seq_feat    = h_n.squeeze(0)
        method_feat = self.method_emb(methods)  # methods 也在 device 上
        num_feat    = numeric_feats.to(device)
        x           = torch.cat([seq_feat, method_feat, num_feat], dim=1)
        return self.fc(x)

# ====================================================
# Training, Validation, Evaluation, Visualization
# ====================================================
def train(model, train_loader, optimizer, criterion, device='cpu'):
    model.train()
    for epoch in range(1, 11):
        epoch_loss = 0.0
        for seqs, methods, nums, labels in train_loader:
            # —— 关键：整批数据都 .to(device) —— 
            seqs    = seqs.to(device)    
            methods = methods.to(device)
            nums    = nums.to(device)
            labels  = labels.to(device)

            optimizer.zero_grad()
            preds = model(seqs, methods, nums)
            loss  = criterion(preds.squeeze(), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch:02d}: Loss={epoch_loss/len(train_loader):.4f}")

def validate(model, val_loader, criterion, device='cpu'):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seqs, methods, nums, labels in val_loader:
            preds = model(seqs, methods.to(device), nums.to(device))
            loss = criterion(preds.squeeze(), labels.to(device))
            val_loss += loss.item()
    print(f"Validation Loss={val_loss/len(val_loader):.4f}")

def evaluate(model, test_loader, device='cpu'):
    from sklearn.metrics import r2_score, mean_squared_error
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, methods, nums, labels in test_loader:
            preds = model(seqs, methods.to(device), nums.to(device)).squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    metrics = {'R2': r2_score(all_labels, all_preds), 'MSE': mean_squared_error(all_labels, all_preds)}
    with open(os.path.join('results', 'evaluation.txt'), 'w') as f:
        f.write(str(metrics))
    return metrics

def visualize(metrics):
    import matplotlib.pyplot as plt
    names, values = zip(*metrics.items())
    plt.figure()
    plt.bar(names, values)
    plt.title('Evaluation Metrics')
    plt.savefig(os.path.join('results', 'metrics_bar.png'))

# ====================================================
# Main Execution
# ====================================================
if __name__ == "__main__":
    print(f"Starting experiment: {project_name}")
    torch.manual_seed(42)
    np.random.seed(42)

    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    df = load_data(data_path)
    df = preprocess_data(df)

    dataset = ASODataset(df)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16)
    test_loader  = DataLoader(test_ds, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASOPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train(model, train_loader, optimizer, criterion, device)
    validate(model, val_loader, criterion, device)

    metrics = evaluate(model, test_loader, device)
    visualize(metrics)

    print("Experiment completed.")
