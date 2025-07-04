# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
# %%
# 加载全部数据
df = pd.read_csv('data//household_power_consumption.txt', sep=";")
# %%
# 计算分组数
group_size = 10000
num_groups = len(df) // group_size
for group_idx in range(num_groups):
    print(f"Processing group {group_idx+1}/{num_groups}")
    df_group = df.iloc[group_idx*group_size : (group_idx+1)*group_size].copy()
    # 数据预处理
    df_group['datetime'] = pd.to_datetime(df_group['Date'] + " " + df_group['Time'], dayfirst=True)
    df_group.drop(['Date', 'Time'], axis=1, inplace=True)
    for col in df_group.columns:
        if col != 'datetime':
            df_group[col] = pd.to_numeric(df_group[col], errors='coerce')
    df_group.dropna(inplace=True)
    # 分割训练集和测试集
    split_date = df_group['datetime'].iloc[int(len(df_group)*0.7)]
    train = df_group[df_group['datetime'] <= split_date]
    test = df_group[df_group['datetime'] > split_date]
    # 归一化
    feature_cols = [col for col in train.columns if col not in ['datetime', 'Global_active_power']]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    train_X = scaler_X.fit_transform(train[feature_cols])
    train_y = scaler_y.fit_transform(train[['Global_active_power']])
    test_X = scaler_X.transform(test[feature_cols])
    test_y = scaler_y.transform(test[['Global_active_power']])
    # 构造序列样本
    def create_sequences(X, y, seq_len=24):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i-seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)
    seq_len = 24
    train_X_seq, train_y_seq = create_sequences(train_X, train_y, seq_len)
    test_X_seq, test_y_seq = create_sequences(test_X, test_y, seq_len)
    # DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TensorDataset(torch.FloatTensor(train_X_seq), torch.FloatTensor(train_y_seq))
    test_dataset = TensorDataset(torch.FloatTensor(test_X_seq), torch.FloatTensor(test_y_seq))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # LSTM模型
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, output_size=1):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
    input_size = train_X_seq.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=64, output_size=1).to(device)
    # 训练
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5  # 每组数据训练轮数可以适当减少以加快整体速度
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        print(f"Group {group_idx+1}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    # 预测
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            preds.append(output.cpu().numpy())
            trues.append(y_batch.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds_inv = scaler_y.inverse_transform(preds)
    trues_inv = scaler_y.inverse_transform(trues)
    # 横坐标为真实时间
    time_index = test['datetime'].iloc[seq_len:].reset_index(drop=True)
    # 绘图
    plt.figure(figsize=(12,5))
    plt.plot(time_index, trues_inv, label='Ground Truth')
    plt.plot(time_index, preds_inv, label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('Global_active_power')
    plt.legend()
    plt.title(f'LSTM Prediction vs Ground Truth (Group {group_idx+1})')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
# %%