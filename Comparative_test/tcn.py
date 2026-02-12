import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
# ---------------------------------------------------------
# 1. 参数配置
# ---------------------------------------------------------
WINDOW_SIZE = 60  # 建议根据 TCN 感受野设定（此处设为60）
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. 数据准备与窗口化
# ---------------------------------------------------------
# 替换为你自己的路径
data_path = 'C:\\Users\\wangjunwen\\Desktop\\article2\\code\\MCSs\\data\\Mtraining+Mtest\\MCSs_train&test_data.mat'
data = sio.loadmat(data_path)
m_train = data['Mtraining']
m_test_full = data['Mtest']
m_test = m_test_full[:1600, :]

scaler = MinMaxScaler(feature_range=(0, 1))
train_n = scaler.fit_transform(m_train)
test_n = scaler.transform(m_test)


def create_sequences(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size, :])
        y.append(data[i + window_size, :])  # 预测窗口后的下一个点
    return np.array(x), np.array(y)


# 构造训练集
x_train_np, y_train_np = create_sequences(train_n, WINDOW_SIZE)
x_train_tensor = torch.FloatTensor(x_train_np).to(DEVICE)
y_train_tensor = torch.FloatTensor(y_train_np).to(DEVICE)

train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor),
                          batch_size=BATCH_SIZE, shuffle=True)


# ---------------------------------------------------------
# 3. TCN 模型定义 (保持原有架构，优化 Forward)
# ---------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, num_channels=[128, 256, 512], kernel_size=5, dropout=0.1):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, 4) -> (batch, 4, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        # 只需要最后一个时间步的输出进行预测
        out = self.fc(y[:, :, -1])
        return out


# ---------------------------------------------------------
# 4. 训练与预测函数
# ---------------------------------------------------------
def get_nrmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2)) / np.std(true)


def get_mae(true, pred):
    return np.mean(np.abs(true - pred))


def train_model(model, loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(loader):.6f}")
    return model


def predict_with_drive(model, train_data, test_data, test_len, window_size):
    model.eval()
    predictions = []
    # 初始窗口：训练集最后 window_size 个点
    current_window = torch.FloatTensor(train_data[-window_size:, :]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for t in range(test_len):
            # 1. 预测
            pred_point = model(current_window).cpu().numpy().flatten()
            predictions.append(pred_point)

            # 2. 准备下一帧：用测试集真实 x1 修正预测值
            next_point = pred_point.copy()
            next_point[0] = test_data[t, 0]
            next_point_tensor = torch.FloatTensor(next_point).view(1, 1, 4).to(DEVICE)

            # 3. 滑动窗口更新
            current_window = torch.cat((current_window[:, 1:, :], next_point_tensor), dim=1)

    return np.array(predictions)


# ---------------------------------------------------------
# 5. 执行流程
# ---------------------------------------------------------
print(">>> 正在训练 TCN 模型 (滑动窗口模式)...")
model = TCNModel().to(DEVICE)
model = train_model(model, train_loader, EPOCHS)

print(">>> 正在进行驱动力引导预测...")
preds_n = predict_with_drive(model, train_n, test_n, len(test_n), WINDOW_SIZE)
preds_final = scaler.inverse_transform(preds_n)

# ---------------------------------------------------------
# 6. 保存预测结果和真实值
# ---------------------------------------------------------
save_path = 'C:\\Users\\wangjunwen\\Desktop\\第二篇审稿\\对比实验'
np.savetxt(f'{save_path}\\preds_tcn.csv', preds_final, delimiter=',', 
           header='x1,x2,x3,x4', comments='', fmt='%.8f')
np.savetxt(f'{save_path}\\m_test.csv', m_test, delimiter=',', 
           header='x1,x2,x3,x4', comments='', fmt='%.8f')
print(f"\n数据已保存到: {save_path}")
print(f"  - preds_tcn.csv: {preds_final.shape}")
print(f"  - m_test.csv: {m_test.shape}")

# 计算并输出评价指标
print("\n" + "=" * 100)
print(f"{'维度':<8} | {'TCN NRMSE':<12} | {'TCN MAE':<10}")
print("-" * 100)

for i in range(4):
    n_tcn = get_nrmse(m_test[:, i], preds_final[:, i])
    m_tcn = get_mae(m_test[:, i], preds_final[:, i])
    print(f"x{i + 1:<7} | {n_tcn:<12.6f} | {m_tcn:<10.6f}")

# ---------------------------------------------------------
# 7. 绘图展示
# ---------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
labels = ['x1 (Drive)', 'x2', 'x3', 'x4']
for i in range(4):
    axes[i].plot(m_test[:, i], 'k', label='True', alpha=0.6)
    axes[i].plot(preds_final[:, i], 'r--', label='TCN Prediction')
    axes[i].set_ylabel(labels[i])
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.xlabel("Time Steps")
plt.tight_layout()
plt.show()