import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 数据准备 (保持不变)
# ---------------------------------------------------------
data = sio.loadmat('C:\\Users\\wangjunwen\\Desktop\\article2\\code\\MCSs\\data\\Mtraining+Mtest\\MCSs_train&test_data.mat')
m_train = data['Mtraining']  # (6500, 4)
m_test_full = data['Mtest']  # 原始测试集 (可能是 1000000, 4)

# 关键约束：仅取前 1600 行用于预测实验
test_limit = 1600
m_test = m_test_full[:test_limit, :]

[trainLen, dim] = m_train.shape
testLen = m_test.shape[0]

scaler = MinMaxScaler(feature_range=(0, 1))
train_n = scaler.fit_transform(m_train)
test_n = scaler.transform(m_test)

x_train_tensor = torch.FloatTensor(train_n[:-1, :]).unsqueeze(0)
y_train_tensor = torch.FloatTensor(train_n[1:, :]).unsqueeze(0)


# ---------------------------------------------------------
# 2. 模型定义 (LSTM & GRU & Transformer)
# ---------------------------------------------------------
class RNNModel(nn.Module):
    def __init__(self, mode='LSTM', input_dim=4, hidden_dim=150, output_dim=4):
        super(RNNModel, self).__init__()
        self.mode = mode
        if mode == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


# 训练函数
def train_model(model, x_train, y_train, epochs, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, _ = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        # ======== 打印当前训练进度（带名称）========
        print(f"[{name}] 正在训练第 {epoch + 1}/{epochs} 代，当前 Loss = {loss.item():.6f}")
    return model

# ---------------------------------------------------------
# 3. 带驱动力的预测函数 (x1 驱动)
# ---------------------------------------------------------
def predict_with_drive(model, train_data, test_data, test_len):
    """
    预测流程说明：
    1. 初始输入：训练集最后一个点 train_data[-1, :]
    2. 隐藏状态：初始化为 None（不进行预热）
    3. 每次预测：
       - 保存模型原始输出（包括 x1）
       - 用测试集真实 x1 替换预测的 x1，作为下一步输入
    4. 输出对齐：predictions[i] 对应 test_data[i, :]
    """
    model.eval()
    with torch.no_grad():
        predictions = []
        h_state = None  # 不进行预热，隐藏状态初始化为 None
        current_input = torch.FloatTensor(train_data[-1, :]).view(1, 1, 4)

        for t in range(test_len):
            # 预测当前时刻
            out, h_state = model(current_input, h_state)
            pred_point = out.squeeze().numpy()
            predictions.append(pred_point)  # 保存模型原始输出（包括预测的 x1）

            # 构造下一步输入（最后一步无需构造）
            if t < test_len - 1:
                next_input_np = pred_point.copy()
                next_input_np[0] = test_data[t, 0]  # 用测试集第 t 个点的真实 x1 替换
                current_input = torch.FloatTensor(next_input_np).view(1, 1, 4)

    return np.array(predictions)

# 执行训练与预测
print("\n=" * 40)
print("\u5f00始训练 LSTM 模型...")
lstm_model = train_model(RNNModel(mode='LSTM'), x_train_tensor, y_train_tensor, epochs=200 ,name ='LSTM')

print("\n开始训练 GRU 模型...")
gru_model = train_model(RNNModel(mode='GRU'), x_train_tensor, y_train_tensor, epochs=200, name ='GRU')

print("\n开始预测...")
preds_lstm_n = predict_with_drive(lstm_model, train_n, test_n, testLen)
preds_gru_n = predict_with_drive(gru_model, train_n, test_n, testLen)

preds_lstm = scaler.inverse_transform(preds_lstm_n)
preds_gru = scaler.inverse_transform(preds_gru_n)

# ---------------------------------------------------------
# 4. 指标计算 (NRMSE & MAE)
# ---------------------------------------------------------
def get_nrmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2)) / np.std(true)


def get_mae(true, pred):
    return np.mean(np.abs(true - pred))


print("\n" + "=" * 100)
print(f"{'维度':<8} | {'LSTM NRMSE':<12} | {'GRU NRMSE':<12} | {'LSTM MAE':<10} | {'GRU MAE':<10}")
print("-" * 100)

for i in range(4):
    n_l = get_nrmse(m_test[:, i], preds_lstm[:, i])
    n_g = get_nrmse(m_test[:, i], preds_gru[:, i])
    m_l = get_mae(m_test[:, i], preds_lstm[:, i])
    m_g = get_mae(m_test[:, i], preds_gru[:, i])
    print(f"x{i + 1:<7} | {n_l:<12.6f} | {n_g:<12.6f} |  {m_l:<10.6f} | {m_g:<10.6f}")

# ---------------------------------------------------------
# 保存预测结果和真实值
# ---------------------------------------------------------
save_path = 'C:\\Users\\wangjunwen\\Desktop\\第二篇审稿\\对比实验'
np.savetxt(f'{save_path}\\preds_lstm.csv', preds_lstm, delimiter=',', 
           header='x1,x2,x3,x4', comments='', fmt='%.8f')
np.savetxt(f'{save_path}\\preds_gru.csv', preds_gru, delimiter=',', 
           header='x1,x2,x3,x4', comments='', fmt='%.8f')
np.savetxt(f'{save_path}\\m_test.csv', m_test, delimiter=',', 
           header='x1,x2,x3,x4', comments='', fmt='%.8f')
print(f"\n数据已保存到: {save_path}")
print(f"  - preds_lstm.csv: {preds_lstm.shape}")
print(f"  - preds_gru.csv: {preds_gru.shape}")
print(f"  - m_test.csv: {m_test.shape}")

# ---------------------------------------------------------
# 5. 四维度绘图展示
# ---------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
titles = ['x1 (Driven Dimension)', 'x2 Component', 'x3 Component', 'x4 Component']

for i in range(4):
    axes[i].plot(m_test[:, i], 'k', label='True Value', alpha=0.7, linewidth=1.5)
    axes[i].plot(preds_lstm[:, i], 'r--', label='LSTM Predict', linewidth=1)
    axes[i].plot(preds_gru[:, i], 'b:', label='GRU Predict', linewidth=1.5)
    axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
    axes[i].legend(loc='upper right', fontsize=9)
    axes[i].grid(True, linestyle='--', alpha=0.5)
    axes[i].set_ylabel('Value', fontsize=10)

plt.xlabel('Time Step', fontsize=11)
plt.tight_layout()
plt.show()