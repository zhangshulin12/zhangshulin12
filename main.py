import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, \
                            mean_absolute_percentage_error, \
                            mean_squared_error, root_mean_squared_error, \
                            r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchinfo import summary
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import pywt
from scipy.signal import medfilt
# import os
#
# # 设置为CPU训练
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# 定义 prediction函数
def prediction(model, iterator):
    all_targets = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            inputs, targets = batch
            predictions = model(inputs)

            all_targets.extend(targets.numpy())
            all_predictions.extend(predictions.numpy())
    return all_targets, all_predictions


# 小波变换去噪函数
def wavelet_denoising(data, wavelet='db1', level=1):
    coeff = pywt.wavedec(data, wavelet, mode='per')
    sigma = (1/0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    uthresh = sigma * np.sqrt(2*np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


class DataHandler(Dataset):
    def __init__(self, X_train_tensor, y_train_tensor, X_valid_tensor, y_valid_tensor):
        self.X_train_tensor = X_train_tensor
        self.y_train_tensor = y_train_tensor
        self.X_valid_tensor = X_valid_tensor
        self.y_valid_tensor = y_valid_tensor

    def __len__(self):
        return len(self.X_train_tensor)

    def __getitem__(self, idx):
        sample = self.X_train_tensor[idx]
        labels = self.y_train_tensor[idx]
        return sample, labels

    def train_loader(self):
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        return DataLoader(train_dataset, batch_size=32, shuffle=True)

    def valid_loader(self):
        valid_dataset = TensorDataset(self.X_valid_tensor, self.y_valid_tensor)
        return DataLoader(valid_dataset, batch_size=32, shuffle=False)


class LSTM_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, transformer_heads, transformer_layers, output_dim, dropout=0.5):
        super(LSTM_Transformer, self).__init__()
        # LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        # Transformer 编码器层
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=transformer_heads, dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=transformer_layers)
        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM 输出
        lstm_out, _ = self.lstm(x)
        # Transformer 输入
        transformer_input = lstm_out
        # Transformer 输出
        transformer_out = self.transformer_encoder(transformer_input)
        # 预测输出
        output = self.fc(transformer_out[:, -1, :])
        # output = self.fc(lstm_out[:, -1, :])
        return output


def train(model, iterator, optimizer):
    device = torch.device("cpu")  # 使用 CPU 进行训练
    model.to(device)  # 将模型移动到 CPU
    epoch_loss_mse = 0
    epoch_loss_mae = 0

    model.train()  # 确保模型处于训练模式
    for batch in iterator:
        optimizer.zero_grad()  # 清空梯度
        inputs, targets = batch  # 获取输入和目标值
        inputs, targets = inputs.to(device), targets.to(device)  # 将输入和目标移动到 CPU

        outputs = model(inputs)  # 前向传播

        loss_mse = criterion_mse(outputs, targets)  # 计算损失
        loss_mae = criterion_mae(outputs, targets)

        combined_loss = loss_mse + loss_mae  # 可以根据需要调整两者的权重

        combined_loss.backward()
        optimizer.step()

        epoch_loss_mse += loss_mse.item()  # 累计损失
        epoch_loss_mae += loss_mae.item()

    average_loss_mse = epoch_loss_mse / len(iterator)  # 计算平均损失
    average_loss_mae = epoch_loss_mae / len(iterator)

    return average_loss_mse, average_loss_mae


def evaluate(model, iterator):
    epoch_loss_mse = 0
    epoch_loss_mae = 0

    model.eval()  # 将模型设置为评估模式，例如关闭 Dropout 等
    with torch.no_grad():  # 不需要计算梯度
        for batch in iterator:
            inputs, targets = batch
            outputs = model(inputs)  # 前向传播

            loss_mse = criterion_mse(outputs, targets)  # 计算损失
            loss_mae = criterion_mae(outputs, targets)

            epoch_loss_mse += loss_mse.item()  # 累计损失
            epoch_loss_mae += loss_mae.item()

    return epoch_loss_mse / len(iterator), epoch_loss_mae / len(iterator)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    data = pd.read_csv('D:/股票预测/AAPL.csv')
    data.tail(5).style.background_gradient()

    print(type(data['Close'].iloc[0]),type(data['Date'].iloc[0]))
    # Let's convert the data type of timestamp column to datatime format
    data['Date'] = pd.to_datetime(data['Date'])
    print(type(data['Close'].iloc[0]),type(data['Date'].iloc[0]))

    # Selecting subset
    # cond_1 = data['Date'] >= '2021-04-23 00:00:00'
    # cond_2 = data['Date'] <= '2024-04-23 00:00:00'
    cond_1 = data['Date'] >= '2014-04-23 00:00:00'
    cond_2 = data['Date'] <= '2024-04-23 00:00:00'

    data = data[cond_1 & cond_2].set_index('Date')
    print(data.shape)

    sns.set_style('whitegrid')  # 设置图片风格

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Close'], color='darkorange' ,label='AAPL')

    # 设置x轴为时间轴，并显示具体日期
    locator = mdates.AutoDateLocator(minticks=8, maxticks=12)  # 自动定位刻度
    formatter = mdates.DateFormatter('%Y-%m-%d')  # 自定义刻度标签格式
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # 设置标题
    plt.title('Close Price History',
              fontdict={'family': 'Times New Roman', 'fontsize': 16, 'color':'green'})
    plt.xticks(rotation=45) # 旋转刻度标签以提高可读性
    plt.ylabel('Close Price USD ($)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
    plt.legend(loc="upper right", prop={'family': 'Times New Roman'})
    plt.show()

    # 应用小波去噪到'Close'价格
    # data['Close'] = wavelet_denoising(data['Close'])
    data['Close'] = medfilt(data['Close'], kernel_size=3)

    # 只可视化去噪后的数据
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Denoised Data', color='blue')
    plt.title('Denoised Close Price History', fontdict={'family': 'Times New Roman', 'fontsize': 16, 'color': 'green'})
    plt.xticks(rotation=45)
    plt.ylabel('Close Price USD ($)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
    plt.legend(loc="upper right", prop={'family': 'Times New Roman'})
    plt.show()

    # 使用选定的特征来训练模型
    features = data.drop(['Adj Close', 'Volume'], axis=1)  # 删除Adj Close， Volume两列特征。
    target = data['Adj Close'].values.reshape(-1, 1)

    # 创建MinMaxScaler实例，对特征进行拟合和变换，生成NumPy数组
    scaler_features = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(features)

    # 创建MinMaxScaler实例，对目标进行拟合和变换，生成NumPy数组
    scaler_target = MinMaxScaler()
    target_scaled = scaler_target.fit_transform(target)

    print(features_scaled.shape, target_scaled.shape)

    # 使用随机森林回归进行特征重要性评估
    rf = RandomForestRegressor()
    rf.fit(features_scaled, target_scaled.ravel())
    importances = rf.feature_importances_
    feature_names = features.columns

    # 打印特征重要性
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(feature_importances)

    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()  # 反转y轴，使重要性高的特征排在上方
    plt.show()

    # 选择重要特征
    # 这里选择前3个最重要的特征进行预测
    selected_features = feature_importances['Feature'].head(2).values
    features_selected = data[selected_features]

    # 构建时间序列数据
    time_steps = 30
    X_list = []
    y_list = []

    for i in range(len(features_selected) - time_steps):
        X_list.append(features_selected[i:i+time_steps])
        y_list.append(target_scaled[i+time_steps])

    X = np.array(X_list) # [samples, time_steps, num_features]
    y = np.array(y_list) # [target]

    X_train, X_valid,\
        y_train, y_valid = train_test_split(X, y,
                                            test_size=0.2,
                                            random_state=45,
                                            shuffle=False)
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

    # 将 NumPy数组转换为 tensor张量
    X_train_tensor = torch.from_numpy(X_train).type(torch.Tensor)
    X_valid_tensor = torch.from_numpy(X_valid).type(torch.Tensor)
    y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor).view(-1, 1)
    y_valid_tensor = torch.from_numpy(y_valid).type(torch.Tensor).view(-1, 1)

    print(X_train_tensor.shape, X_valid_tensor.shape, y_train_tensor.shape, y_valid_tensor.shape)

    data_handler = DataHandler(X_train_tensor, y_train_tensor, X_valid_tensor, y_valid_tensor)
    train_loader = data_handler.train_loader()
    valid_loader = data_handler.valid_loader()
    num_features = 2
    input_dim = num_features  # 输入特征维度
    hidden_dim = 64  # LSTM 和 Transformer 的隐藏维度
    lstm_layers = 1  # LSTM 层数
    transformer_heads = 8  # Transformer 头数
    transformer_layers = 1  # Transformer 层数
    output_dim = 1  # 输出维度

    model = LSTM_Transformer(input_dim, hidden_dim, lstm_layers, transformer_heads, transformer_layers, output_dim)
    criterion_mse = nn.MSELoss()  # 定义均方误差损失函数
    criterion_mae = nn.L1Loss()  # 定义平均绝对误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # 定义优化器

    # batch_size, seq_len(time_steps), input_size(in_channels)
    summary(model, (32, time_steps, num_features))

    epoch = 500
    train_mselosses = []
    valid_mselosses = []
    train_maelosses = []
    valid_maelosses = []

    for epoch in range(epoch):
        train_loss_mse, train_loss_mae = train(model, train_loader, optimizer)
        valid_loss_mse, valid_loss_mae = evaluate(model, valid_loader)

        train_mselosses.append(train_loss_mse)
        valid_mselosses.append(valid_loss_mse)
        train_maelosses.append(train_loss_mae)
        valid_maelosses.append(valid_loss_mae)

        print(
            f'Epoch: {epoch + 1:02}, Train MSELoss: {train_loss_mse:.5f}, Train MAELoss: {train_loss_mae:.3f}, Val. MSELoss: {valid_loss_mse:.5f}, Val. MAELoss: {valid_loss_mae:.3f}')

    # 绘制 MSE损失图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_mselosses, label='Train MSELoss')
    plt.plot(valid_mselosses, label='Validation MSELoss')
    plt.xlabel('Epoch')
    plt.ylabel('MSELoss')
    plt.title('Train and Validation MSELoss')
    plt.legend()
    plt.grid(True)

    # 绘制 MAE损失图
    plt.subplot(1, 2, 2)
    plt.plot(train_maelosses, label='Train MAELoss')
    plt.plot(valid_maelosses, label='Validation MAELoss')
    plt.xlabel('Epoch')
    plt.ylabel('MAELoss')
    plt.title('Train and Validation MAELoss')
    plt.legend()
    plt.grid(True)

    plt.show()

    # 模型预测
    targets, predictions = prediction(model, valid_loader)

    # 反归一化
    denormalized_targets = scaler_target.inverse_transform(targets)
    denormalized_predictions = scaler_target.inverse_transform(predictions)

    # Visualize the data
    plt.figure(figsize=(12,6))
    plt.style.use('_mpl-gallery')
    plt.title('Comparison of validation set prediction results')
    plt.plot(denormalized_targets, color='blue',label='Actual Value')
    plt.plot(denormalized_predictions, color='orange', label='Valid Value')
    plt.legend()
    plt.show()

    plt.figure(figsize=(5, 5), dpi=100)
    sns.regplot(x=denormalized_targets, y=denormalized_predictions, scatter=True, marker="*", color='orange',line_kws={'color': 'red'})
    plt.show()

    mae = mean_absolute_error(targets, predictions)
    print(f"MAE: {mae:.4f}")

    mape = mean_absolute_percentage_error(targets, predictions)
    print(f"MAPE: {mape * 100:.4f}%")

    mse = mean_squared_error(targets, predictions)
    print(f"MSE: {mse:.4f}")

    rmse = root_mean_squared_error(targets, predictions)
    print(f"RMSE: {rmse:.4f}")

    r2 = r2_score(targets, predictions)
    print(f"R²: {r2:.4f}")







