import os
import torch
import utils
import matplotlib.pyplot as plt
from torch import optim
from model import HousePriceModel
from torch.utils.data import TensorDataset, DataLoader

# 超参数设定
lr = 0.005
epochs = 40
batch_size = 128
num_workers = 0
weight_decay = 1e-5

# 训练函数
def train():
    # 获取计算设备  
    device = utils.get_device()
    print(f'本次训练所用计算设备: {device}')

    # 加载训练数据
    features = torch.load('predict_house_price_2020/data/processed/train_features.pt', map_location='cpu')
    labels = torch.load('predict_house_price_2020/data/processed/train_labels.pt', map_location='cpu').reshape(-1, 1)

    # 定义模型、损失函数和优化器
    model = HousePriceModel(in_features=features.shape[1])
    loss = utils.rmsle
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    # 创建数据集和数据加载器
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 记录epoch损失用于绘图
    loss_history = []

    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            # 前向传播
            pre_y = model(batch_features)
            batch_loss = loss(pre_y, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # 记录epoch损失
            epoch_loss += batch_loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
    
    folder_path = 'predict_house_price_2020/model'
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, 'house_price_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f'训练参数已保存到 {save_path}')

    # 绘制损失曲线
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss') 
    plt.title('Training Loss')
    plt.show()

if __name__ == '__main__':
    train()