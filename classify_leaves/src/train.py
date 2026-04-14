import torch
import os
import utils    
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from sklearn.model_selection import StratifiedKFold

# 超参数
batch_size = 128 # 批量大小
lr = 0.001 # 学习率
epochs = 20 # 训练轮数
k = 3 # k折交叉验证
patience = 3 # early-stopping耐心值
num_workers = 4 # 数据加载时使用的线程数

def main():
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    if __name__ == '__main__':
        # 加载数据集
        dataset, train_dataset, val_dataset = utils.data_loader()
        
        # 获取设备
        device = utils.get_device()
        
        # 创建模型保存目录
        model_dir = 'classify_leaves/model/'
        os.makedirs(model_dir, exist_ok=True) 
    
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), dataset.labels)):
            # 打印当前折数
            print(f'Fold {fold+1}/{k}')

            # 加载预训练的ResNet34模型
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

            # 替换全连接层
            model.fc = nn.Linear(model.fc.in_features, 176)

            # 定义损失函数和优化器
            loss = nn.CrossEntropyLoss() # 交叉熵损失函数适用于多分类问题
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # 只优化需要更新的参数

            # 创建训练和验证数据加载器
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(val_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
            # 打印当前折的训练和验证集大小以及使用的设备
            print(f"训练集大小: {len(train_subset)}, 验证集大小: {len(val_subset)}, 本次训练使用的设备: {device}")

            # 调用训练函数
            utils.train(model, train_loader, val_loader, loss, optimizer, device, epochs, patience, model_dir, fold)


if __name__ == '__main__':
    main()