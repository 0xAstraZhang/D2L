import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import classify_leaves.utils as utils
import torchvision.models as models
import os
from sklearn.model_selection import StratifiedKFold


# 数据增强
train_augs = torchvision.transforms.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomResizedCrop(200, scale=(0.8, 1.0)),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.ToTensor(),
])

# 超参数
batch_size = 32 # 批量大小
out_features = 10 # 输出类别数
lr = 0.001 # 学习率
epochs = 20 # 训练轮数
k = 5 # k折交叉验证
seed = 176 # 固定种子
patience = 3 # early-stopping耐心值

# 自定义数据集类
class LeavesDataset(Dataset):
    def __init__(self, csv_file, img_dir, resize_width=256, resize_height=256, transform=None):
        self.df = pd.read_csv(os.path.join(img_dir, csv_file), header=None)
        self.img_dir = img_dir
        self.transform = transform  
        self.classes = sorted(self.df.iloc[:, 1].unique())
        self.cls_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_cls = {idx: cls_name for idx, cls_name in enumerate(self.classes)}
        self.labels = np.array(self.df.iloc[:, 1].map(self.cls_to_idx).values, dtype=np.int32)
        self.resize_width = resize_width
        self.resize_height = resize_height
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx, 0]
        label = self.labels[idx]

        image_path = os.path.join(self.img_dir, image_name)
        image = torchvision.io.read_image(image_path)
        image = T.Resize((self.resize_height, self.resize_width))(image)

        if self.transform:
            image = self.transform(image)

        return image, label

train_dataset = LeavesDataset(csv_file='train.csv', img_dir='data/classify_leaves/', transform=train_augs)

# 加载预训练的ResNet34模型
model = models.resnet34(pretrained=True)

# 冻结全局参数
for param in model.parameters():
    param.requires_grad = False

# 解冻最后一个Block参数
for param in model.layer4.parameters():
    param.requires_grad = True

# 替换全连接层
model.fc = nn.Linear(model.fc.in_features, out_features)

# 定义损失函数和优化器
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 获取设备
device = utils.get_device()

# 创建模型保存目录
model_dir = 'model/classify_leaves/'
os.makedirs(model_dir, exist_ok=True) 

# 训练模型
def train(model, train_loader, val_loader, loss, optimizer, device):
    animator = utils.Animator(xlabel='epoch', ylabel='value', legend=['train loss', 'train acc', 'val acc'])
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_val_loss = float('inf')
    model.to(device)
    for epoch in range(epochs):

        # 训练阶段
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            train_loss += l.item()
    
        # 记录训练损失和准确率
        train_loss_history.append(train_loss / len(train_loader))
        train_acc = utils.calculate_accuracy(model, train_loader, device)
        train_acc_history.append(train_acc)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                l = loss(outputs, labels)
                val_loss += l.item()

        # 记录验证损失和准确率
        val_loss_history.append(val_loss / len(val_loader))
        val_acc = utils.calculate_accuracy(model, val_loader, device)
        val_acc_history.append(val_acc)

        # 保存当前折内最佳模型
        if val_loss_history[-1] < best_val_loss:
            best_val_loss = val_loss_history[-1]
            torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_fold{fold+1}.pth'))

        # 早停判断
        if utils.early_stopping(val_acc_history, patience=patience):
            print(f"Early stopping at epoch {epoch+1}")
            break

        # 更新动画
        animator.add(epoch + 1, [train_loss_history[-1], train_acc_history[-1], val_acc_history[-1]])
    print(f"Best Validation Loss: {best_val_loss:.4f}")
 
# k折交叉验证
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(train_dataset)), train_dataset.labels)):
    print(f'Fold {fold+1}/{k}')
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    train(model, train_loader, val_loader, loss, optimizer, device)
    