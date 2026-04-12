import torch
from torch.backends import mps
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import torchvision
from torch.utils.data import Dataset

# 自定义数据集类
class LeavesDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        super().__init__()
        self.df = pd.read_csv(os.path.join(img_dir, csv_file))
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.df.iloc[:, 1].unique())
        self.cls_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_cls = {idx: cls_name for idx, cls_name in enumerate(self.classes)}
        self.labels = np.array(self.df.iloc[:, 1].map(self.cls_to_idx).values, dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx, 0]
        label = self.labels[idx]

        image_path = os.path.join(self.img_dir, image_name)
        image = torchvision.io.read_image(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


# 定义device自动选择函数
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# 画训练曲线的函数
def plot_history(train_loss, train_acc, val_acc, fold, model_dir):
    """训练结束后，画一张静态图并保存到本地"""
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, train_acc, 'g--', label='Train Acc')
    plt.plot(epochs, val_acc, 'r-.', label='Val Acc')
    
    plt.title(f'Fold {fold + 1} Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # 保存图片到模型目录下
    save_path = os.path.join(model_dir, f'learning_curve_fold{fold + 1}.png')
    plt.savefig(save_path)
    plt.close() # 关闭画板释放内存
    print(f"第 {fold + 1} 折的学习曲线已保存至: {save_path}")

# 训练模型
def train(model, train_loader, val_loader, loss, optimizer, device, epochs, patience, model_dir, fold):
    # 将模型移动到设备上
    model.to(device)
    
    # 创建空列表用于记录训练和验证的损失与准确率
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # 用于记录当前折内最佳验证损失
    best_val_loss = float('inf')
    
    # 早停计数器
    early_stop_counter = 0

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            _, predicted = torch.max(outputs.data, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 记录训练损失和准确率
        train_loss_history.append(train_loss / len(train_loader))
        train_acc_history.append(train_correct / train_total)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loss, val_correct, val_total = 0.0, 0, 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                l = loss(outputs, labels)
                val_loss += l.item()
                # 计算验证准确率
                _, predicted = torch.max(outputs.data, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 记录验证损失和准确率
        val_loss_history.append(val_loss / len(val_loader))
        val_acc_history.append(val_correct / val_total)

        # 打印当前epoch的训练和验证结果
        print(f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss_history[-1]:.4f}, Train Acc: {train_acc_history[-1]:.4f} | "
            f"Val Loss: {val_loss_history[-1]:.4f}, Val Acc: {val_acc_history[-1]:.4f}")

        # 保存当前折内最佳模型及早停判断
        if val_loss_history[-1] < best_val_loss:
            best_val_loss = val_loss_history[-1]
            torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_fold{fold+1}.pth'))
            early_stop_counter = 0  # 表现变好，清零计数器
        else:
            early_stop_counter += 1 # 表现没变好，增加计数器

        # 早停判断
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"已保存第 {fold + 1} 折内最佳模型，其测试损失为: {best_val_loss:.4f}")
    plot_history(train_loss_history, train_acc_history, val_acc_history, fold, model_dir)
