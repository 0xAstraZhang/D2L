import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import torchvision
from torch.utils.data import Dataset
import torch.distributed as dist

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
    
    save_path = os.path.join(model_dir, f'learning_curve_fold{fold + 1}.png')
    plt.savefig(save_path)
    plt.close() 
    print(f"第 {fold + 1} 折的学习曲线已保存至: {save_path}")

# 同步各卡的指标数据
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

# 训练模型 (增加了 local_rank 和 train_sampler 参数)
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience, model_dir, fold, local_rank, train_sampler):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        # DDP必须在每个epoch开始时set_epoch，以保证每轮shuffle的结果不同
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            l = criterion(outputs, labels)
            l.backward()
            optimizer.step()
            
            train_loss += l.item() * labels.size(0) # 改为乘batch_size，方便后续全局求和
            _, predicted = torch.max(outputs.data, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                l = criterion(outputs, labels)
                
                val_loss += l.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # ---------- 汇总4张卡的统计数据 ----------
        metrics = torch.tensor([
            train_loss, train_correct, train_total,
            val_loss, val_correct, val_total
        ], dtype=torch.float32).to(device)
        
        if dist.is_initialized():
            metrics = reduce_tensor(metrics)
            
        # 计算全局指标
        g_train_loss = metrics[0].item() / metrics[2].item()
        g_train_acc = metrics[1].item() / metrics[2].item()
        g_val_loss = metrics[3].item() / metrics[5].item()
        g_val_acc = metrics[4].item() / metrics[5].item()

        train_loss_history.append(g_train_loss)
        train_acc_history.append(g_train_acc)
        val_loss_history.append(g_val_loss)
        val_acc_history.append(g_val_acc)

        # 只在主进程(卡0)打印和保存模型，避免控制台混乱和文件冲突
        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {g_train_loss:.4f}, Train Acc: {g_train_acc:.4f} | "
                f"Val Loss: {g_val_loss:.4f}, Val Acc: {g_val_acc:.4f}")

            if g_val_loss < best_val_loss:
                best_val_loss = g_val_loss
                # 注意：DDP模型保存时要使用 model.module，这样单卡预测时也能直接加载
                torch.save(model.module.state_dict(), os.path.join(model_dir, f'best_model_fold{fold+1}.pth'))
                early_stop_counter = 0 
            else:
                early_stop_counter += 1 

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
        
        # 广播early_stop的标志到所有卡，保证所有进程同步退出
        early_stop_tensor = torch.tensor([early_stop_counter], dtype=torch.int32).to(device)
        if dist.is_initialized():
            dist.broadcast(early_stop_tensor, src=0)
        if early_stop_tensor.item() >= patience:
            break

    if local_rank == 0:
        print(f"已保存第 {fold + 1} 折内最佳模型，其测试损失为: {best_val_loss:.4f}")
        plot_history(train_loss_history, train_acc_history, val_acc_history, fold, model_dir)