import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler # 导入分布式采样器
import torch.distributed as dist # 导入分布式库
from torch.nn.parallel import DistributedDataParallel as DDP # 导入DDP
import utils    
import torchvision.models as models
import os
import random
from sklearn.model_selection import StratifiedKFold
import warnings

# 忽略sklearn的UserWarning
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 设置固定的随机种子，保证所有进程的K折划分和参数初始化完全一致
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# 数据增强
train_augs = torchvision.transforms.Compose([
    T.Resize(256, antialias=True), # type: ignore
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.ConvertImageDtype(torch.float32), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_augs = torchvision.transforms.Compose([
    T.Resize(256, antialias=True), # type: ignore
    T.CenterCrop(224),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 超参数
batch_size = 128
out_features = 177
lr = 0.001
epochs = 20
k = 3
patience = 3
num_workers = 4 # 每张卡的worker数

# DDP初始化
def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return False, 0, 'cuda:0'
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier() # 等待所有进程到达这里
    return True, local_rank, device

# 创建模型保存目录
model_dir = 'model/classify_leaves/'
if int(os.environ.get('RANK', 0)) == 0: # 只有主进程创建目录
    os.makedirs(model_dir, exist_ok=True) 

if __name__ == '__main__':
    # 初始化DDP
    is_dist, local_rank, device = init_distributed_mode()

    # 加载数据集
    dataset = utils.LeavesDataset(csv_file='train.csv', img_dir='data/classify_leaves/', transform=None)
    train_dataset = utils.LeavesDataset(csv_file='train.csv', img_dir='data/classify_leaves/', transform=train_augs)
    val_dataset= utils.LeavesDataset(csv_file='train.csv', img_dir='data/classify_leaves/', transform=val_augs)

    # k折交叉验证，必须指定random_state确保各进程划分一致
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), dataset.labels)):
        if local_rank == 0:
            print(f'========== Fold {fold+1}/{k} ==========')
        
        # 加载预训练的ResNet34模型
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # 冻结全局参数
        for param in model.parameters():
            param.requires_grad = False

        # 解冻最后一个Block参数
        for param in model.layer4.parameters():
            param.requires_grad = True

        # 替换全连接层
        model.fc = nn.Linear(model.fc.in_features, out_features)
        
        # 将模型移动到对应的GPU并用DDP包装
        model.to(device)

        # 只有在分布式环境下才使用DDP包装模型
        if is_dist:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # 定义损失函数和优化器
        loss = nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr) 

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        
        # DDP环境下必须使用 DistributedSampler 分配数据
        train_sampler = DistributedSampler(train_subset, shuffle=True) if is_dist else None
        val_sampler = DistributedSampler(val_subset, shuffle=False) if is_dist else None
        
        # 注意：使用 DistributedSampler 时 DataLoader 不能设置 shuffle=True
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, 
                                  shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler, 
                                shuffle=False, num_workers=num_workers, pin_memory=True)

        if local_rank == 0:
            print(f"每张卡分配的 训练集大小: {len(train_subset)//dist.get_world_size()}, 验证集大小: {len(val_subset)//dist.get_world_size()}")
        
        utils.train(model, train_loader, val_loader, loss, optimizer, device, epochs, patience, model_dir, fold, local_rank, train_sampler)
    
    # 销毁分布式环境
    if is_dist:
        dist.destroy_process_group()