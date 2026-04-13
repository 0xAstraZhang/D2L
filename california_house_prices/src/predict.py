import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from .model import HousePriceModel
import utils
def test():
    # 获取计算设备  
    device = utils.get_device()
    print(f'本次测试所用计算设备: {device}')

    # 加载测试数据
    features = torch.load('california_house_prices/data/processed/test_features.pt', map_location='cpu')

    # 定义模型并加载训练好的参数
    model = HousePriceModel(in_features=features.shape[1])
    model.load_state_dict(torch.load('california_house_prices/model/model_weights.pth', map_location='cpu'))
    model.to(device)
    model.eval()

    # 进行预测
    with torch.no_grad():
        features = features.to(device)
        predictions = model(features)
        predictions = predictions.detach().cpu().numpy().reshape(-1)
    return predictions

if __name__ == '__main__':
    predictions = test()
    test_id = torch.load('california_house_prices/data/processed/test_id.pt', map_location='cpu').numpy().reshape(-1).astype(int)
    result = pd.DataFrame({
        'Id': test_id, 'Sold Price': predictions
    })

    folder_path = 'california_house_prices/data/predictions'
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, 'predictions.csv')
    
    result.to_csv(save_path, index=False)
    print(f'预测结果已保存到 {save_path}')
    
