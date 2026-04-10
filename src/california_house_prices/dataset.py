import torch
import os
import pandas as pd
import numpy as np
from . import utils

# 读取数据
train_data = pd.read_csv('data/california_house_prices/raw/train.csv')
test_data = pd.read_csv('data/california_house_prices/raw/test.csv')

# 统一处理数据集和测试集
all_data = pd.concat([train_data, test_data])

# 删除有大量文字内容的列
drop_cols = ['Id', 'Address', 'Sold Price', 'Summary', 'Type', 
                'Heating', 'Cooling', 'Parking', 
                'Region', 'Elementary School', 'Middle School', 
                'High School', 'Flooring', 'Appliances included', 
                'Laundry features', 'City', 'Zip', 'State']
all_data = all_data.drop(columns=[col for col in drop_cols if col in all_data.columns])

numeric_cols = utils.get_numeric_cols(all_data)
all_data[numeric_cols] = all_data[numeric_cols].astype('float32')

# 特殊处理 Bedrooms 列，部分数据中该列包含非数值内容，转换为数值类型，非数值部分记为缺失值
all_data['Bedrooms'] = pd.to_numeric(all_data['Bedrooms'], errors='coerce')

# 处理异常数值，使用1%和99%的分位数进行截断，防止极端值对模型训练的影响
all_data = utils.clip_outl(all_data, ['Lot', 'Total interior livable area'])

# 处理日期列，计算距离2021-01-01的天数
data_cols = ['Listed On', 'Last Sold On']
all_data = utils.date_to_days(all_data, date_cols=data_cols, ref_time='2021-01-01')
all_data['Year built'] = 2021 - all_data['Year built']

# 距离取exp(-x)，距离越远数值越小，距离越近数值越大，同时数值范围控制在0到1之间
dist_cols = ['Elementary School Distance', 'Middle School Distance', 'High School Distance']
all_data = utils.exp_trans(all_data, dist_cols)

# 独热编码
all_data = utils.one_hot(all_data)

# 分割数据集和测试集
train_data_size = train_data.shape[0]
train_features = all_data[:train_data_size]
test_features = all_data[train_data_size:]

# 对数据集进行标准化处理，使用训练集的均值和标准差进行标准化，确保测试集的处理与训练集一致
train_mean = train_features.mean()
train_std = train_features.std()
train_std = train_std.replace(0, 1e-8) 
train_features = utils.std_func(train_features, mean=train_mean, std=train_std)
test_features = utils.std_func(test_features, mean=train_mean, std=train_std)

# 获取训练集的标签、测试集的Id
train_labels = train_data['Sold Price']
test_id = test_data['Id']

# 将数据转换为 PyTorch 张量
train_features = torch.from_numpy(train_features.values.astype(np.float32))
test_features = torch.from_numpy(test_features.values.astype(np.float32))
train_labels = torch.from_numpy(train_labels.values.astype(np.float32))
test_id = torch.from_numpy(test_id.values.astype(np.int32))

# 将处理后的数据保存到字典中
dataset = {
    'train_features': train_features,
    'test_features': test_features,
    'train_labels': train_labels,
    'test_id': test_id
}

# 将处理后的数据保存为 .pt 文件
folder_path = 'data/california_house_prices/processed'
os.makedirs(folder_path, exist_ok=True)
for name, value in dataset.items():
    print(f'{name}.shape: {value.shape}') # 输出数据维度以验证正确性
    torch.save(value, os.path.join(folder_path, f'{name}.pt'))

print(f'数据预处理完成，处理后的数据已保存到 {folder_path}')