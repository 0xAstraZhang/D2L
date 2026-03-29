import pandas as pd
import numpy as np
import torch

# 读取数据
train_data = pd.read_csv('predict_house_price_2020/data/raw/train.csv')
test_data = pd.read_csv('predict_house_price_2020/data/raw/test.csv')

# 统一处理数据集和测试集
all_data = pd.concat([train_data.iloc[:, 1:], test_data.iloc[:, 1:]]) 

# 删除有大量文字内容的列
all_data = all_data.drop(columns=['Address', 'Sold Price', 'Summary', 
                                      'Heating', 'Cooling', 'Parking', 
                                      'Elementary School', 'Middle School', 'High School', 
                                      'Zip'])

# 定义函数来获取数值列索引
def numeric(df):
    return df.dtypes[df.dtypes != 'object'].index

all_data[numeric(all_data)] = all_data[numeric(all_data)].apply(pd.to_numeric, errors='coerce')

# 定义函数来处理异常值
def outlier_handling(df):
    df = df.copy()
    # 计算1%分位数和99%分位数
    q1 = df.quantile(0.01)
    q99 = df.quantile(0.99)
    # 去除掉异常值
    df = df.where((df >= q1) & (df <= q99), np.nan)
    return df

all_data['Lot'] = outlier_handling(all_data['Lot'])
all_data['Total interior livable area'] = outlier_handling(all_data['Total interior livable area'])

# 处理字符串和数字混合的列，保留数字，字符串记为缺失值
all_data['Bedrooms'] = pd.to_numeric(all_data['Bedrooms'], errors='coerce')

# 定义一个函数来处理日期列
def data_conv(df):
    df = df.copy()
    # 将数据转换为时间格式，遇到不合法格式记为NaT
    df = pd.to_datetime(df, errors='coerce')
    # 指定2020-01-01为参考时间
    ref_time = pd.Timestamp('2020-01-01')
    # 数据与参考时间相减，得到时间差；NaT变为NaN
    df = (ref_time - df).dt.days
    return df

all_data['Listed On'] = data_conv(all_data['Listed On'])
all_data['Last Sold On'] = data_conv(all_data['Last Sold On'])

# 距离取exp(-x)，距离越远数值越小，距离越近数值越大，同时数值范围控制在0到1之间
all_data['Elementary School Distance'] = np.exp(-all_data['Elementary School Distance'])
all_data['Middle School Distance'] = np.exp(-all_data['Middle School Distance'])
all_data['High School Distance'] = np.exp(-all_data['High School Distance'])

# 定义标准化函数
def standardization(df):
    df = df.copy()
    # 均值为0，方差为1
    df.loc[:, numeric(df)] = df[numeric(df)].apply(
    lambda x: (x - x.mean()) / (x.std())
)
    # NaN部分替换为0
    df.loc[:, numeric(df)] = df[numeric(df)].fillna(0)
    return df

# 定义One-Hot函数
def one_hot(df):
    df = df.copy()
    new_cols =  [] # 存放新生成的 dummy 列
    cols_to_drop = [] # 记录要删除的原列名
    for col in df.columns:
        if df[col].dtype == 'object':
            cols_to_drop.append(col)
            # 判断是否是多标签列（有无','）
            if df[col].astype(str).str.contains(',').any():
                dummies = df[col].fillna('None').str.get_dummies(sep=', ')
                dummies = dummies.add_prefix(f"{col}_")
                new_cols.append(dummies)
            # 处理单标签列
            else:
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                new_cols.append(dummies)
    # 删除原始列
    df = df.drop(columns=cols_to_drop)
    # 合并新列
    df = pd.concat([df] + new_cols, axis=1)
    return df

# 分割数据集和测试集，记录训练集的样本数量，以便后续分割数据集和测试集
num_train = train_data.shape[0]
train_features = all_data[:num_train]
test_features = all_data[num_train:]

# 对数据集和测试集分别进行标准化处理，防止数据污染
train_features, test_features = standardization(train_features), standardization(test_features)

# 再次合并数据集和测试集，进行One-Hot编码，保持列的一致性
all_data = pd.concat([train_features.iloc[:, :], test_features.iloc[:, :]]) 
all_data = one_hot(all_data)

# 分割数据集和测试集，并对标签进行对数变换，减小数值范围
train_features = all_data[:num_train]
test_features = all_data[num_train:]
train_labels = np.log1p(train_data['Sold Price'].values).reshape(-1, 1)

# 将数据转换为 PyTorch 张量并保存为 .pt 文件    
train_features = torch.from_numpy(train_features.values.astype(np.float32))
test_features = torch.from_numpy(test_features.values.astype(np.float32))
train_labels = torch.from_numpy(train_labels.astype(np.float32))

# 输出数据维度以验证正确性
print(train_features.shape)
print(test_features.shape)
print(train_labels.shape)

# 将处理后的数据保存为 .pt 文件，供模型训练使用
torch.save(train_features, 'predict_house_price_2020/data/processed/train_features.pt')
torch.save(test_features, 'predict_house_price_2020/data/processed/test_features.pt')
torch.save(train_labels, 'predict_house_price_2020/data/processed/train_labels.pt')

