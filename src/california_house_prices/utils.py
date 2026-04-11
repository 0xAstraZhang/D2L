import torch
import pandas as pd
import numpy as np
from torch.backends import mps  

# 定义对数均方差损失函数
def rmsle(y_pred, y_true):
    log_pred = torch.log1p(y_pred)
    log_true = torch.log1p(y_true)
    return torch.sqrt(torch.mean((log_pred - log_true) ** 2))

# 定义device自动选择函数
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# 定义函数来处理异常数值
def clip_outl(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q99)
    return df 
    
# 定义函数来获取数值列索引
def get_numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

# 定义标准化函数
def std_func(df, mean, std):
    df = df.copy()
    # 标准化处理
    df = (df - mean) / std
    # NaN部分替换为0
    df = df.fillna(0)
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

# 距离取exp(-x)，距离越远数值越小，距离越近数值越大，同时数值范围控制在0到1之间
def exp_trans(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0).fillna(0)
            df[col] = np.exp(-df[col])
    return df

# 定义函数来处理日期列
def date_to_days(df, date_cols, ref_time):
    for col in date_cols:
        if col in df.columns:
            df[col] = (pd.Timestamp(ref_time) - pd.to_datetime(df[col], errors='coerce')).dt.days
    return df
