import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def apply_log_transform(df, cols):
    # 针对偏态数据做对数平滑，处理异方差
    res_df = df.copy()
    for c in cols:
        # 兼容数据中含0或负数的情况
        if (res_df[c] <= 0).any():
            res_df[c] = np.log1p(res_df[c])
        else:
            res_df[c] = np.log(res_df[c])
    return res_df

def minmax_scale(df, cols):
    # 归一化到 [0, 1]
    res_df = df.copy()
    scaler = MinMaxScaler()
    res_df[cols] = scaler.fit_transform(res_df[cols])
    return res_df, scaler

def zscore_scale(df, cols):
    # Z-score 标准化，主要供 SVR 和岭回归等对量纲敏感的模型使用
    res_df = df.copy()
    scaler = StandardScaler()
    res_df[cols] = scaler.fit_transform(res_df[cols])
    return res_df, scaler

if __name__ == '__main__':
    # 路径配置
    input_path = '../../data/processed/data_interpolated.xlsx'
    if not os.path.exists(input_path):
        input_path = 'data_interpolated.xlsx'  # 降级到当前目录找文件

    df = pd.read_excel(input_path)

    # 划分特征和目标变量
    features = [
        'GDP(现价):第三产业:年', 
        'GDP(现价):交通运输、仓储和邮政业:年',
        '城镇单位就业人员平均工资:年', 
        '铁路旅客周转量:年',
        '铁路客车拥有量:软卧车:年', 
        '铁路客车拥有量:软座车:年',
        '铁路客车拥有量:硬座车:年', 
        '高速铁路营业里程:年'
    ]
    targets = ['客运量:铁路:年']
    process_cols = features + targets

    # 1. 归一化 (去量纲)
    df_norm, _ = minmax_scale(df, process_cols)

    # 2. 对数变换 
    df_log = apply_log_transform(df_norm, process_cols)

    # 3. Z-score 标准化
    df_final, _ = zscore_scale(df_log, process_cols)

    # 结果保存
    out_dir = '../../data/processed'
    os.makedirs(out_dir, exist_ok=True)
    
    log_out = os.path.join(out_dir, 'data_for_stat_tests.xlsx')
    final_out = os.path.join(out_dir, 'data_transformed_final.xlsx')
    
    df_log.to_excel(log_out, index=False)
    df_final.to_excel(final_out, index=False)
    
    print(">> 数据转换流水线执行完毕。")
    print(f"统计检验版本 (ADF等) -> {log_out}")
    print(f"模型训练版本 (SVR/Ridge) -> {final_out}")
