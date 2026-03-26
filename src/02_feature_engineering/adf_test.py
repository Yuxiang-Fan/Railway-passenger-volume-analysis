import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def auto_adf_test(series, var_name, max_diff=3, alpha=0.05):
    # 基于 ADF 检验自动寻找序列的单整阶数，并返回平稳后的序列
    ts = series.dropna()
    
    for d in range(max_diff + 1):
        if d > 0:
            ts = ts.diff().dropna()
            
        # 异常捕获：防止差分后序列方差为0导致 adfuller 算法崩溃
        try:
            res = adfuller(ts, autolag='AIC')
            p_value = res[1]
            
            if p_value < alpha:
                print(f">> {var_name:<20} | I({d}) 平稳 | p-value: {p_value:.4f}")
                return d, p_value, ts
        except Exception as e:
            print(f">> [警告] {var_name}: 检验异常 ({str(e)})")
            return None, None, ts
            
    print(f">> [警告] {var_name:<20} | 达到最大差分阶数({max_diff})仍未平稳 | p-value: {p_value:.4f}")
    return max_diff, p_value, ts

def process_stationarity(df, max_diff=3, alpha=0.05):
    # 批量处理数据集的平稳性检验与自动差分流水线
    print(f"--- 自动 ADF 检验与差分流水线 (Alpha={alpha}) ---")
    diff_orders = {}
    
    # 构建空 DataFrame 承接差分后的数据，保留原始索引以便对齐
    stationary_data = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        # 工程防御：仅对数值型特征进行平稳性检验，跳过时间或类别列
        if pd.api.types.is_numeric_dtype(df[col]):
            d, p_val, ts_diff = auto_adf_test(df[col], col, max_diff, alpha)
            if d is not None:
                diff_orders[col] = d
                # Pandas 会自动根据 index 对齐数据，差分产生的头部 NaN 会被保留
                stationary_data[col] = ts_diff
                
    return diff_orders, stationary_data

if __name__ == '__main__':
    # 统一的数据管道接口规范
    data_file = 'data/processed/dataset_ready.xlsx'
    
    if os.path.exists(data_file):
        print(f">> 加载时间序列数据集: {data_file}")
        df = pd.read_excel(data_file)
        
        # 假设第一列是时间索引（无需检验）
        time_col = df.columns[0]
        features_df = df.drop(columns=[time_col])
        
        # 将最大差分阶数降低为 3，防止宏观数据过度差分引入移动平均噪声
        orders, df_stationary = process_stationarity(features_df, max_diff=3)
        
        print("\n--- 单整阶数 (Integration Order) 汇总 ---")
        for c, d in orders.items():
            print(f" - {c}: I({d})")
            
        # 重新拼接时间列，并导出平稳化后的数据，供下游 VAR 或 Granger 模型使用
        df_stationary.insert(0, time_col, df[time_col])
        
        out_dir = 'data/processed'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'data_stationary.xlsx')
        
        df_stationary.to_excel(out_path, index=False)
        print(f"\n>> 平稳化后的数据集已安全导出至: {out_path}")
    else:
        print(f">> [提示] 未找到 {data_file}。请确保数据已完成预处理。")
