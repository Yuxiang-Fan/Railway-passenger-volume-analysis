import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def auto_adf_test(series, var_name, max_diff=4, alpha=0.05):
    # 基于 ADF 检验自动寻找序列的单整阶数 d
    ts = series.dropna()
    
    for d in range(max_diff + 1):
        if d > 0:
            ts = ts.diff().dropna()
            
        res = adfuller(ts, autolag='AIC')
        p_value = res[1]
        
        if p_value < alpha:
            print(f">> {var_name}: I({d}) 平稳, p-value={p_value:.4f}")
            return d, p_value, ts
            
    print(f">> [警告] {var_name}: 达到最大差分阶数({max_diff})仍未平稳, p-value={p_value:.4f}")
    return max_diff, p_value, ts

if __name__ == '__main__':
    try:
        df = pd.read_excel('../../data/processed/data_for_stat_tests.xlsx')
    except FileNotFoundError:
        df = pd.read_excel('chafen_4.xlsx')
        
    # 假设第0列是时间列，从第1列开始是特征
    cols = df.columns[1:] 
    
    print("--- ADF Test 执行 ---")
    diff_orders = {}
    
    for c in cols:
        d, p_val, _ = auto_adf_test(df[c], c, max_diff=4)
        diff_orders[c] = d
        
    print("\n--- 单整阶数 (I(d)) 汇总 ---")
    for c, d in diff_orders.items():
        print(f"{c}: I({d})")
