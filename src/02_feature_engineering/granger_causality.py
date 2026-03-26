import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def granger_test(df, target_col, feature_cols, max_lag=3):
    # 执行 Granger 检验，重点捕获小样本下的 VAR 完美拟合异常
    print(f"--- Granger Causality Test (Target: {target_col}, Max Lag: {max_lag}) ---")
    
    status_dict = {}

    for feat in feature_cols:
        test_data = df[[target_col, feat]].dropna()
        
        try:
            res = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            
            print(f">> {feat} -> {target_col}:")
            for lag in range(1, max_lag + 1):
                p_val = res[lag][0]['ssr_ftest'][1]
                sig = "显著" if p_val < 0.05 else "不显著"
                print(f"   Lag {lag}: p-value = {p_val:.4f} ({sig})")
                
            status_dict[feat] = "计算成功"
            
        except Exception as e:
            err_str = str(e).lower()
            # 捕获 statsmodels 在自由度耗尽时抛出的完美拟合异常
            if "perfect fit" in err_str:
                print(f">> [警告] {feat} -> {target_col}: VAR 完美拟合 (样本量不足)，跳过检验。")
                status_dict[feat] = "完美拟合"
            else:
                print(f">> [错误] {feat}: {err_str}")
                status_dict[feat] = "计算异常"

    return status_dict

if __name__ == '__main__':
    try:
        df = pd.read_excel('../../data/processed/data_for_stat_tests.xlsx')
    except FileNotFoundError:
        df = pd.read_excel('data_sd.xlsx')

    target = df.columns[-1]
    features = df.columns[1:-1]

    summary = granger_test(df, target, features, max_lag=3)
    
    print("\n--- Granger Test 状态汇总 ---")
    for f, s in summary.items():
        print(f"{f}: {s}")
