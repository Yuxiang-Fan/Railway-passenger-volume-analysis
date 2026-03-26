import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def backward_stepwise_selection(X, y, criterion='aic'):
    # 基于信息准则的后向逐步回归，用于处理多重共线性与特征降维
    current_features = X.columns.tolist()
    
    # 拟合包含所有特征的初始 OLS 模型
    X_init = sm.add_constant(X[current_features])
    best_score = getattr(sm.OLS(y, X_init).fit(), criterion)
    
    print(f"--- Backward Stepwise Selection ({criterion.upper()}) ---")
    print(f">> Initial {criterion.upper()}: {best_score:.4f}")
    
    while current_features:
        step_scores = {}
        
        for feature in current_features:
            reduced_features = list(set(current_features) - {feature})
            
            # 处理特征被全部剔除的极端情况 (仅保留常数项)
            if not reduced_features:
                X_subset = sm.add_constant(pd.DataFrame(index=X.index))
            else:
                X_subset = sm.add_constant(X[reduced_features])
            
            model = sm.OLS(y, X_subset).fit()
            # 动态获取对应的评估准则属性 (如 model.aic)
            step_scores[feature] = getattr(model, criterion)
            
        # 找出剔除后准则值最小的特征
        best_drop = min(step_scores, key=step_scores.get)
        best_step_score = step_scores[best_drop]
        
        # 判断是否接受剔除
        if best_step_score < best_score:
            current_features.remove(best_drop)
            best_score = best_step_score
            print(f">> Drop: {best_drop: <20} | New {criterion.upper()}: {best_score:.4f}")
        else:
            print(f">> 停止搜索: 剔除任何现有特征均会导致 {criterion.upper()} 上升。")
            break
            
    return current_features

def evaluate_model(X, y, selected_features):
    # 核心特征子集的 OLS 建模与拟合优度评估
    X_final = sm.add_constant(X[selected_features])
    model = sm.OLS(y, X_final).fit()
    
    print("\n--- Final Model Summary ---")
    print(f"Selected Features: {selected_features}")
    print(model.summary())
    
    y_pred = model.predict(X_final)
    
    # 引入无量纲的 R2 和调整后 R2，增强模型说服力
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(selected_features) - 1)
    mse = mean_squared_error(y, y_pred)
    
    print("\n--- Model Performance ---")
    print(f">> R-squared:          {r2:.4f}")
    print(f">> Adjusted R-squared: {adj_r2:.4f}")
    print(f">> Normalized MSE:     {mse:.4e} (基于预处理后的无量纲数据)")
    
    plt.figure(figsize=(10, 6))
    plt.plot(y.values, label='Actual', marker='o', linestyle='-', alpha=0.8)
    plt.plot(y_pred.values, label='Predicted', marker='^', linestyle='--', alpha=0.8)
    
    plt.title('Stepwise Regression: Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Target Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out_dir = 'outputs/figures'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'stepwise_fit.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n>> 拟合图已保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    # 提供通用的数据接口规范，而非强绑定特定文件
    data_file = 'data/processed/dataset_ready.xlsx'
    
    if os.path.exists(data_file):
        print(f"加载数据集: {data_file}")
        df = pd.read_excel(data_file)
        
        # 假设第一列为时间或ID索引，最后一列为目标变量
        X_data = df.iloc[:, 1:-1]
        y_data = df.iloc[:, -1]
        
        optimal_features = backward_stepwise_selection(X_data, y_data, criterion='aic')
        evaluate_model(X_data, y_data, optimal_features)
    else:
        print(f">> [提示] 未找到 {data_file}。")
        print(">> 请确保原始数据已通过数据预处理流水线，并放置在正确目录。")
