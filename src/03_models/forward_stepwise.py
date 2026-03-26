import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def forward_stepwise_selection(X, y, criterion='aic'):
    # 基于信息准则的前向逐步回归特征筛选 (贪心算法)
    initial_features = X.columns.tolist()
    best_features = []
    
    # 拟合仅包含常数项的空模型，获取基准得分
    X_null = sm.add_constant(pd.DataFrame(index=X.index))
    best_score = getattr(sm.OLS(y, X_null).fit(), criterion)
    
    print(f"--- Forward Stepwise Selection ({criterion.upper()}) ---")
    print(f">> Null Model {criterion.upper()}: {best_score:.4f}")
    
    while initial_features:
        remaining_features = list(set(initial_features) - set(best_features))
        step_scores = {}
        
        for feature in remaining_features:
            model_features = best_features + [feature]
            X_subset = sm.add_constant(X[model_features])
            
            model = sm.OLS(y, X_subset).fit()
            step_scores[feature] = getattr(model, criterion)
            
        # 找出加入后使准则值最小的特征
        best_add = min(step_scores, key=step_scores.get)
        best_step_score = step_scores[best_add]
        
        # 判断是否接受加入新特征
        if best_step_score < best_score:
            best_features.append(best_add)
            best_score = best_step_score
            print(f">> Add: {best_add: <20} | New {criterion.upper()}: {best_score:.4f}")
        else:
            print(f">> 停止搜索: 加入任何剩余特征均会导致 {criterion.upper()} 上升。")
            break
            
    return best_features

def evaluate_model(X, y, selected_features):
    # 核心特征子集的 OLS 建模与拟合优度评估
    X_final = sm.add_constant(X[selected_features])
    model = sm.OLS(y, X_final).fit()
    
    print("\n--- Final Model Summary ---")
    print(f"Selected Features: {selected_features}")
    print(model.summary())
    
    y_pred = model.predict(X_final)
    
    # 引入无量纲的 R2 和调整后 R2
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(selected_features) - 1)
    mse = mean_squared_error(y, y_pred)
    
    print("\n--- Model Performance ---")
    print(f">> R-squared:          {r2:.4f}")
    print(f">> Adjusted R-squared: {adj_r2:.4f}")
    print(f">> Normalized MSE:     {mse:.4e} (基于预处理数据)")
    
    plt.figure(figsize=(10, 6))
    plt.plot(y.values, label='Actual', marker='o', linestyle='-', alpha=0.8)
    plt.plot(y_pred.values, label='Predicted', marker='^', linestyle='--', alpha=0.8)
    
    plt.title('Forward Stepwise Regression: Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Target Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out_dir = 'outputs/figures'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'forward_stepwise_fit.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n>> 拟合图已保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    data_file = 'data/processed/dataset_ready.xlsx'
    
    if os.path.exists(data_file):
        print(f"加载数据集: {data_file}")
        df = pd.read_excel(data_file)
        
        X_data = df.iloc[:, 1:-1]
        y_data = df.iloc[:, -1]
        
        optimal_features = forward_stepwise_selection(X_data, y_data, criterion='aic')
        evaluate_model(X_data, y_data, optimal_features)
    else:
        print(f">> [提示] 未找到 {data_file}。请确保数据已完成预处理流水线。")
