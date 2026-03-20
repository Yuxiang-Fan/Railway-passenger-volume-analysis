# Railway Passenger Volume Factor Analysis
# 铁路客运量影响因素统计分析

This repository contains the source code and documentation for a study on the influencing factors of national railway passenger volume, originally developed for the 10th National Undergraduate Statistical Modeling Competition. The project explores the relationship between macroeconomic indicators and railway demand using a variety of statistical and machine learning approaches.

本项目包含针对全国铁路客运量影响因素研究的源代码及文档，最初为第十届全国大学生统计建模大赛开发。该项目通过多种统计学和机器学习方法，探讨了宏观经济指标与铁路需求之间的关系。

---

## English Version

### Project Overview
The study aims to identify key drivers of railway passenger volume in China by analyzing data from 2008 to 2021. Given the impact of digital transformation and infrastructure investment, understanding these variables is essential for future transportation planning.

### Methodology
1.  **Data Preprocessing**: Newton interpolation was utilized to address missing data points in historical records. Features underwent de-dimensionalization and logarithmic transformation to mitigate heteroscedasticity and stabilize variance.
2.  **Feature Engineering**: 
    * **Grey Relational Analysis (GRA)**: Used to rank potential indicators such as GDP, urban wages, and railway mileage based on their correlation with passenger volume.
    * **Statistical Testing**: ADF tests were conducted to check for stationarity, requiring up to fourth-order differencing for certain variables. Granger causality tests were explored but faced limitations due to the small sample size.
3.  **Modeling**: The project compares three regression techniques: Support Vector Regression (SVR), Ridge Regression, and Stepwise Regression.

### Key Findings
* Stepwise regression achieved the highest accuracy with a Mean Squared Error (MSE) of approximately 0.000183.
* Key significant factors identified include average wages of urban employees and railway passenger turnover.
* The analysis suggests that increasing high-end service capacity (e.g., soft sleepers) correlates positively with passenger volume.

### Repository Structure
* `src/`: Modular Python scripts for interpolation, transformation, and modeling.
* `data/`: Directory for raw and processed datasets.
* `outputs/`: Visualizations of model fits and statistical tables.
* `main.py`: A unified script to execute the entire analysis pipeline.

### Limitations
The study acknowledges several constraints, including a relatively small sample size, reliance on data ending in 2021, and a focus on linear modeling techniques.

---

## 中文版

### 项目简介
本项目旨在通过分析 2008 年至 2021 年的数据，识别影响中国铁路客运量的核心驱动因素。考虑到数字化转型和基础设施投资的影响，理解这些变量对未来的交通规划至关重要。

### 研究方法
1.  **数据预处理**：采用牛顿插值法处理历史记录中的缺失值。对特征进行了去量纲化和对数变换，以减弱异方差性并增强数据稳定性。
2.  **特征工程**：
    **灰色关联分析 (GRA)**：根据指标（如 GDP、城镇工资、铁路里程）与客运量的关联程度进行排序。
    **统计检验**：通过 ADF 检验验证平稳性，部分变量需经过四阶差分方达平稳。对格兰杰因果检验进行了探索，但受限于样本量较小，存在过拟合局限。
3.  **模型建立**：项目对比了三种回归技术：支持向量回归 (SVR)、岭回归和逐步回归。

### 核心结论
* 逐步回归模型表现最佳，均方误差 (MSE) 约为 0.000183。
* 城镇单位就业人员平均工资和铁路旅客周转量被确定为显著影响因素。
* 分析表明，增加高品质服务（如软卧车）的供给与客运量提升呈正相关。

### 仓库结构
* `src/`：包含插值、变换和建模的模块化 Python 脚本。
* `data/`：存放原始数据和处理后数据的目录。
* `outputs/`：存放模型拟合可视化图表和统计表格。
* `main.py`：用于执行整个分析流程的统一调度脚本。

### 研究不足
本研究承认存在一定的局限性，包括样本量较小、数据更新截止至 2021 年以及主要采用线性拟合方法等。

---
