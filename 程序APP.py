import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap  # 导入SHAP库
import matplotlib.pyplot as plt

# 加载预训练的XGBoost模型
model = joblib.load('vote排名前6.pkl')

# 更新后的特征范围定义
feature_ranges = {
    "Sex": {"type": "categorical", "options": [0, 1]},
    'Long-standing illness or disability': {"type": "categorical", "options": [0, 1]},
    "Age": {"type": "numerical"},
    'Number of non-cancer illnesses': {"type": "numerical"},
    'Number of medications taken': {"type": "numerical"},
    "Systolic Blood Pressure": {"type": "numerical"},
    'Cholesterol ratio': {"type": "numerical"},
    "Plasma GDF15": {"type": "numerical"},
    "Plasma MMP12": {"type": "numerical"},
    "Plasma NTproBNP": {"type": "numerical"},
    "Plasma AGER": {"type": "numerical"},
    "Plasma PRSS8": {"type": "numerical"},
    "Plasma PSPN": {"type": "numerical"},
    "Plasma WFDC2": {"type": "numerical"},
    "Plasma LPA": {"type": "numerical"},
    "Plasma CXCL17": {"type": "numerical"},
    "Plasma GAST": {"type": "numerical"},
    "Plasma RGMA": {"type": "numerical"},
    "Plasma EPHA4": {"type": "numerical"},
}

# Streamlit界面标题
st.title("10-Year MACE Risk Prediction")

# 创建两个列，显示输入项
col1, col2 = st.columns(2)

feature_values = []

# 通过 feature_ranges 保持顺序
for i, (feature, properties) in enumerate(feature_ranges.items()):
    if properties["type"] == "numerical":
        # 数值型输入框
        if i % 2 == 0:
            with col1:
                value = st.number_input(
                    label=f"{feature}",
                    value=0.0,  # 默认值为0
                    key=f"{feature}_input"
                )
        else:
            with col2:
                value = st.number_input(
                    label=f"{feature}",
                    value=0.0,  # 默认值
                    key=f"{feature}_input"
                )
    elif properties["type"] == "categorical":
        if feature == "Sex":
            with col1:  # 将"Sex"放在第一个列中
                value = st.radio(
                    label="Sex",
                    options=[0, 1],  # 0 = Female, 1 = Male
                    format_func=lambda x: "Female" if x == 0 else "Male",
                    key=f"{feature}_input"
                )
        elif feature == 'Long-standing illness or disability':
            with col2:  # 将"Long-standing illness or disability"放在第二个列中
                value = st.radio(
                    label="Long-standing illness or disability",
                    options=[0, 1],  # 0 = No, 1 = Yes
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key=f"{feature}_input"
                )
    feature_values.append(value)

# 将特征值转换为模型输入格式
features = np.array([feature_values])

# 预测与SHAP可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

# 获取 MACE 类的概率，确保获取的是 "MACE" 类别的概率
    mace_probability = predicted_proba[1] * 100  # 第二列是 MACE 类别的概率
    # 显示预测结果，使用Matplotlib渲染指定字体
    text = f"Predicted probability of MACE in the next 10 years: {mace_probability:.2f}%."
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.text(
        0.5, 0.1, text,
        fontsize=18,
        ha='center', va='center',
        fontname='Times New Roman',  # 使用Times New Roman字体
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)  # Adjust margins tightly
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=1200)
    st.image("prediction_text.png")

# 计算SHAP值
explainer = shap.TreeExplainer(model)
X_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
shap_values = explainer.shap_values(X_df)

# 只处理1个样本
shap_values_single = shap_values[0]
abs_shap = np.abs(shap_values_single)
top_10_indices = abs_shap.argsort()[-10:][::-1]  # 绝对值最大的前10

# 取前10特征的值、名字和SHAP值
shap_values_top_10 = shap_values_single[top_10_indices]
feature_names_top_10 = X_df.columns[top_10_indices]
data_top_10 = X_df.iloc[0, top_10_indices]

# 瀑布图绘制
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 7))
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value, 
    shap_values_top_10, 
    feature_names=feature_names_top_10, 
    feature_values=data_top_10
)
plt.tight_layout()
plt.savefig("shap_waterfall_top10.png", dpi=400, bbox_inches='tight')
plt.close()
st.image("shap_waterfall_top10.png")

