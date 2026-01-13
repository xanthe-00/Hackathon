import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta
import joblib  # 用于加载 .pkl 文件
import os
import time
import httpx

# ==========================================
# 1. 核心模型加载逻辑
# ==========================================
MODEL_PATH = 'wind_turbine_predictor.pkl'

@st.cache_resource  # 使用缓存避免重复加载模型，提高性能
def load_my_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            return None
    else:
        st.error(f"未找到模型文件: {MODEL_PATH}")
        return None

model = load_my_model()

# ==========================================
# 2. AI 配置 (集成你的 API Key)
# ==========================================
GEMINI_API_KEY = "AIzaSyDIT8tm4lDizk3gmJhkP9MOmxIw6dXw29I"
# 系统代理（推荐，兼容性最好）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 配置 genai
genai.configure(api_key=GEMINI_API_KEY)

# 封装带重试的 Gemini 调用函数（保留原有逻辑）
def call_gemini_safe(prompt, max_retries=3):
    """带重试机制的 Gemini 调用，失败时抛出明确错误"""
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    for retry in range(max_retries):
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(timeout=20)
            )
            return response.text
        except Exception as e:
            if retry == max_retries - 1:
                raise Exception(f"Gemini 调用失败（重试{max_retries}次）：{str(e)}")
            st.warning(f"第{retry+1}次调用失败，重试中...")
            time.sleep(1)  # 增加重试间隔，避免高频请求被限制
            continue

# ==========================================
# 3. 页面 UI 布局
# ==========================================
st.set_page_config(page_title="WindWise 4D | 建模助手", layout="wide")

st.title("🌬️ WindWise AI: 海上垂直轴风机功率预测平台")
st.markdown("---")

# 侧边栏：四维建造参数输入
with st.sidebar:
    st.header("🏗️ 四维建造参数配置")
    rotor_diameter = st.sidebar.slider("风轮直径 (米)", min_value=31.0, max_value=35.0, value=33.0, step=0.05)
    rotor_height = st.sidebar.slider("风轮高度 (米)", min_value=1.0, max_value=1.5, value=1.25, step=0.05)
    Tip_Speed_Ratio = st.sidebar.select_slider("叶尖速比", options=[3, 4, 5, 6], value=3)
    Solidity = st.sidebar.slider("密实度", min_value=0.06, max_value=0.12, value=0.09, step=0.01)
    
    st.divider()
    run_btn = st.button("🚀 运行模型预测", type="primary")
    
    if model:
        st.success("✅ 模型已成功加载")
    else:
        st.error("❌ 模型未加载")

# 主展示区
col1, col2 = st.columns([2, 1])

if run_btn and model:
    # --- 模型推理逻辑 ---
    with st.spinner("正在调用 .pkl 模型生成 4D 映射曲线..."):
        try:
            # 1. 准备输入数据 (形状通常为 [1, 4])
            input_features = np.array([[rotor_diameter, rotor_height, Tip_Speed_Ratio, Solidity]])
            
            # 2. 调用模型预测
            # 注意：如果你的模型输出是 24 小时的功率，prediction 将是一个数组
            prediction = model.predict(input_features)
            
            # 3. 后处理成时间序列数据
            # 假设模型输出是 24 个点，如果是 1 个点，我们需要模拟时间轴
            if prediction.ndim == 2 and prediction.shape[1] > 1:
                y_values = prediction[0] # 取出第一行作为曲线
            else:
                # 如果模型只预测一个值，我们模拟一个基于该值的波动曲线（演示用）
                base_val = prediction[0] if hasattr(prediction, "__len__") else prediction
                y_values = base_val * (1 + 0.1 * np.sin(np.linspace(0, 2*np.pi, 24)))
            
            times = pd.date_range(start="2024-01-01 00:00", periods=len(y_values), freq='H')
            result_df = pd.DataFrame({"时间": times, "预测功率 (MW)": y_values})

            # --- 可视化 ---
            with col1:
                st.subheader("📈 预测时间功率曲线")
                fig = px.line(result_df, x="时间", y="预测功率 (MW)", 
                             template="plotly_dark",
                             color_discrete_sequence=['#00D4FF'])
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                # 统计数据
                st.metric("预测平均功率", f"{np.mean(y_values):.2f} MW")
                st.metric("预测峰值功率", f"{np.max(y_values):.2f} MW")

            # --- AI 分析 ---
            with col2:
                st.subheader("🤖 Gemini 深度分析")
                analysis_prompt = f"""
                我是风电设计师，我的 4D 建造参数输入为: {rotor_diameter}, {rotor_height}, {Tip_Speed_Ratio}, {Solidity}。
                模型预测出的平均功率为 {np.mean(y_values):.2f} MW。
                请结合这些数据，利用你的风电专业知识：
                1. 评价该建造参数组合下的功率表现是否符合预期？
                2. 针对这组 4D 参数，指出可能的优化方向。
                3. 分析在当前配置下，随着时间推移，该机组的可靠性趋势。
                """
                
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                response = gemini_model.generate_content(analysis_prompt)
                st.markdown(response.text)

        except Exception as e:
            st.error(f"推理过程中出错: {e}")
            st.info("提示：请检查模型期望的输入维度和 .pkl 兼容性。")

elif not run_btn:
    with col1:
        st.info("请在左侧调整 4D 建造参数并点击运行。")
        # 修复：替换已弃用的 use_column_width，改用 width="100%" 实现列宽适配
        st.image("https://static.cnbetacdn.com/article/2022/09/82d4c8e933a9189.png", width="stretch")

# ==========================================
# 4. 交互对话框 (底栏)
# ==========================================
st.divider()
st.subheader("💬 设计师自由追问")
if user_q := st.chat_input("关于这个模型预测结果，你还有什么想问 AI 的吗？"):
    with st.chat_message("user"):
        st.write(user_q)
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            res = genai.GenerativeModel('gemini-1.5-flash').generate_content(user_q)
            st.write(res.text)