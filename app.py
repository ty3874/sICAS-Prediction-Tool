import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面配置与全局设置
# ==========================================
st.set_page_config(
    page_title="sICAS Recurrence Risk Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 核心修复 1：Matplotlib 强制深色底 ---
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "savefig.facecolor": "#0e1117",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "font.size": 12,
    "font.family": "sans-serif"
})

# --- 核心修复 2：CSS 样式调整 ---
st.markdown("""
<style>
    /* 1. 强制整个网页背景为深色 */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }

    /* 2. 强制侧边栏背景为深灰 */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }

    /* 3. 强制基础文本为白色 */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, span {
        color: white !important;
    }

    /* === 【关键修复】侧边栏折叠/展开按钮可见性 === */
    /* 1. 针对收起后出现在左上角的“>”箭头 */
    [data-testid="collapsedControl"] {
        color: white !important; /* 强制箭头变白 */
        background-color: #262730 !important; /* 给它加个深灰底色 */
        border: 1px solid #4f4f4f; /* 加个边框 */
        border-radius: 5px; /* 圆角 */
        top: 1rem !important; /* 稍微调整位置 */
        left: 1rem !important;
    }
    
    /* 2. 鼠标放上去变亮 */
    [data-testid="collapsedControl"]:hover {
        background-color: #4fc3f7 !important; /* 悬停变蓝 */
        color: black !important;
    }

    /* 3. 针对展开状态下的“X”关闭按钮（在侧边栏右上角） */
    [data-testid="stSidebar"] button {
        color: white !important;
    }

    /* === 侧边栏折叠框 (Expander) 样式 === */
    [data-testid="stSidebar"] details > summary {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #4f4f4f;
        border-radius: 5px;
    }
    
    [data-testid="stSidebar"] details > summary:hover {
        background-color: #383940 !important;
        color: #4fc3f7 !important;
    }

    [data-testid="stSidebar"] details {
        background-color: #262730 !important;
        border-color: #262730 !important;
    }

    /* 4. 修复输入框标签颜色 */
    .stNumberInput label, .stSlider label {
        color: white !important;
    }

    /* 全局字体优化 */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* 高危卡片 */
    .risk-card-high { 
        background: linear-gradient(135deg, #b71c1c 0%, #d32f2f 100%);
        padding: 20px; 
        border-radius: 12px; 
        color: white !important;
        box-shadow: 0 4px 15px rgba(183, 28, 28, 0.4);
        border: 1px solid #ffcdd2;
        margin-bottom: 20px;
    }
    .risk-card-high h2 { color: white !important; margin: 0; font-weight: 800; font-size: 24px; }
    .risk-card-high p { color: #ffcdd2 !important; margin-top: 5px; font-size: 16px; font-weight: 500; }
    .risk-card-high .rec { border-top: 1px solid rgba(255,255,255,0.3); margin-top:15px; padding-top:10px; font-style: italic; font-size: 15px; color: #ffebee !important; line-height: 1.4; }

    /* 低危卡片 */
    .risk-card-low { 
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        padding: 20px; 
        border-radius: 12px; 
        color: white !important;
        box-shadow: 0 4px 15px rgba(27, 94, 32, 0.4);
        border: 1px solid #c8e6c9;
        margin-bottom: 20px;
    }
    .risk-card-low h2 { color: white !important; margin: 0; font-weight: 800; font-size: 24px; }
    .risk-card-low p { color: #c8e6c9 !important; margin-top: 5px; font-size: 16px; font-weight: 500; }
    .risk-card-low .rec { border-top: 1px solid rgba(255,255,255,0.3); margin-top:15px; padding-top:10px; font-style: italic; font-size: 15px; color: #e8f5e9 !important; line-height: 1.4; }

    /* 关键驱动因素卡片 */
    .driver-card {
        background-color: #262730;
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid #4f4f4f;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .driver-title { font-weight: 700; font-size: 16px; color: #ffffff !important; display: flex; align-items: center; justify-content: space-between; }
    .driver-val { font-size: 14px; color: #b0bec5 !important; margin-top: 2px; }
    .driver-effect { font-size: 13px; font-weight: bold; margin-top: 8px; padding-top: 6px; border-top: 1px dashed #555; }

    /* 按钮美化 */
    .stButton>button {
        background: linear-gradient(90deg, #1565c0 0%, #0d47a1 100%);
        color: white; border: none; border-radius: 8px; height: 55px; font-size: 20px; font-weight: bold; transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(13, 71, 161, 0.4); }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. 核心：加载双模型
# ==========================================
@st.cache_resource
def load_models():
    try:
        predictor = joblib.load('sICAS_Recurrence_Model.pkl')
    except:
        st.error("❌ Critical Error: 'sICAS_Recurrence_Model.pkl' not found. Please upload the model file.")
        return None, None

    try:
        explainer_model = joblib.load('sICAS_RF_Surrogate.pkl')
    except:
        st.warning("⚠️ Warning: 'sICAS_RF_Surrogate.pkl' not found. Explanations module will be disabled.")
        return predictor, None

    return predictor, explainer_model


predictor, explainer_model = load_models()

# ==========================================
# 3. 侧边栏：临床数据输入
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=80)
st.sidebar.title("Patient Parameters")
st.sidebar.markdown("---")


def user_input_features():
    # 影像学 (默认展开)
    with st.sidebar.expander("📊 Imaging (CTP/Angio)", expanded=True):
        st.caption("Hemodynamic & Anatomical features")
        rcbf34 = st.number_input("rCBF < 34% Volume (ml)", min_value=0.0, max_value=400.0, value=0.0, step=1.0,
                                 help="Volume of core infarct (severely hypoperfused tissue).")
        tmax6 = st.number_input("Tmax > 6s Volume (ml)", min_value=0.0, max_value=600.0, value=0.0, step=1.0,
                                help="Volume of tissue with delayed perfusion (penumbra).")
        stenosis = st.slider("Stenosis Severity (%)", 0, 100, 50, help="Degree of intracranial artery stenosis.")

    # 生物标志物 (默认展开)
    with st.sidebar.expander("🩸 Biomarkers & Labs", expanded=True):
        st.caption("Metabolic & Inflammatory markers")
        egfr = st.number_input("eGFR (ml/min)", min_value=0.0, max_value=150.0, value=90.0, step=1.0,
                               help="Renal function. Normal > 90.")
        hscrp = st.number_input("hs-CRP (mg/L)", min_value=0.0, max_value=200.0, value=1.0, step=0.1,
                                help="Inflammatory marker. High risk if > 3.0.")
        ldl = st.number_input("LDL-C (mmol/L)", min_value=0.5, max_value=20.0, value=2.5, step=0.1,
                              help="Low-density lipoprotein cholesterol.")
        glucose = st.number_input("Blood Glucose (mmol/L)", min_value=1.0, max_value=40.0, value=5.5, step=0.1)

    # 人口学 (默认展开)
    with st.sidebar.expander("👤 Demographics", expanded=True):
        age = st.slider("Age (years)", 18, 100, 60)
        sbp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=240, value=130, step=1)
        nihss = st.slider("NIHSS Score (Baseline)", 0, 42, 2)

    data = {
        'rcbf34': rcbf34, 'Stenosis_Pct': stenosis, 'NIHSS_In': nihss,
        'eGFR': egfr, 'age': age, 'SBP': sbp,
        'tmax6': tmax6, 'Glucose': glucose, 'LDL': ldl, 'hsCRP': hscrp
    }
    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

# ==========================================
# 4. 主界面逻辑
# ==========================================
st.title("🧠 sICAS Recurrence Prediction Tool")

st.markdown("""
<div style="background-color: #262730; padding: 10px 15px; border-radius: 5px; border-left: 4px solid #1565c0; margin-bottom: 20px;">
    <strong>Model Architecture:</strong> 
    <span style="color: #4fc3f7;">Voting Ensemble</span> (for High-Performance Prediction) + 
    <span style="color: #81c784;">RF Surrogate</span> (for Mechanistic Interpretation)
</div>
""", unsafe_allow_html=True)

# 适用人群警告
st.warning("""
**⚠️ Target Population & Exclusions:**
This tool is intended for **symptomatic ICAS patients receiving medical management**. 
It is **NOT applicable** to patients who underwent acute reperfusion therapy (**IV Thrombolysis or Mechanical Thrombectomy**) during the index event, as their clinical profiles and risk patterns likely differ, and this model was **not trained or validated** on this specific population.
""")

CLINICAL_THRESHOLD = 0.289

if st.button("🚀 Run Analysis"):
    if predictor:
        # --- A. 预测模块 (Voting) ---
        prob = predictor.predict_proba(input_df)[0][1]

        st.subheader("1. Clinical Risk Prediction")

        col1, col2 = st.columns([3, 1])

        with col1:
            if prob >= CLINICAL_THRESHOLD:
                st.markdown(f"""
                <div class="risk-card-high">
                    <h2>⚠️ High Risk of Recurrence</h2>
                    <p>
                        Prediction Probability: <strong>{prob:.1%}</strong>
                        <span style="font-size:14px; opacity:0.8; margin-left: 10px;">(Threshold: {CLINICAL_THRESHOLD:.3f})</span>
                    </p>
                    <div class="rec">
                        💡 Recommendation: Suggest comprehensive vascular risk factor assessment, stricter target control, and closer clinical follow-up.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-card-low">
                    <h2>✅ Low Risk Profile</h2>
                    <p>
                        Prediction Probability: <strong>{prob:.1%}</strong>
                        <span style="font-size:14px; opacity:0.8; margin-left: 10px;">(Threshold: {CLINICAL_THRESHOLD:.3f})</span>
                    </p>
                    <div class="rec">
                        💡 Recommendation: Maintain standard secondary prevention strategies according to current guidelines.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("##### Confidence")
            st.progress(float(prob))
            st.caption(f"Risk Score: {prob:.3f}")

        # --- B. 解释模块 (RF Surrogate) ---
        if explainer_model:
            st.divider()
            st.subheader("2. Mechanistic Driver Analysis")
            st.markdown("""
            <div style="font-size: 14px; color: #b0bec5; margin-bottom: 15px;">
                ℹ️ This section uses a <strong>Surrogate Model (SHAP)</strong> to visualize the key factors driving the risk score UP (Red) or DOWN (Blue) for this specific patient.
            </div>
            """, unsafe_allow_html=True)

            explainer = shap.TreeExplainer(explainer_model)
            shap_values = explainer(input_df)

            if len(shap_values.values.shape) == 3:
                shap_obj = shap_values[0, :, 1]
            else:
                shap_obj = shap_values[0]

            col_graph, col_text = st.columns([2, 1])

            with col_graph:
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_obj, max_display=9, show=False)

                # --- 核心修复：强制修改图表颜色适配深色模式 ---
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')

                for text in ax.texts:
                    text.set_color("white")
                    text.set_fontsize(11)

                # 设置背景色为深灰
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')

                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)

            with col_text:
                st.markdown("#### Key Drivers")

                vals = shap_obj.values
                names = input_df.columns
                top_indices = np.argsort(np.abs(vals))[::-1][:3]

                for idx in top_indices:
                    val = vals[idx]
                    name = names[idx]
                    patient_val = input_df.iloc[0, idx]

                    if val > 0:
                        icon = "🔺"
                        color_code = "#ff5252"
                        effect_text = "Increases Risk"
                        border_color = "#d32f2f"
                    else:
                        icon = "🛡️"
                        color_code = "#69f0ae"
                        effect_text = "Protects / Lowers Risk"
                        border_color = "#2e7d32"

                    st.markdown(f"""
                    <div class="driver-card" style="border-left: 4px solid {border_color};">
                        <div class="driver-title">
                            <span>{name}</span> <span>{icon}</span>
                        </div>
                        <div class="driver-val">
                            Value: <b>{patient_val:.1f}</b>
                        </div>
                        <div class="driver-effect" style="color: {color_code} !important;">
                            {effect_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.error("Model file missing.")

# ==========================================
# 5. 免责声明 (Footer)
# ==========================================
st.divider()
st.markdown("""
### ⚠️ Disclaimer & Usage Guide

**1. Research Use Only (RUO):** This tool is designed for **academic research and educational purposes only**. It has not been cleared or approved by the FDA, NMPA, or other regulatory bodies for clinical diagnosis or treatment guidance.

**2. Target Population:** This tool is validated **ONLY** for sICAS patients receiving medical management. **DO NOT** use for patients post-acute reperfusion therapy (IVT/EVT).

**3. Local Validation Required:** The underlying model was trained on a specific single-center cohort. **External validation and recalibration** using your local patient data are strictly required before any consideration of clinical deployment.

**4. No Medical Advice:** The output of this tool should **not** replace professional clinical judgment. All treatment decisions must be made by qualified healthcare providers based on the comprehensive evaluation of the patient.
""")
