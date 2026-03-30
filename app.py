import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer  # noqa: F401

st.set_page_config(
    page_title="sICAS Recurrence Prediction Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    header[data-testid="stHeader"] {
        background-color: #0e1117 !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, span {
        color: white !important;
    }
    [data-testid="stNumberInput"] button {
        display: none !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #4f4f4f !important;
        border-radius: 5px !important;
        z-index: 1000000 !important;
        display: block !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        background-color: #1565c0 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] button {
        color: white !important;
    }
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
    .stNumberInput label, .stSlider label {
        color: white !important;
    }
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .risk-card-high {
        background: linear-gradient(135deg, #b71c1c 0%, #d32f2f 100%);
        padding: 20px;
        border-radius: 12px;
        color: white !important;
        box-shadow: 0 4px 15px rgba(183, 28, 28, 0.4);
        border: 1px solid #ffcdd2;
        margin-bottom: 20px;
    }
    .risk-card-high h2 {
        color: white !important;
        margin: 0;
        font-weight: 800;
        font-size: 24px;
    }
    .risk-card-high p {
        color: #ffcdd2 !important;
        margin-top: 5px;
        font-size: 16px;
        font-weight: 500;
    }
    .risk-card-high .rec {
        border-top: 1px solid rgba(255,255,255,0.3);
        margin-top: 15px;
        padding-top: 10px;
        font-style: italic;
        font-size: 15px;
        color: #ffebee !important;
        line-height: 1.4;
    }
    .risk-card-low {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        padding: 20px;
        border-radius: 12px;
        color: white !important;
        box-shadow: 0 4px 15px rgba(27, 94, 32, 0.4);
        border: 1px solid #c8e6c9;
        margin-bottom: 20px;
    }
    .risk-card-low h2 {
        color: white !important;
        margin: 0;
        font-weight: 800;
        font-size: 24px;
    }
    .risk-card-low p {
        color: #c8e6c9 !important;
        margin-top: 5px;
        font-size: 16px;
        font-weight: 500;
    }
    .risk-card-low .rec {
        border-top: 1px solid rgba(255,255,255,0.3);
        margin-top: 15px;
        padding-top: 10px;
        font-style: italic;
        font-size: 15px;
        color: #e8f5e9 !important;
        line-height: 1.4;
    }
    .driver-card {
        background-color: #262730;
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid #4f4f4f;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .driver-title {
        font-weight: 700;
        font-size: 16px;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .driver-val {
        font-size: 14px;
        color: #b0bec5 !important;
        margin-top: 2px;
    }
    .driver-effect {
        font-size: 13px;
        font-weight: bold;
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px dashed #555;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1565c0 0%, #0d47a1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        height: 55px;
        font-size: 20px;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(13, 71, 161, 0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        predictor = joblib.load("sICAS_Calibrated_Model.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load prediction model: {type(e).__name__}: {e}")
        return None, None, None

    try:
        explainer_model = joblib.load("sICAS_RF_Surrogate.pkl")
        explainer = shap.TreeExplainer(explainer_model)
    except Exception as e:
        st.warning(f"⚠️ Explanation module unavailable: {type(e).__name__}: {e}")
        return predictor, None, None

    return predictor, explainer_model, explainer


predictor, explainer_model, explainer = load_models()

st.sidebar.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=80)
st.sidebar.title("Patient Parameters")
st.sidebar.markdown("---")


def user_input_features():
    with st.sidebar.expander("📊 Imaging (CTP/CTA)", expanded=True):
        st.caption("Hemodynamic and anatomical features")
        rcbf34 = st.number_input(
            "rCBF < 34% Volume (mL)",
            min_value=0.0, max_value=400.0, value=0.0, step=1.0,
            help="Volume of severely hypoperfused tissue."
        )
        tmax6 = st.number_input(
            "Tmax > 6 s Volume (mL)",
            min_value=0.0, max_value=600.0, value=0.0, step=1.0,
            help="Volume of tissue with delayed perfusion."
        )
        stenosis = st.slider(
            "Stenosis Severity (%)",
            0, 100, 50,
            help="Degree of intracranial artery stenosis."
        )

    with st.sidebar.expander("🩸 Biomarkers & Labs", expanded=True):
        st.caption("Laboratory variables")
        egfr = st.number_input(
            "eGFR (mL/min/1.73 m²)",
            min_value=0.0, max_value=150.0, value=90.0, step=1.0,
            help="Estimated glomerular filtration rate."
        )
        ldl = st.number_input(
            "LDL-C (mmol/L)",
            min_value=0.5, max_value=20.0, value=2.5, step=0.1,
            help="Low-density lipoprotein cholesterol."
        )
        glucose = st.number_input(
            "Blood Glucose (mmol/L)",
            min_value=1.0, max_value=40.0, value=5.5, step=0.1,
            help="Blood glucose concentration."
        )

    with st.sidebar.expander("👤 Demographics", expanded=True):
        age = st.slider("Age (years)", 18, 100, 60)
        sbp = st.number_input(
            "Systolic BP (mmHg)",
            min_value=60, max_value=240, value=130, step=1
        )
        nihss = st.slider("NIHSS Score (Baseline)", 0, 42, 2)

    data = {
        "age": age,
        "SBP": sbp,
        "NIHSS_In": nihss,
        "eGFR": egfr,
        "Glucose": glucose,
        "LDL": ldl,
        "Stenosis_Pct": stenosis,
        "tmax6": tmax6,
        "rcbf34": rcbf34,
    }

    df_input = pd.DataFrame(data, index=[0])
    df_input = df_input[
        ["age", "SBP", "NIHSS_In", "eGFR", "Glucose", "LDL", "Stenosis_Pct", "tmax6", "rcbf34"]
    ]
    return df_input


input_df = user_input_features()

CLINICAL_THRESHOLD = 0.14

DISPLAY_NAMES = {
    "age": "Age",
    "SBP": "Systolic BP",
    "NIHSS_In": "NIHSS Score",
    "eGFR": "eGFR",
    "Glucose": "Blood Glucose",
    "LDL": "LDL-C",
    "Stenosis_Pct": "Stenosis Severity",
    "tmax6": "Tmax > 6 s Volume",
    "rcbf34": "rCBF < 34% Volume",
}

DISPLAY_UNITS = {
    "age": "years",
    "SBP": "mmHg",
    "NIHSS_In": "",
    "eGFR": "mL/min/1.73 m²",
    "Glucose": "mmol/L",
    "LDL": "mmol/L",
    "Stenosis_Pct": "%",
    "tmax6": "mL",
    "rcbf34": "mL",
}

WATERFALL_NAMES = {
    "age": "Age",
    "SBP": "SBP",
    "NIHSS_In": "NIHSS",
    "eGFR": "eGFR",
    "Glucose": "Glucose",
    "LDL": "LDL-C",
    "Stenosis_Pct": "Stenosis",
    "tmax6": "Tmax > 6 s",
    "rcbf34": "rCBF < 34%",
}


def format_value(val, unit=""):
    if isinstance(val, (int, np.integer)):
        text = f"{val:d}"
    elif isinstance(val, (float, np.floating)):
        text = f"{val:.1f}"
    else:
        text = str(val)
    return f"{text} {unit}".strip()


def extract_base_value(base_values):
    arr = np.asarray(base_values)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if arr.size == 1:
            return float(arr[0])
        return float(arr[-1])
    return float(arr[0, -1])


def build_shap_explanation(shap_values, input_frame):
    feature_names = [WATERFALL_NAMES.get(col, col) for col in input_frame.columns]
    data_row = input_frame.iloc[0].to_numpy(dtype=float)

    if len(shap_values.values.shape) == 3:
        values = np.asarray(shap_values.values[0, :, 1], dtype=float)
        base_value = extract_base_value(shap_values.base_values[0])
    else:
        values = np.asarray(shap_values.values[0], dtype=float)
        base_value = extract_base_value(shap_values.base_values[0])

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=data_row,
        feature_names=feature_names
    )


def render_waterfall_plot(shap_obj, max_display=9):
    plt.close("all")
    plt.style.use("default")
    plt.figure(figsize=(10.5, 6.2))

    shap.plots.waterfall(
        shap_obj,
        max_display=min(max_display, len(shap_obj.values)),
        show=False
    )

    fig = plt.gcf()
    fig.patch.set_facecolor("#0e1117")

    for ax in fig.axes:
        ax.set_facecolor("#0e1117")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("white")
            label.set_fontsize(11)

        for text in ax.texts:
            text.set_color("white")
            text.set_fontsize(11)

        for spine in ax.spines.values():
            spine.set_color("white")

    plt.subplots_adjust(left=0.38, right=0.98, top=0.94, bottom=0.14)
    return fig


st.title("🧠 sICAS Recurrence Prediction Tool")

st.markdown("""
<div style="font-size: 17px; color: #d0d7de; margin-bottom: 18px;">
    This tool predicts the <strong>1-year risk of the composite endpoint</strong> of
    <strong>target-vessel territory ischemic stroke and neurological death</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #262730; padding: 10px 15px; border-radius: 5px; border-left: 4px solid #1565c0; margin-bottom: 20px;">
    <strong>Model Architecture:</strong>
    <span style="color: #4fc3f7;">Voting Ensemble (LR + SVM)</span> (for risk prediction) +
    <span style="color: #81c784;">Post hoc RF surrogate</span> (for interpretability)
</div>
""", unsafe_allow_html=True)

st.warning("""
**⚠️ Target Population & Exclusions:**  
This tool is intended for **symptomatic ICAS patients receiving medical management**.  
It is **not applicable** to patients who underwent acute reperfusion therapy (**IV thrombolysis or mechanical thrombectomy**) during the index event, as this model was **not trained or validated** in that population.
""")

if st.button("🚀 Run Analysis"):
    if predictor is not None:
        try:
            prob = predictor.predict_proba(input_df)[0][1]
        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__}: {e}")
            st.stop()

        st.subheader("1. Clinical Risk Prediction for the 1-Year Composite Endpoint")
        st.caption("Composite endpoint: target-vessel territory ischemic stroke or neurological death")

        col1, col2 = st.columns([3, 1])

        with col1:
            if prob >= CLINICAL_THRESHOLD:
                st.markdown(f"""
                <div class="risk-card-high">
                    <h2>⚠️ High Risk of Recurrence</h2>
                    <p>
                        Prediction Probability: <strong>{prob:.1%}</strong>
                        <span style="font-size:14px; opacity:0.8; margin-left: 10px;">
                            (Threshold: {CLINICAL_THRESHOLD:.2f})
                        </span>
                    </p>
                    <div class="rec">
                        💡 Recommendation: Consider closer follow-up and comprehensive optimization of vascular risk-factor control.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-card-low">
                    <h2>✅ Lower Risk Profile</h2>
                    <p>
                        Prediction Probability: <strong>{prob:.1%}</strong>
                        <span style="font-size:14px; opacity:0.8; margin-left: 10px;">
                            (Threshold: {CLINICAL_THRESHOLD:.2f})
                        </span>
                    </p>
                    <div class="rec">
                        💡 Recommendation: Continue standard secondary prevention and routine follow-up according to current clinical practice.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("##### Confidence")
            display_prob = min(prob / 0.5, 1.0)
            st.progress(float(display_prob))
            st.caption(f"Risk Prob: {prob:.3f}")

        if explainer_model is not None and explainer is not None:
            st.divider()
            st.subheader("2. Mechanistic Driver Analysis")

            st.markdown("""
            <div style="font-size: 14px; color: #b0bec5; margin-bottom: 5px;">
                ℹ️ This section uses a <strong>post hoc tree-based surrogate model</strong> to visualize the major factors associated with higher or lower predicted risk.
            </div>
            """, unsafe_allow_html=True)

            try:
                shap_values = explainer(input_df)
                shap_obj = build_shap_explanation(shap_values, input_df)
            except Exception as e:
                st.warning(f"SHAP explanation generation failed: {type(e).__name__}: {e}")
                shap_obj = None

            if shap_obj is not None:
                col_graph, col_text = st.columns([2, 1])

                with col_graph:
                    try:
                        fig = render_waterfall_plot(shap_obj, max_display=9)
                        st.pyplot(fig, clear_figure=True, bbox_inches="tight")
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"Waterfall plot rendering failed: {type(e).__name__}: {e}")

                with col_text:
                    st.markdown("#### Key Drivers")

                    vals = np.asarray(shap_obj.values)
                    names = list(input_df.columns)
                    top_indices = np.argsort(np.abs(vals))[::-1][:3]

                    for idx in top_indices:
                        val = float(vals[idx])
                        name = names[idx]
                        display_name = DISPLAY_NAMES.get(name, name)
                        patient_val = input_df.iloc[0, idx]
                        unit = DISPLAY_UNITS.get(name, "")
                        value_text = format_value(patient_val, unit)

                        if val > 0:
                            icon = "🔺"
                            color_code = "#ff5252"
                            effect_text = "Associated with higher risk"
                            border_color = "#d32f2f"
                        else:
                            icon = "🛡️"
                            color_code = "#69f0ae"
                            effect_text = "Associated with lower risk"
                            border_color = "#2e7d32"

                        st.markdown(f"""
                        <div class="driver-card" style="border-left: 4px solid {border_color};">
                            <div class="driver-title">
                                <span>{display_name}</span> <span>{icon}</span>
                            </div>
                            <div class="driver-val">
                                Value: <b>{value_text}</b>
                            </div>
                            <div class="driver-effect" style="color: {color_code} !important;">
                                {effect_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.error("Model file missing or failed to load.")

st.divider()
st.markdown("""
### ⚠️ Disclaimer & Usage Guide
**1. Research Use Only (RUO):** This tool is intended for **academic research and educational purposes only**. It has not been cleared or approved by regulatory agencies for clinical decision-making.  
**2. Predicted Outcome:** The model estimates the **1-year risk of the composite endpoint of target-vessel territory ischemic stroke and neurological death**.  
**3. Target Population:** This tool is validated **only** for sICAS patients receiving medical management. **Do not** apply it to patients treated with acute reperfusion therapy (IVT/EVT).  
**4. Local Validation Required:** The model was developed using a single-center cohort. **External validation and recalibration** are required before any clinical deployment.  
**5. No Medical Advice:** The output should **not** replace clinical judgment. Treatment decisions must be made by qualified healthcare professionals after comprehensive evaluation.
""")
