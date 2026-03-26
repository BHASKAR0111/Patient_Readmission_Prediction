import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .risk-high   { background:#FFEBEE; border-left:5px solid #EF5350; padding:16px; border-radius:8px; }
    .risk-low    { background:#E8F5E9; border-left:5px solid #4CAF50; padding:16px; border-radius:8px; }
    .metric-card { background:white; padding:20px; border-radius:10px;
                   box-shadow:0 2px 8px rgba(0,0,0,0.08); text-align:center; }
    .section-header { font-size:18px; font-weight:600; color:#37474F;
                      border-bottom:2px solid #B2EBF2; padding-bottom:6px; margin-bottom:16px; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('data/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('data/feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except FileNotFoundError:
        return None, None

model, feature_names = load_model()

# ── Header ───────────────────────────────────────────────────
st.title("🏥 Patient Readmission Risk Predictor")
st.markdown("*Predicts probability of 30-day hospital readmission using Machine Learning*")
st.divider()

# ── Sidebar: Patient Info Input ──────────────────────────────
st.sidebar.header("👤 Patient Information")
st.sidebar.markdown("Fill in patient details to get risk prediction.")

age = st.sidebar.slider("Age", 10, 100, 65, step=5)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 4)
num_medications = st.sidebar.slider("Number of Medications", 1, 30, 12)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 1, 100, 45)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 7)
number_inpatient = st.sidebar.slider("Prior Inpatient Visits", 0, 10, 1)
number_emergency = st.sidebar.slider("Prior Emergency Visits", 0, 10, 0)
number_outpatient = st.sidebar.slider("Prior Outpatient Visits", 0, 10, 0)

st.sidebar.markdown("---")
st.sidebar.subheader("🩺 Clinical Info")
insulin = st.sidebar.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
a1c = st.sidebar.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
max_glu = st.sidebar.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
diabetes_med = st.sidebar.selectbox("On Diabetes Medication?", ["Yes", "No"])
change = st.sidebar.selectbox("Medication Change?", ["No Change", "Changed"])

predict_btn = st.sidebar.button("🔍 Predict Risk", type="primary", use_container_width=True)

# ── Main Area ────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="section-header">📋 Patient Summary</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Age", f"{age} yrs")
        st.metric("Hospital Stay", f"{time_in_hospital} days")
    with c2:
        st.metric("Medications", num_medications)
        st.metric("Lab Procedures", num_lab_procedures)
    with c3:
        st.metric("Diagnoses", number_diagnoses)
        st.metric("Prior Inpatient", number_inpatient)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)

    # Static feature importance chart (demo)
    features_demo = ['num_lab_procedures', 'time_in_hospital', 'num_medications',
                     'number_diagnoses', 'number_inpatient', 'age',
                     'total_visits', 'medication_per_day', 'A1Cresult', 'insulin']
    importances_demo = [0.18, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.05, 0.04]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors_bar = ['#EF5350' if f in ['number_inpatient','time_in_hospital'] else '#26A69A'
                  for f in features_demo]
    ax.barh(features_demo, importances_demo, color=colors_bar, alpha=0.85)
    ax.set_xlabel('Importance Score')
    ax.set_title('Top Features for Readmission Prediction', fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown('<div class="section-header">🎯 Risk Prediction</div>', unsafe_allow_html=True)

    if predict_btn:
        if model is None:
            st.warning("⚠️ Model not found. Please run notebooks first to train the model.")
            st.info("**Steps:**\n1. Download dataset from Kaggle\n2. Place in `data/` folder\n3. Run notebooks 01 → 02 → 03\n4. Come back here!")
        else:
            # Build input vector
            dose_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2}
            a1c_map  = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
            glu_map  = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}

            total_visits = number_outpatient + number_emergency + number_inpatient
            med_per_day  = num_medications / max(time_in_hospital, 1)

            input_data = {
                'race': 0, 'gender': 1 if gender == 'Male' else 0,
                'age': age, 'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': 1, 'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'number_diagnoses': number_diagnoses,
                'max_glu_serum': glu_map[max_glu],
                'A1Cresult': a1c_map[a1c],
                'metformin': 1, 'repaglinide': 0, 'nateglinide': 0,
                'chlorpropamide': 0, 'glimepiride': 0, 'acetohexamide': 0,
                'glipizide': 0, 'glyburide': 0, 'tolbutamide': 0,
                'pioglitazone': 0, 'rosiglitazone': 0, 'acarbose': 0,
                'miglitol': 0, 'troglitazone': 0, 'tolazamide': 0,
                'examide': 0, 'citoglipton': 0,
                'insulin': dose_map[insulin],
                'glyburide-metformin': 0, 'glipizide-metformin': 0,
                'glimepiride-pioglitazone': 0, 'metformin-rosiglitazone': 0,
                'metformin-pioglitazone': 0,
                'change': 1 if change == 'Changed' else 0,
                'diabetesMed': 1 if diabetes_med == 'Yes' else 0,
                'total_visits': total_visits,
                'medication_per_day': med_per_day
            }

            # Align with feature names
            input_df = pd.DataFrame([input_data])
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]

            prob = model.predict_proba(input_df)[0][1]
            risk_pct = round(prob * 100, 1)

            # Risk gauge
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            color = '#EF5350' if risk_pct > 50 else '#FFA726' if risk_pct > 25 else '#4CAF50'
            ax2.barh(['Risk'], [risk_pct], color=color, height=0.4, alpha=0.85)
            ax2.barh(['Risk'], [100 - risk_pct], left=risk_pct,
                     color='#ECEFF1', height=0.4, alpha=0.5)
            ax2.set_xlim(0, 100)
            ax2.set_xlabel('Readmission Probability (%)')
            ax2.axvline(50, color='gray', linestyle='--', alpha=0.5, lw=1)
            ax2.set_title(f'Readmission Risk: {risk_pct}%', fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # Risk message
            if risk_pct > 50:
                st.markdown(f"""
                <div class="risk-high">
                <h3 style="color:#C62828; margin:0">🔴 High Risk — {risk_pct}%</h3>
                <p style="margin:8px 0 0">This patient has a high probability of readmission within 30 days.<br>
                <b>Recommended:</b> Discharge planning, follow-up appointment, medication review.</p>
                </div>""", unsafe_allow_html=True)
            elif risk_pct > 25:
                st.markdown(f"""
                <div style="background:#FFF3E0; border-left:5px solid #FFA726; padding:16px; border-radius:8px;">
                <h3 style="color:#E65100; margin:0">🟡 Moderate Risk — {risk_pct}%</h3>
                <p style="margin:8px 0 0">Monitor this patient closely.<br>
                <b>Recommended:</b> Schedule 2-week follow-up, patient education.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                <h3 style="color:#1B5E20; margin:0">🟢 Low Risk — {risk_pct}%</h3>
                <p style="margin:8px 0 0">Patient has low readmission risk.<br>
                <b>Recommended:</b> Standard discharge protocol.</p>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Fill in patient details on the left and click **Predict Risk**")

        # Stats overview cards
        st.markdown("---")
        st.markdown("**Dataset Overview (Diabetes 130-US)**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown('<div class="metric-card"><h2>101K+</h2><p>Patient Records</p></div>',
                        unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="metric-card"><h2>11.2%</h2><p>Readmission Rate</p></div>',
                        unsafe_allow_html=True)
        with col_c:
            st.markdown('<div class="metric-card"><h2>~0.68</h2><p>Model ROC-AUC</p></div>',
                        unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown(
    "*Built by **Bhaskar Baluni** | MCA — Graphic Era Hill University | "
    "[GitHub](https://github.com/BHASKAR0111) | [Portfolio](https://bhaskar0111.github.io/Bhaskar-Portfolio/)*",
    unsafe_allow_html=False
)
