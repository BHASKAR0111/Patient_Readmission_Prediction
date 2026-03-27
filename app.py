import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .risk-high   { background:#FFEBEE; border-left:5px solid #EF5350; padding:16px; border-radius:8px; }
    .risk-low    { background:#E8F5E9; border-left:5px solid #4CAF50; padding:16px; border-radius:8px; }
    .metric-card { background:white; padding:20px; border-radius:10px;
                   box-shadow:0 2px 8px rgba(0,0,0,0.08); text-align:center; }
    .section-header { font-size:18px; font-weight:600; color:#37474F;
                      border-bottom:2px solid #B2EBF2; padding-bottom:6px; margin-bottom:16px; }
</style>
""", unsafe_allow_html=True)


# ── Load data from YOUR GitHub repo & train model ─────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    # ✅ Fetches directly from your GitHub repo — no pickle, no Kaggle
    DATA_URL =  "https://raw.githubusercontent.com/BHASKAR0111/Patient_Readmission_Prediction/main/data.csv"

    df = pd.read_csv(DATA_URL)
    df.replace('?', np.nan, inplace=True)

    drop_cols = ['encounter_id', 'patient_nbr', 'weight',
                 'payer_code', 'medical_specialty',
                 'diag_1', 'diag_2', 'diag_3']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.dropna(inplace=True)

    n_records = len(df)

    # Target
    df['readmitted'] = (df['readmitted'] == '<30').astype(int)
    readmit_rate = round(df['readmitted'].mean() * 100, 1)

    # Age midpoint
    age_map = {
        '[0-10)': 5,  '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65,
        '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)

    # Feature engineering
    df['total_visits']       = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    df['medication_per_day'] = df['num_medications'] / df['time_in_hospital'].replace(0, 1)

    # Encode
    df['change']      = (df['change'] == 'Ch').astype(int)
    df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)
    df['gender']      = (df['gender'] == 'Male').astype(int)

    med_cols = [
        'metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
        'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
        'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
        'examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin',
        'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone'
    ]
    dose_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2}
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].map(dose_map).fillna(0).astype(int)

    df['A1Cresult']     = df['A1Cresult'].map({'None': 0, 'Norm': 1, '>7': 2, '>8': 3}).fillna(0)
    df['max_glu_serum'] = df['max_glu_serum'].map({'None': 0, 'Norm': 1, '>200': 2, '>300': 3}).fillna(0)

    le = LabelEncoder()
    df['race'] = le.fit_transform(df['race'].astype(str))

    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_res, y_res)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = round(roc_auc_score(y_test, y_proba), 3)

    importances = pd.Series(model.feature_importances_, index=feature_names)

    return model, feature_names, importances, n_records, readmit_rate, auc


# ── Spinner while loading ─────────────────────────────────────
with st.spinner("🔄 Fetching data from GitHub & training model... (~30 sec on first load)"):
    try:
        model, feature_names, importances, n_records, readmit_rate, auc = load_and_train()
        model_ready = True
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        model_ready = False

# ── Header ────────────────────────────────────────────────────
st.title("🏥 Patient Readmission Risk Predictor")
st.markdown("*Predicts probability of 30-day hospital readmission using Machine Learning*")
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.header("👤 Patient Information")
st.sidebar.markdown("Fill in patient details to get risk prediction.")

age               = st.sidebar.slider("Age", 10, 100, 65, step=5)
gender            = st.sidebar.selectbox("Gender", ["Female", "Male"])
time_in_hospital  = st.sidebar.slider("Days in Hospital", 1, 14, 4)
num_medications   = st.sidebar.slider("Number of Medications", 1, 30, 12)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 1, 100, 45)
number_diagnoses  = st.sidebar.slider("Number of Diagnoses", 1, 16, 7)
number_inpatient  = st.sidebar.slider("Prior Inpatient Visits", 0, 10, 1)
number_emergency  = st.sidebar.slider("Prior Emergency Visits", 0, 10, 0)
number_outpatient = st.sidebar.slider("Prior Outpatient Visits", 0, 10, 0)

st.sidebar.markdown("---")
st.sidebar.subheader("🩺 Clinical Info")
insulin      = st.sidebar.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
a1c          = st.sidebar.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
max_glu      = st.sidebar.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
diabetes_med = st.sidebar.selectbox("On Diabetes Medication?", ["Yes", "No"])
change       = st.sidebar.selectbox("Medication Change?", ["No Change", "Changed"])

predict_btn = st.sidebar.button("🔍 Predict Risk", type="primary", use_container_width=True)

# ── Main layout ───────────────────────────────────────────────
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

    if model_ready:
        top10 = importances.nlargest(10).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_colors = ['#EF5350' if f in ['number_inpatient', 'time_in_hospital']
                      else '#26A69A' for f in top10.index]
        ax.barh(top10.index, top10.values, color=bar_colors, alpha=0.85)
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Features — Random Forest (Real Data)', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with col2:
    st.markdown('<div class="section-header">🎯 Risk Prediction</div>', unsafe_allow_html=True)

    if predict_btn:
        if not model_ready:
            st.error("Model not loaded. Please refresh the page.")
        else:
            dose_map_i = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2}
            a1c_map_i  = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
            glu_map_i  = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}

            total_visits = number_outpatient + number_emergency + number_inpatient
            med_per_day  = num_medications / max(time_in_hospital, 1)

            input_data = {
                'race': 0,
                'gender': 1 if gender == 'Male' else 0,
                'age': age,
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': 1,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'number_diagnoses': number_diagnoses,
                'max_glu_serum': glu_map_i[max_glu],
                'A1Cresult': a1c_map_i[a1c],
                'metformin': 1, 'repaglinide': 0, 'nateglinide': 0,
                'chlorpropamide': 0, 'glimepiride': 0, 'acetohexamide': 0,
                'glipizide': 0, 'glyburide': 0, 'tolbutamide': 0,
                'pioglitazone': 0, 'rosiglitazone': 0, 'acarbose': 0,
                'miglitol': 0, 'troglitazone': 0, 'tolazamide': 0,
                'examide': 0, 'citoglipton': 0,
                'insulin': dose_map_i[insulin],
                'glyburide-metformin': 0, 'glipizide-metformin': 0,
                'glimepiride-pioglitazone': 0, 'metformin-rosiglitazone': 0,
                'metformin-pioglitazone': 0,
                'change': 1 if change == 'Changed' else 0,
                'diabetesMed': 1 if diabetes_med == 'Yes' else 0,
                'total_visits': total_visits,
                'medication_per_day': med_per_day
            }

            input_df = pd.DataFrame([input_data])
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]

            prob     = model.predict_proba(input_df)[0][1]
            risk_pct = round(prob * 100, 1)

            fig2, ax2 = plt.subplots(figsize=(5, 2.5))
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

            if risk_pct > 50:
                st.markdown(f"""<div class="risk-high">
                <h3 style="color:#C62828;margin:0">🔴 High Risk — {risk_pct}%</h3>
                <p style="margin:8px 0 0">High probability of readmission within 30 days.<br>
                <b>Recommended:</b> Discharge planning, follow-up appointment, medication review.</p>
                </div>""", unsafe_allow_html=True)
            elif risk_pct > 25:
                st.markdown(f"""<div style="background:#FFF3E0;border-left:5px solid #FFA726;
                padding:16px;border-radius:8px;">
                <h3 style="color:#E65100;margin:0">🟡 Moderate Risk — {risk_pct}%</h3>
                <p style="margin:8px 0 0">Monitor this patient closely.<br>
                <b>Recommended:</b> Schedule 2-week follow-up, patient education.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="risk-low">
                <h3 style="color:#1B5E20;margin:0">🟢 Low Risk — {risk_pct}%</h3>
                <p style="margin:8px 0 0">Patient has low readmission risk.<br>
                <b>Recommended:</b> Standard discharge protocol.</p>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Fill in patient details on the left and click **Predict Risk**")
        st.markdown("---")
        st.markdown("**Dataset Overview (Diabetes 130-US)**")
        ca, cb, cc = st.columns(3)
        with ca:
            val = f"{n_records:,}" if model_ready else "101K+"
            st.markdown(f'<div class="metric-card"><h2>{val}</h2><p>Patient Records</p></div>',
                        unsafe_allow_html=True)
        with cb:
            val = f"{readmit_rate}%" if model_ready else "11.2%"
            st.markdown(f'<div class="metric-card"><h2>{val}</h2><p>Readmission Rate</p></div>',
                        unsafe_allow_html=True)
        with cc:
            val = str(auc) if model_ready else "~0.68"
            st.markdown(f'<div class="metric-card"><h2>{val}</h2><p>Model ROC-AUC</p></div>',
                        unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "*Built by **Bhaskar Baluni**  "
    "[GitHub](https://github.com/BHASKAR0111) | "
    "[Portfolio](https://bhaskar0111.github.io/Bhaskar-Portfolio/)*"
)
