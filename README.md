# 🏥 Patient Readmission Prediction

> Predicting 30-day hospital readmission using Machine Learning on the Diabetes 130-US dataset.

**Built by:** Bhaskar Baluni 

**Portfolio:** [bhaskar0111.github.io/Bhaskar-Portfolio](https://bhaskar0111.github.io/Bhaskar-Portfolio/)

**Check Live** (https://patientreadmissionprediction-pbzurimedjoycjik3xwrtz.streamlit.app/)

---

## 📌 Project Overview

Hospital readmissions within 30 days are costly and often preventable. This project builds an ML pipeline to identify high-risk patients at the time of discharge — enabling targeted follow-up care.

**Dataset:** [Diabetes 130-US Hospitals (1999–2008)](https://www.kaggle.com/datasets/brandao/diabetic)  
**Records:** 101,766 patient encounters | **Features:** 50

---

## 🎯 Results

| Model | Accuracy | ROC-AUC | Recall |
|---|---|---|---|
| Logistic Regression | ~0.62 | ~0.64 | ~0.60 |
| Random Forest | ~0.67 | ~0.68 | ~0.63 |
| Gradient Boosting | ~0.66 | ~0.67 | ~0.62 |

**Best Model:** Random Forest | **Key Metric:** ROC-AUC (more important than accuracy for imbalanced data)

---

## 🗂️ Project Structure

```
patient-readmission/
├── data/
│   ├── diabetic_data.csv          ← Download from Kaggle
│   ├── processed_data.pkl         ← Generated after preprocessing
│   ├── best_model.pkl             ← Generated after training
│   └── feature_names.pkl          ← Generated after preprocessing
├── notebooks/
│   ├── 01_EDA.ipynb               ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb     ← Cleaning, encoding, SMOTE
│   └── 03_model.ipynb             ← Training, evaluation, feature importance
├── app.py                         ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone & Setup
```bash
git clone https://github.com/BHASKAR0111/patient-readmission-prediction
cd patient-readmission
pip install -r requirements.txt
```

### 2. Download Dataset
- Go to [Kaggle Dataset](https://www.kaggle.com/datasets/brandao/diabetic)
- Download `diabetic_data.csv`
- Place it in the `data/` folder

### 3. Run Notebooks (in order)
```bash
jupyter notebook
# Run: 01_EDA.ipynb → 02_preprocessing.ipynb → 03_model.ipynb
```

### 4. Launch Dashboard
```bash
streamlit run app.py
```

---

## 🔍 Key Insights

- **Prior inpatient visits** is the strongest predictor of readmission
- Patients with **longer hospital stays** have higher readmission risk
- **Number of medications** correlates strongly with readmission
- **Class imbalance** (11% readmission rate) handled using SMOTE

---

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `Imbalanced-learn` `Matplotlib` `Seaborn` `Streamlit`

---

## 📊 Skills Demonstrated

- End-to-end ML pipeline (EDA → Preprocessing → Modeling → Deployment)
- Handling class imbalance with SMOTE
- Model comparison and evaluation
- Interactive dashboard with Streamlit
- Healthcare domain understanding

---

*This project was built as part of my Data Analytics & ML portfolio.*
