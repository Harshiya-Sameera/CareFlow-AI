# CareFlow AI — Healthcare Readmission Analytics & Decision Support

A portfolio-quality healthcare data analytics application that examines historical hospital encounter data for diabetic patients, identifies patterns associated with early hospital readmissions, and generates evidence-based operational insights.

---

## Dataset

| Property | Detail |
|----------|--------|
| **Name** | Diabetes 130-US Hospitals for Years 1999–2008 |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| **Citation** | Strack B, DeShazo JP, Gennings C, et al. *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International. 2014. |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Rows** | 101,766 encounters |
| **Columns** | 50 original; reduced after cleaning |
| **Patients** | 71,518 unique patients (some appear in multiple encounters) |

---

## Problem Statement

Hospital readmissions within 30 days are a primary quality indicator and a source of CMS financial penalties. This application helps hospital administrators and quality improvement teams:

- Track early (within-30-day) readmission KPIs across the eligible patient population
- Identify which clinical and operational factors are most associated with early readmissions
- Use a validated predictive model to understand the relative importance of risk factors
- Receive evidence-based, data-grounded operational recommendations

**Target variable:** Binary — `<30` (readmitted within 30 days) = positive class; `NO` and `>30` = negative class.  
**Eligible population for KPIs and modeling:** All encounters excluding hospice/expired-discharge records (discharge disposition codes 11, 13, 14, 19, 20, 21).

---

## Technology Stack

| Layer | Library | Version |
|-------|---------|---------|
| Frontend + Backend | Streamlit | ≥1.32 |
| Data processing | Pandas, NumPy | ≥2.1, ≥1.26 |
| Visualization | Plotly | ≥5.20 |
| Machine Learning | scikit-learn | ≥1.4 |
| Report generation | python-docx | ≥1.1 |

---

## Project Structure

```
HealthCare_Analytics/
├── app.py                         # Complete application (single file)
├── diabetic_data.csv              # Dataset (UCI ML Repository)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── Sameers_CareFlowAI_Report.docx # Project report
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Navigate to the project directory
cd path/to/HealthCare_Analytics

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The application will open automatically at `http://localhost:8501`.

**Ensure `diabetic_data.csv` is in the same directory as `app.py`.**

---

## Application Features

### Tab 1 — Executive Dashboard
- 5 KPI cards: eligible encounters, early readmission rate, mean length of stay, unique patients, A1C test coverage
- Readmission outcome distribution donut chart
- Early readmission rate by age group and admission type
- Length-of-stay distribution by outcome
- Data quality and population summary (expandable)

### Tab 2 — Patient & Encounter Profile
- Readmission rate by gender and race (descriptive)
- Impact of prior inpatient visits and number of diagnoses
- Encounter volume table by admission type

### Tab 3 — Clinical Patterns
- Early readmission rate by ICD-9 primary diagnosis chapter
- Readmission outcome by insulin usage
- A1C test result vs. early readmission rate
- Polypharmacy (number of medications) vs. readmission
- Medication change during stay vs. readmission
- Medical specialty readmission rates

### Tab 4 — Predictive Model
- Binary Random Forest classifier (early readmit vs. not)
- Patient-grouped train/test split (same patient cannot appear in both sets)
- Confusion matrix, precision/recall/F1 report, ROC-AUC, ROC curve, Precision-Recall curve
- Top-20 feature importances
- Comparison against majority-class baseline

### Tab 5 — Insights & Recommendations
- 6 evidence-based insights derived directly from the loaded dataset
- Operational recommendations tied to specific statistics
- Summary statistics table

---

## Sidebar Filters

All views (except the data-quality summary) respond to:
- **Age Group** — filter by decade bucket
- **Admission Type** — Emergency, Urgent, Elective, etc.
- **Diabetes Medication** — whether a diabetes medication was prescribed

---

## Data Cleaning Summary

| Action | Detail |
|--------|--------|
| Dropped columns | `weight` (96.9% missing), `max_glu_serum` (94.7%), `payer_code` (39.6%), `examide`/`citoglipton` (0% variance), 3 near-zero combo drug columns |
| Missing recoded | `race` `?` → `"Unknown"`, `medical_specialty` `?` → `"Unknown"`, `A1Cresult` `None` → `"Not Tested"` |
| Removed rows | 3 rows with `gender = "Unknown/Invalid"` |
| Hospice/expired | 2,423 rows flagged and excluded from KPI denominator and ML model; included in raw counts |
| ICD-9 mapping | `diag_1` and `diag_2` mapped to 18 standard ICD-9 disease chapters |
| ML split | `GroupShuffleSplit` by `patient_nbr` (80/20); identifiers excluded from features |

---

## Model Performance Notes

The Random Forest classifier is trained with `class_weight='balanced'` to account for class imbalance (≈11% positive class). Evaluation metrics are reported on a held-out test set where no patient from training appears. ROC-AUC is the primary metric given the imbalanced target.

**The model is intended for operational pattern discovery only. It is not validated for clinical use and should not be used to make individual patient care decisions.**

---

## Limitations

1. This dataset is from 1999–2008 and may not reflect current treatment protocols or patient populations.
2. The target variable (`readmitted`) reflects readmission to the same hospital system; transfers or readmissions elsewhere are not captured.
3. Missing values in `weight`, `max_glu_serum`, and `A1Cresult` limit the completeness of some clinical analyses.
4. `medical_specialty` is missing for 49% of encounters, limiting specialty-level analysis.
5. All associations are descriptive. Correlation does not imply causation; clinical and operational context is required before acting on findings.
6. The predictive model uses administrative features only. Clinical notes, lab trends, and social determinants of health are not available.

---

## License

This project uses the UCI Diabetes 130-US Hospitals dataset under CC BY 4.0.  
Application code: MIT License.
