"""
CareFlow AI — Healthcare Readmission Analytics & Decision Support
Dataset: UCI ML Repository — Diabetes 130-US Hospitals (1999–2008)
        Strack et al., BioMed Research International, 2014
        License: CC BY 4.0
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score
)
from sklearn.preprocessing import LabelEncoder
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "diabetic_data.csv"
PAGE_TITLE = "CareFlow AI — Readmission Analytics"

ADMISSION_TYPE_MAP = {
    "1": "Emergency", "2": "Urgent", "3": "Elective",
    "4": "Newborn", "5": "Not Available", "6": "NULL",
    "7": "Trauma Center", "8": "Not Mapped"
}
ADMISSION_SOURCE_MAP = {
    "1": "Physician Referral", "2": "Clinic Referral", "3": "HMO Referral",
    "4": "Transfer from Hospital", "5": "Transfer from SNF",
    "6": "Transfer from Another", "7": "Emergency Room",
    "8": "Court/Law Enforcement", "9": "Not Available",
    "10": "Transfer: Federal", "11": "Normal Delivery",
    "12": "Premature Delivery", "13": "Sick Baby",
    "14": "Extramural Birth", "15": "Not Available",
    "17": "NULL", "18": "Transfer: Ambulatory",
    "19": "Transfer: Acute Care", "20": "Unknown",
    "22": "Transfer: Rehab", "25": "Transfer: CAH", "26": "Transfer: Cancer"
}
# Hospice / expired — these records are kept in descriptive analysis but
# excluded from the eligible-population for readmission modeling
HOSPICE_EXPIRED_CODES = {"11", "13", "14", "19", "20", "21"}

# ICD-9 chapter mapping (reliable top-level numeric ranges + E/V codes)
def map_icd9_chapter(code):
    """Map an ICD-9 code string to a broad disease chapter label."""
    if pd.isna(code) or str(code).strip() in ("?", "", "nan"):
        return "Unknown"
    code = str(code).strip()
    if code.startswith("V"):
        return "V-Codes (Supplementary)"
    if code.startswith("E"):
        return "E-Codes (External Causes)"
    try:
        num = float(code)
    except ValueError:
        return "Unknown"
    if 1 <= num <= 139:      return "Infectious & Parasitic"
    if 140 <= num <= 239:    return "Neoplasms"
    if 240 <= num <= 279:    return "Endocrine/Nutritional/Metabolic"
    if 280 <= num <= 289:    return "Blood & Blood-Forming Organs"
    if 290 <= num <= 319:    return "Mental Disorders"
    if 320 <= num <= 389:    return "Nervous System & Sense Organs"
    if 390 <= num <= 459:    return "Circulatory System"
    if 460 <= num <= 519:    return "Respiratory System"
    if 520 <= num <= 579:    return "Digestive System"
    if 580 <= num <= 629:    return "Genitourinary System"
    if 630 <= num <= 679:    return "Pregnancy & Childbirth"
    if 680 <= num <= 709:    return "Skin & Subcutaneous Tissue"
    if 710 <= num <= 739:    return "Musculoskeletal System"
    if 740 <= num <= 759:    return "Congenital Anomalies"
    if 760 <= num <= 779:    return "Perinatal Conditions"
    if 780 <= num <= 799:    return "Symptoms & Ill-Defined"
    if 800 <= num <= 999:    return "Injury & Poisoning"
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning dataset…")
def load_and_clean(path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns:
        df_all   — full cleaned dataset (all encounters, for descriptive stats)
        df_elig  — eligible population (hospice/expired excluded, for KPIs + model)
        meta     — data-quality summary dict
    """
    df = pd.read_csv(path, dtype=str, na_values=["?", "", "None"])

    meta = {}
    meta["raw_rows"] = len(df)
    meta["raw_cols"] = len(df.columns)

    # --- Drop near-zero or near-fully-missing columns ---
    drop_cols = [
        "weight",           # 96.9% missing
        "max_glu_serum",    # 94.7% missing
        "payer_code",       # 39.6% missing, low analytic value
        "examide",          # 100% "No" — zero variance
        "citoglipton",      # 100% "No" — zero variance
        "glimepiride-pioglitazone",   # >99.999% "No"
        "metformin-rosiglitazone",    # >99.999% "No"
        "metformin-pioglitazone",     # >99.999% "No"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # --- Recode missing categoricals ---
    df["race"] = df["race"].fillna("Unknown")
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    df["A1Cresult"] = df["A1Cresult"].fillna("Not Tested")

    # --- Drop 3 rows with unknown gender ---
    df = df[df["gender"] != "Unknown/Invalid"].copy()

    # --- Label-map admission type and source ---
    df["admission_type_label"] = df["admission_type_id"].map(ADMISSION_TYPE_MAP).fillna("Unknown")
    df["admission_source_label"] = df["admission_source_id"].map(ADMISSION_SOURCE_MAP).fillna("Unknown")

    # --- Numeric conversions ---
    num_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- ICD-9 chapter mapping ---
    df["diag1_chapter"] = df["diag_1"].apply(map_icd9_chapter)
    df["diag2_chapter"] = df["diag_2"].apply(map_icd9_chapter)

    # --- Target variable ---
    df["readmitted"] = df["readmitted"].fillna("NO")
    df["is_early_readmit"] = (df["readmitted"] == "<30").astype(int)

    # --- Flag hospice/expired records ---
    df["is_hospice_expired"] = df["discharge_disposition_id"].isin(HOSPICE_EXPIRED_CODES)

    meta["total_cleaned"] = len(df)
    meta["hospice_expired_count"] = int(df["is_hospice_expired"].sum())
    # unique_patients_all: across full cleaned dataset (includes hospice/expired records)
    # used only in the Data Quality expander to show total scope
    meta["unique_patients_all"] = df["patient_nbr"].nunique()
    meta["repeated_patient_encounters"] = int(
        (df["patient_nbr"].map(df["patient_nbr"].value_counts()) > 1).sum()
    )

    df_all = df.copy()

    # Eligible population: exclude hospice/expired from KPI denominator and model
    df_elig = df[~df["is_hospice_expired"]].copy()
    meta["eligible_rows"] = len(df_elig)
    # unique_patients: patients in the eligible population only — consistent with KPI denominator
    meta["unique_patients"] = df_elig["patient_nbr"].nunique()
    meta["early_readmit_count"] = int(df_elig["is_early_readmit"].sum())
    meta["early_readmit_rate"] = round(
        100 * meta["early_readmit_count"] / meta["eligible_rows"], 2
    )

    return df_all, df_elig, meta


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
]
MED_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}

@st.cache_data(show_spinner="Training predictive model…")
def train_model(df_elig: pd.DataFrame):
    """
    Binary classification: is_early_readmit (1 = readmitted <30 days).
    Uses patient-grouped 80/20 split to prevent data leakage from repeated patients.
    Excludes patient/encounter identifiers from features.
    """
    df = df_elig.copy()

    # --- Feature engineering ---
    cat_features = ["race", "gender", "age", "admission_type_id",
                    "discharge_disposition_id", "admission_source_id",
                    "A1Cresult", "change", "diabetesMed",
                    "diag1_chapter", "diag2_chapter"]
    num_features = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses"
    ]

    # Medication columns: map to ordinal
    for col in MED_COLS:
        if col in df.columns:
            df[col + "_enc"] = df[col].map(MED_MAP).fillna(0).astype(int)
    med_enc_cols = [c + "_enc" for c in MED_COLS if c in df.columns]

    # Label-encode categoricals
    le_dict = {}
    for col in cat_features:
        df[col] = df[col].astype(str).fillna("Unknown")
        le = LabelEncoder()
        df[col + "_le"] = le.fit_transform(df[col])
        le_dict[col] = le
    cat_enc_cols = [c + "_le" for c in cat_features]

    feature_cols = num_features + cat_enc_cols + med_enc_cols
    X = df[feature_cols].fillna(0).values
    y = df["is_early_readmit"].values
    groups = df["patient_nbr"].values

    # --- Patient-grouped split (same patient cannot appear in both sets) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --- Baseline: majority class ---
    majority_class = int(np.bincount(y_train).argmax())
    y_baseline = np.full(len(y_test), majority_class)
    baseline_f1 = f1_score(y_test, y_baseline, zero_division=0)
    baseline_auc = 0.5  # majority class gives 0.5 AUC by definition

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    rf_auc = roc_auc_score(y_test, y_prob)
    rf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Friendly feature name mapping
    name_map = {
        "time_in_hospital": "Length of Stay",
        "num_lab_procedures": "# Lab Procedures",
        "num_procedures": "# Procedures",
        "num_medications": "# Medications",
        "number_outpatient": "# Prior Outpatient",
        "number_emergency": "# Prior Emergency",
        "number_inpatient": "# Prior Inpatient",
        "number_diagnoses": "# Diagnoses",
        "age_le": "Age Group",
        "race_le": "Race",
        "gender_le": "Gender",
        "admission_type_id_le": "Admission Type",
        "discharge_disposition_id_le": "Discharge Disposition",
        "admission_source_id_le": "Admission Source",
        "A1Cresult_le": "A1C Result",
        "change_le": "Medication Change",
        "diabetesMed_le": "Diabetes Medication",
        "diag1_chapter_le": "Primary Dx Chapter",
        "diag2_chapter_le": "Secondary Dx Chapter",
        "insulin_enc": "Insulin Usage",
        "metformin_enc": "Metformin",
        "glipizide_enc": "Glipizide",
        "glyburide_enc": "Glyburide",
        "pioglitazone_enc": "Pioglitazone",
        "rosiglitazone_enc": "Rosiglitazone",
        "glimepiride_enc": "Glimepiride",
    }
    importances.index = [name_map.get(i, i) for i in importances.index]

    return {
        "rf": rf,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "rf_auc": rf_auc,
        "rf_report": rf_report,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
        "importances": importances.head(20),
        "baseline_f1": baseline_f1,
        "baseline_auc": baseline_auc,
        "feature_cols": feature_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#3b82d4",
    "danger": "#e05252",
    "success": "#22a86a",
    "warning": "#e09a27",
    "neutral": "#7c5cd8",
    "bg": "#f7f8fa",
}

def kpi_card(label, value, delta=None, suffix="", color=None):
    color = color or COLORS["primary"]
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        delta_color = COLORS["danger"] if delta >= 0 else COLORS["success"]
        delta_html = f'<div style="font-size:12px;color:{delta_color}">{arrow} {abs(delta):.1f} pp vs benchmark</div>'
    return f"""
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;
                padding:16px 20px;text-align:center;min-height:90px">
      <div style="font-size:12px;color:#57606a;margin-bottom:4px">{label}</div>
      <div style="font-size:28px;font-weight:700;color:{color}">{value}{suffix}</div>
      {delta_html}
    </div>"""


def readmission_donut(df_elig):
    counts = df_elig["readmitted"].value_counts().reset_index()
    counts.columns = ["Outcome", "Count"]
    label_map = {"NO": "Not Readmitted", ">30": "Readmitted >30 Days", "<30": "Readmitted <30 Days"}
    counts["Outcome"] = counts["Outcome"].map(label_map).fillna(counts["Outcome"])
    fig = px.pie(
        counts, names="Outcome", values="Count", hole=0.45,
        color="Outcome",
        color_discrete_map={
            "Not Readmitted": COLORS["success"],
            "Readmitted >30 Days": COLORS["warning"],
            "Readmitted <30 Days": COLORS["danger"],
        },
        title="Readmission Outcome Distribution"
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
    return fig


def readmit_by_age(df_elig):
    age_order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                 "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
    grp = (
        df_elig.groupby("age")["is_early_readmit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "early_readmit", "count": "total"})
        .reset_index()
    )
    grp["rate"] = 100 * grp["early_readmit"] / grp["total"]
    grp["age"] = pd.Categorical(grp["age"], categories=age_order, ordered=True)
    grp = grp.sort_values("age")
    fig = px.bar(
        grp, x="age", y="rate", text=grp["rate"].round(1).astype(str) + "%",
        title="Early Readmission Rate by Age Group",
        labels={"age": "Age Group", "rate": "Readmission Rate (%)"},
        color="rate",
        color_continuous_scale=[[0, COLORS["success"]], [0.5, COLORS["warning"]], [1, COLORS["danger"]]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10), xaxis_tickangle=-30)
    return fig


def readmit_by_admission_type(df_elig):
    grp = (
        df_elig.groupby("admission_type_label")["is_early_readmit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "early_readmit", "count": "total"})
        .reset_index()
    )
    grp = grp[grp["total"] >= 50]
    grp["rate"] = 100 * grp["early_readmit"] / grp["total"]
    grp = grp.sort_values("rate", ascending=True)
    fig = px.bar(
        grp, x="rate", y="admission_type_label", orientation="h",
        title="Early Readmission Rate by Admission Type",
        labels={"rate": "Rate (%)", "admission_type_label": ""},
        text=grp["rate"].round(1).astype(str) + "%",
        color="rate",
        color_continuous_scale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
    return fig


def los_distribution(df_elig):
    fig = px.histogram(
        df_elig, x="time_in_hospital", color="readmitted",
        barmode="overlay", opacity=0.75,
        title="Length of Stay Distribution by Readmission Outcome",
        labels={"time_in_hospital": "Days in Hospital", "readmitted": "Outcome"},
        color_discrete_map={"NO": COLORS["success"], ">30": COLORS["warning"], "<30": COLORS["danger"]},
        nbins=14,
    )
    fig.update_layout(margin=dict(t=40, b=10))
    return fig


def demographics_bar(df_elig, col, title):
    grp = (
        df_elig.groupby(col)["is_early_readmit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "early", "count": "total"})
        .reset_index()
    )
    grp = grp[grp["total"] >= 30]
    grp["rate"] = 100 * grp["early"] / grp["total"]
    grp = grp.sort_values("rate", ascending=False)
    fig = px.bar(
        grp, x=col, y="rate", text=grp["rate"].round(1).astype(str) + "%",
        title=title,
        labels={"rate": "Early Readmission Rate (%)"},
        color="rate",
        color_continuous_scale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
    return fig


def diagnosis_chapter_heatmap(df_elig):
    grp = (
        df_elig.groupby("diag1_chapter")["is_early_readmit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "early", "count": "total"})
        .reset_index()
    )
    grp = grp[grp["total"] >= 50].copy()
    grp["rate"] = 100 * grp["early"] / grp["total"]
    grp = grp.sort_values("rate", ascending=True)
    fig = px.bar(
        grp, x="rate", y="diag1_chapter", orientation="h",
        title="Early Readmission Rate by Primary Diagnosis Chapter (ICD-9)",
        labels={"rate": "Rate (%)", "diag1_chapter": ""},
        text=grp["rate"].round(1).astype(str) + "%",
        color="rate",
        color_continuous_scale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=520, margin=dict(t=40, b=10, l=200))
    return fig


def insulin_readmit_chart(df_elig):
    grp = (
        df_elig.groupby(["insulin", "readmitted"])
        .size()
        .reset_index(name="count")
    )
    total_by_insulin = df_elig.groupby("insulin").size().rename("total")
    grp = grp.merge(total_by_insulin, on="insulin")
    grp["pct"] = 100 * grp["count"] / grp["total"]
    label_map = {"NO": "Not Readmitted", ">30": "Readmitted >30d", "<30": "Readmitted <30d"}
    grp["readmitted_label"] = grp["readmitted"].map(label_map)
    fig = px.bar(
        grp, x="insulin", y="pct", color="readmitted_label",
        barmode="stack",
        title="Readmission Outcome by Insulin Usage",
        labels={"insulin": "Insulin", "pct": "% of Encounters", "readmitted_label": "Outcome"},
        color_discrete_map={
            "Not Readmitted": COLORS["success"],
            "Readmitted >30d": COLORS["warning"],
            "Readmitted <30d": COLORS["danger"],
        },
    )
    fig.update_layout(margin=dict(t=40, b=10))
    return fig


def a1c_readmit_chart(df_elig):
    grp = (
        df_elig.groupby("A1Cresult")["is_early_readmit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "early", "count": "total"})
        .reset_index()
    )
    grp["rate"] = 100 * grp["early"] / grp["total"]
    a1c_order = ["Not Tested", "Norm", ">7", ">8"]
    grp["A1Cresult"] = pd.Categorical(grp["A1Cresult"], categories=a1c_order, ordered=True)
    grp = grp.sort_values("A1Cresult")
    fig = px.bar(
        grp, x="A1Cresult", y="rate",
        text=grp["rate"].round(1).astype(str) + "%",
        title="Early Readmission Rate by A1C Test Result",
        labels={"A1Cresult": "A1C Result", "rate": "Early Readmission Rate (%)"},
        color="rate",
        color_continuous_scale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
    return fig


def medications_scatter(df_elig):
    grp = (
        df_elig.groupby("num_medications")
        .agg(
            total=("is_early_readmit", "count"),
            early=("is_early_readmit", "sum")
        )
        .reset_index()
    )
    grp = grp[grp["total"] >= 20]
    grp["rate"] = 100 * grp["early"] / grp["total"]
    fig = px.scatter(
        grp, x="num_medications", y="rate", size="total",
        title="Early Readmission Rate vs. Number of Medications",
        labels={"num_medications": "Number of Medications", "rate": "Early Readmission Rate (%)", "total": "Encounters"},
        trendline="ols",
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig.update_layout(margin=dict(t=40, b=10))
    return fig


def specialty_table(df_elig, top_n=15):
    grp = (
        df_elig[df_elig["medical_specialty"] != "Unknown"]
        .groupby("medical_specialty")["is_early_readmit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Early Readmissions", "count": "Total Encounters"})
        .reset_index()
    )
    grp = grp[grp["Total Encounters"] >= 100]
    grp["Early Readmission Rate (%)"] = (100 * grp["Early Readmissions"] / grp["Total Encounters"]).round(2)
    grp = grp.rename(columns={"medical_specialty": "Medical Specialty"})
    grp = grp.sort_values("Early Readmission Rate (%)", ascending=False).head(top_n)
    return grp.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ML CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm):
    labels = ["Negative\n(No early readmit)", "Positive\n(Early readmit)"]
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        x=labels, y=labels,
        title="Confusion Matrix (Test Set)",
    )
    fig.update_layout(margin=dict(t=40, b=10), height=380)
    return fig


def plot_roc(fpr, tpr, auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"Random Forest (AUC = {auc:.3f})",
                             line=dict(color=COLORS["primary"], width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Majority Baseline (AUC = 0.500)",
                             line=dict(color="#aaa", width=1, dash="dash")))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        margin=dict(t=40, b=10),
    )
    return fig


def plot_pr_curve(prec, rec):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                             name="Random Forest",
                             line=dict(color=COLORS["neutral"], width=2)))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(t=40, b=10),
    )
    return fig


def plot_feature_importance(importances):
    df_imp = importances.reset_index()
    df_imp.columns = ["Feature", "Importance"]
    df_imp = df_imp.sort_values("Importance", ascending=True)
    fig = px.bar(
        df_imp, x="Importance", y="Feature", orientation="h",
        title="Top 20 Feature Importances (Random Forest, Gini)",
        labels={"Importance": "Mean Decrease in Impurity", "Feature": ""},
        color="Importance",
        color_continuous_scale=[[0, "#c8d8f0"], [1, COLORS["primary"]]],
    )
    fig.update_layout(coloraxis_showscale=False, height=520, margin=dict(t=40, b=10))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INSIGHTS GENERATOR (stats-grounded, no invented facts)
# ─────────────────────────────────────────────────────────────────────────────
def generate_insights(df_elig, meta, model_results):
    insights = []

    # 1. Overall rate
    rate = meta["early_readmit_rate"]
    insights.append({
        "title": "Overall Early Readmission Burden",
        "finding": (
            f"Among the {meta['eligible_rows']:,} eligible encounters, "
            f"{meta['early_readmit_count']:,} ({rate}%) resulted in readmission within 30 days. "
            f"This is the primary KPI for CMS penalty risk."
        ),
        "recommendation": (
            "Establish a baseline tracking dashboard and set a reduction target. "
            "A 1-percentage-point reduction (~1,000 fewer early readmissions per 100K encounters) "
            "represents a material improvement in quality and cost."
        ),
    })

    # 2. Prior inpatient visits
    hi_prior = df_elig[df_elig["number_inpatient"] >= 3]
    if len(hi_prior) > 0:
        hi_rate = 100 * hi_prior["is_early_readmit"].mean()
        insights.append({
            "title": "High-Risk: Patients with ≥3 Prior Inpatient Visits",
            "finding": (
                f"{len(hi_prior):,} encounters involve patients with 3 or more prior inpatient visits. "
                f"Their early readmission rate is {hi_rate:.1f}%, "
                f"compared to the overall {rate}%."
            ),
            "recommendation": (
                "Flag patients with ≥3 prior inpatient visits for intensive discharge planning, "
                "post-discharge follow-up calls within 48 hours, and referral to case management."
            ),
        })

    # 3. A1C testing gap
    tested_pct = 100 * (df_elig["A1Cresult"] != "Not Tested").mean()
    tested_rate = 100 * df_elig[df_elig["A1Cresult"] != "Not Tested"]["is_early_readmit"].mean()
    untested_rate = 100 * df_elig[df_elig["A1Cresult"] == "Not Tested"]["is_early_readmit"].mean()
    insights.append({
        "title": "A1C Testing Coverage Gap",
        "finding": (
            f"Only {tested_pct:.1f}% of eligible encounters have an A1C result documented. "
            f"Untested patients have an early readmission rate of {untested_rate:.1f}%, "
            f"while tested patients have a rate of {tested_rate:.1f}%."
        ),
        "recommendation": (
            "Implement a protocol to ensure A1C testing during every diabetic inpatient admission. "
            "The testing gap may represent a missed opportunity to adjust glucose management before discharge."
        ),
    })

    # 4. Insulin usage
    ins_up = df_elig[df_elig["insulin"] == "Up"]["is_early_readmit"].mean() * 100
    ins_no = df_elig[df_elig["insulin"] == "No"]["is_early_readmit"].mean() * 100
    insights.append({
        "title": "Insulin Dose Adjustment and Readmission",
        "finding": (
            f"Encounters where insulin was increased ('Up') show a {ins_up:.1f}% early readmission rate, "
            f"compared to {ins_no:.1f}% among patients not on insulin. "
            f"Note: this is an association, not causation — sicker patients may receive more insulin adjustments."
        ),
        "recommendation": (
            "Patients with insulin dose escalation warrant enhanced post-discharge glucose monitoring "
            "and structured follow-up within 7 days to prevent glycemic crises that trigger readmission."
        ),
    })

    # 5. Polypharmacy
    high_med = df_elig[df_elig["num_medications"] >= 15]["is_early_readmit"].mean() * 100
    low_med = df_elig[df_elig["num_medications"] < 10]["is_early_readmit"].mean() * 100
    insights.append({
        "title": "Polypharmacy and Readmission Risk",
        "finding": (
            f"Patients prescribed 15 or more medications have an early readmission rate of {high_med:.1f}%, "
            f"compared to {low_med:.1f}% for those with fewer than 10 medications. "
            f"Polypharmacy is a proxy for disease severity as well as an independent complexity factor."
        ),
        "recommendation": (
            "Include a pharmacist-led medication reconciliation in the discharge checklist for patients "
            "with 15+ active medications. Review for drug interactions and simplify regimens where possible."
        ),
    })

    # 6. Model insight
    if model_results:
        auc = model_results["rf_auc"]
        top_feature = model_results["importances"].index[0]
        insights.append({
            "title": "Predictive Model: Key Risk Signals",
            "finding": (
                f"The Random Forest model achieved a ROC-AUC of {auc:.3f} on the held-out test set. "
                f"The most predictive feature was '{top_feature}', followed by "
                f"'{model_results['importances'].index[1]}' and '{model_results['importances'].index[2]}'. "
                f"The model is intended for pattern discovery only — it is not validated for individual clinical decisions."
            ),
            "recommendation": (
                "Use the feature importances as a prioritization guide for quality improvement initiatives. "
                "Focus operational interventions on the highest-importance variables where clinical workflows "
                "can realistically be modified."
            ),
        })

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main { background: #f7f8fa; }
        .block-container { padding-top: 1.5rem; }
        h1 { color: #1f2328; }
        h2 { color: #3b82d4; font-size: 1.15rem; }
        .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
        .metric-note { font-size: 11px; color: #57606a; margin-top: 2px; }
        .caveat-box {
            background: #1E293B; border-left: 4px solid #F59E0B;
            padding: 10px 14px; border-radius: 4px; font-size: 13px;
            margin-bottom: 12px; color: #F8FAFC;
        }
        .insight-card {
            background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
            padding: 16px 20px; margin-bottom: 12px;
        }
        .insight-title { font-weight: 700; color: #1f2328; font-size: 14px; }
        .insight-finding { color: #57606a; font-size: 13px; margin: 6px 0; }
        .insight-rec { color: #3b82d4; font-size: 13px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.title("🏥 CareFlow AI — Healthcare Readmission Analytics")
    st.markdown(
        "<span style='color:#57606a;font-size:13px'>"
        "Dataset: UCI ML Repository — Diabetes 130-US Hospitals (1999–2008) · "
        "Strack et al., BioMed Research International, 2014 · License: CC BY 4.0"
        "</span>",
        unsafe_allow_html=True,
    )

    # ── Data loading ──
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at `{DATA_PATH}`. Place `diabetic_data.csv` in the working directory.")
        st.stop()

    df_all, df_elig, meta = load_and_clean(DATA_PATH)

    # ── Sidebar filters ──
    st.sidebar.header("Filters")
    st.sidebar.caption("Filters apply to all views except the data-quality summary.")

    age_options = sorted([a for a in df_elig["age"].unique() if pd.notna(a)])
    sel_age = st.sidebar.multiselect("Age Group", age_options, default=age_options)

    adm_options = sorted(df_elig["admission_type_label"].unique())
    sel_adm = st.sidebar.multiselect("Admission Type", adm_options, default=adm_options)

    diabmed_options = sorted(df_elig["diabetesMed"].dropna().unique())
    sel_diabmed = st.sidebar.multiselect("Diabetes Medication", diabmed_options, default=diabmed_options)

    # Apply filters
    mask = (
        df_elig["age"].isin(sel_age) &
        df_elig["admission_type_label"].isin(sel_adm) &
        df_elig["diabetesMed"].isin(sel_diabmed)
    )
    df_f = df_elig[mask].copy()

    if len(df_f) == 0:
        st.warning("No data matches the current filter selection. Please adjust the sidebar filters.")
        st.stop()

    filtered_rate = round(100 * df_f["is_early_readmit"].mean(), 2)
    filtered_early = int(df_f["is_early_readmit"].sum())

    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Encounters", f"{len(df_f):,}")
    st.sidebar.metric("Filtered Early Readmission Rate", f"{filtered_rate}%")

    # ── TRAIN MODEL (on full eligible set, not filtered) ──
    model_results = train_model(df_elig)

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Dashboard",
        "👥 Patient & Encounter Profile",
        "🔬 Clinical Patterns",
        "🤖 Predictive Model",
        "💡 Insights & Recommendations",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — EXECUTIVE DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Key Performance Indicators")
        st.markdown(
            "<div class='caveat-box'>"
            "<b>Eligible population:</b> All encounters after excluding hospice/expired-discharge records "
            f"(n = {meta['hospice_expired_count']:,} excluded). "
            "KPIs reflect the filtered view unless stated otherwise."
            "</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(kpi_card("Eligible Encounters", f"{len(df_f):,}"), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("Early Readmission Rate", f"{filtered_rate}", suffix="%",
                                 color=COLORS["danger"]), unsafe_allow_html=True)
        with c3:
            mean_los = round(df_f["time_in_hospital"].mean(), 1)
            st.markdown(kpi_card("Mean Length of Stay", f"{mean_los}", suffix=" days"), unsafe_allow_html=True)
        with c4:
            unique_pts = df_f["patient_nbr"].nunique()
            st.markdown(kpi_card("Unique Patients", f"{unique_pts:,}"), unsafe_allow_html=True)
        with c5:
            a1c_cov = round(100 * (df_f["A1Cresult"] != "Not Tested").mean(), 1)
            st.markdown(kpi_card("A1C Test Coverage", f"{a1c_cov}", suffix="%",
                                 color=COLORS["warning"]), unsafe_allow_html=True)

        st.markdown("---")

        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.plotly_chart(readmission_donut(df_f), use_container_width=True)
        with col_right:
            st.plotly_chart(readmit_by_age(df_f), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(readmit_by_admission_type(df_f), use_container_width=True)
        with col_b:
            st.plotly_chart(los_distribution(df_f), use_container_width=True)

        # Data quality note
        with st.expander("📋 Data Quality & Population Summary"):
            dq_data = {
                "Metric": [
                    "Raw rows in dataset",
                    "Rows after cleaning (gender unknown removed)",
                    "Hospice / expired discharge records (excluded from KPIs & model)",
                    "Eligible population (KPI denominator)",
                    "Unique patients (eligible population only)",
                    "Unique patients (all cleaned records, incl. hospice/expired)",
                    "Patients with >1 encounter",
                    "Encounters matching current filters",
                    "Columns dropped (high missingness / zero variance)",
                    "weight missing (%)", "max_glu_serum missing (%)",
                    "payer_code missing (%)", "A1C 'Not Tested' (%)",
                    "medical_specialty unknown (%)",
                ],
                "Value": [
                    f"{meta['raw_rows']:,}",
                    f"{meta['total_cleaned']:,}",
                    f"{meta['hospice_expired_count']:,}",
                    f"{meta['eligible_rows']:,}",
                    f"{meta['unique_patients']:,}",
                    f"{meta['unique_patients_all']:,}",
                    f"{meta['repeated_patient_encounters']:,} encounters",
                    f"{len(df_f):,}",
                    "weight, max_glu_serum, payer_code, examide, citoglipton, + 3 combo drugs",
                    "96.9%", "94.7%", "39.6%",
                    f"{100*(df_elig['A1Cresult']=='Not Tested').mean():.1f}%",
                    f"{100*(df_elig['medical_specialty']=='Unknown').mean():.1f}%",
                ],
            }
            st.dataframe(pd.DataFrame(dq_data), use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — PATIENT & ENCOUNTER PROFILE
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Patient Demographics & Encounter Characteristics")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(demographics_bar(df_f, "gender", "Early Readmission Rate by Gender"),
                            use_container_width=True)
        with col2:
            st.plotly_chart(demographics_bar(df_f, "race", "Early Readmission Rate by Race"),
                            use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            # Prior utilization
            df_prior = df_f.copy()
            df_prior["prior_inpatient_bucket"] = pd.cut(
                df_prior["number_inpatient"],
                bins=[-1, 0, 1, 2, 4, 100],
                labels=["0", "1", "2", "3–4", "5+"]
            )
            grp_prior = (
                df_prior.groupby("prior_inpatient_bucket", observed=True)["is_early_readmit"]
                .agg(["sum", "count"]).rename(columns={"sum": "early", "count": "total"})
                .reset_index()
            )
            grp_prior["rate"] = 100 * grp_prior["early"] / grp_prior["total"]
            fig_prior = px.bar(
                grp_prior, x="prior_inpatient_bucket", y="rate",
                text=grp_prior["rate"].round(1).astype(str) + "%",
                title="Early Readmission Rate by Prior Inpatient Visits",
                labels={"prior_inpatient_bucket": "Prior Inpatient Visits", "rate": "Rate (%)"},
                color="rate",
                color_continuous_scale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
            )
            fig_prior.update_traces(textposition="outside")
            fig_prior.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig_prior, use_container_width=True)

        with col4:
            # Number of diagnoses
            df_dx = df_f.copy()
            df_dx["dx_bucket"] = pd.cut(
                df_dx["number_diagnoses"],
                bins=[0, 3, 6, 9, 16],
                labels=["1–3", "4–6", "7–9", "10+"]
            )
            grp_dx = (
                df_dx.groupby("dx_bucket", observed=True)["is_early_readmit"]
                .agg(["sum", "count"]).rename(columns={"sum": "early", "count": "total"})
                .reset_index()
            )
            grp_dx["rate"] = 100 * grp_dx["early"] / grp_dx["total"]
            fig_dx = px.bar(
                grp_dx, x="dx_bucket", y="rate",
                text=grp_dx["rate"].round(1).astype(str) + "%",
                title="Early Readmission Rate by Number of Diagnoses",
                labels={"dx_bucket": "Number of Diagnoses", "rate": "Rate (%)"},
                color="rate",
                color_continuous_scale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
            )
            fig_dx.update_traces(textposition="outside")
            fig_dx.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig_dx, use_container_width=True)

        # Encounter volume summary table
        st.markdown("#### Encounter Volume by Admission Type")
        adm_tbl = (
            df_f.groupby("admission_type_label")
            .agg(
                Encounters=("encounter_id", "count"),
                Early_Readmits=("is_early_readmit", "sum"),
                Mean_LOS=("time_in_hospital", "mean"),
                Mean_Medications=("num_medications", "mean"),
            )
            .round(2)
            .reset_index()
        )
        adm_tbl["Early Readmission Rate (%)"] = (
            100 * adm_tbl["Early_Readmits"] / adm_tbl["Encounters"]
        ).round(2)
        adm_tbl = adm_tbl.rename(columns={
            "admission_type_label": "Admission Type",
            "Early_Readmits": "Early Readmissions",
            "Mean_LOS": "Avg LOS (days)",
            "Mean_Medications": "Avg # Medications",
        })
        st.dataframe(adm_tbl, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — CLINICAL PATTERNS
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### Clinical & Operational Patterns")

        st.plotly_chart(diagnosis_chapter_heatmap(df_f), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(insulin_readmit_chart(df_f), use_container_width=True)
        with col2:
            st.plotly_chart(a1c_readmit_chart(df_f), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(medications_scatter(df_f), use_container_width=True)
        with col4:
            # Medication change flag
            grp_ch = (
                df_f.groupby("change")["is_early_readmit"]
                .agg(["sum", "count"]).rename(columns={"sum": "early", "count": "total"})
                .reset_index()
            )
            grp_ch["rate"] = 100 * grp_ch["early"] / grp_ch["total"]
            grp_ch["change"] = grp_ch["change"].map({"No": "No Change", "Ch": "Medication Changed"})
            fig_ch = px.bar(
                grp_ch, x="change", y="rate",
                text=grp_ch["rate"].round(1).astype(str) + "%",
                title="Early Readmission Rate by Medication Change During Stay",
                labels={"change": "", "rate": "Rate (%)"},
                color="rate",
                color_continuous_scale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
            )
            fig_ch.update_traces(textposition="outside")
            fig_ch.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig_ch, use_container_width=True)

        # Medical specialty table
        st.markdown("#### Medical Specialty Readmission Rates (min. 100 encounters, specialty known)")
        spec_tbl = specialty_table(df_f)
        if len(spec_tbl) > 0:
            st.dataframe(spec_tbl, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data for specialty breakdown with current filters.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4 — PREDICTIVE MODEL
    # ═══════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### Early Readmission Predictive Model")

        st.markdown(
            "<div class='caveat-box'>"
            "<b>⚠️ Important limitations:</b> This model is trained on historical administrative data "
            "for research and operational planning only. It is <b>not validated for individual clinical decisions</b>, "
            "and associations identified by the model do not imply causation. "
            "The model target is binary: <b>readmitted within 30 days (<30)</b> vs. not (NO or >30 days).<br/><br/>"
            "<b>Leakage prevention:</b> Patient and encounter identifiers are excluded from all features. "
            "Patients are grouped so the same patient cannot appear in both training and test sets "
            "(GroupShuffleSplit by patient_nbr). Hospice/expired-discharge encounters are excluded."
            "</div>",
            unsafe_allow_html=True,
        )

        mr = model_results

        # Training summary
        col0a, col0b, col0c, col0d = st.columns(4)
        with col0a:
            st.markdown(kpi_card("Training Samples", f"{mr['X_train_shape'][0]:,}"), unsafe_allow_html=True)
        with col0b:
            st.markdown(kpi_card("Test Samples", f"{mr['X_test_shape'][0]:,}"), unsafe_allow_html=True)
        with col0c:
            st.markdown(kpi_card("Model ROC-AUC", f"{mr['rf_auc']:.3f}", color=COLORS["primary"]), unsafe_allow_html=True)
        with col0d:
            st.markdown(kpi_card("Baseline ROC-AUC", "0.500", color="#aaa"), unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_confusion_matrix(mr["cm"]), use_container_width=True)

        with col2:
            # Classification metrics table
            rpt = mr["rf_report"]
            metrics_df = pd.DataFrame({
                "Class": ["Negative (not early readmit)", "Positive (early readmit <30d)"],
                "Precision": [
                    round(rpt.get("0", {}).get("precision", 0), 3),
                    round(rpt.get("1", {}).get("precision", 0), 3),
                ],
                "Recall": [
                    round(rpt.get("0", {}).get("recall", 0), 3),
                    round(rpt.get("1", {}).get("recall", 0), 3),
                ],
                "F1-Score": [
                    round(rpt.get("0", {}).get("f1-score", 0), 3),
                    round(rpt.get("1", {}).get("f1-score", 0), 3),
                ],
                "Support": [
                    int(rpt.get("0", {}).get("support", 0)),
                    int(rpt.get("1", {}).get("support", 0)),
                ],
            })
            st.markdown("#### Classification Report (Test Set)")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            macro_f1 = round(rpt.get("macro avg", {}).get("f1-score", 0), 3)
            weighted_f1 = round(rpt.get("weighted avg", {}).get("f1-score", 0), 3)
            st.markdown(
                f"**Macro F1:** {macro_f1} · **Weighted F1:** {weighted_f1} · "
                f"**Majority-class baseline F1 (positive class):** {mr['baseline_f1']:.3f}"
            )

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_roc(mr["fpr"], mr["tpr"], mr["rf_auc"]), use_container_width=True)
        with col4:
            st.plotly_chart(plot_pr_curve(mr["prec_curve"], mr["rec_curve"]), use_container_width=True)

        st.plotly_chart(plot_feature_importance(mr["importances"]), use_container_width=True)

        st.markdown(
            "<div class='caveat-box'>"
            "Feature importances (Mean Decrease in Impurity) indicate which variables the model "
            "relied on most, but do not establish causal relationships. High importance for a feature "
            "means it is statistically informative, not necessarily actionable."
            "</div>",
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5 — INSIGHTS & RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("### Evidence-Based Insights & Operational Recommendations")
        st.markdown(
            "<div class='caveat-box'>"
            "All statistics below are computed directly from the dataset. "
            "These insights reflect descriptive associations and should be reviewed by clinical and "
            "operational experts before guiding policy changes."
            "</div>",
            unsafe_allow_html=True,
        )

        meta_f = {**meta}
        meta_f["early_readmit_rate"] = round(100 * df_f["is_early_readmit"].mean(), 2)
        meta_f["early_readmit_count"] = int(df_f["is_early_readmit"].sum())
        meta_f["eligible_rows"] = len(df_f)

        insights = generate_insights(df_f, meta_f, model_results)

        for ins in insights:
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>🔍 {ins['title']}</div>"
                f"<div class='insight-finding'><b>Finding:</b> {ins['finding']}</div>"
                f"<div class='insight-rec'><b>Recommendation:</b> {ins['recommendation']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Summary table of key rates
        st.markdown("#### Summary Statistics (Filtered View)")
        summary_rows = [
            ["Total eligible encounters", f"{len(df_f):,}"],
            ["Early readmissions (<30 days)", f"{int(df_f['is_early_readmit'].sum()):,}"],
            ["Early readmission rate", f"{filtered_rate}%"],
            ["Mean length of stay", f"{df_f['time_in_hospital'].mean():.2f} days"],
            ["Mean medications per encounter", f"{df_f['num_medications'].mean():.2f}"],
            ["Mean diagnoses per encounter", f"{df_f['number_diagnoses'].mean():.2f}"],
            ["Encounters on diabetes medication", f"{(df_f['diabetesMed']=='Yes').sum():,} ({100*(df_f['diabetesMed']=='Yes').mean():.1f}%)"],
            ["A1C tested", f"{(df_f['A1Cresult']!='Not Tested').sum():,} ({100*(df_f['A1Cresult']!='Not Tested').mean():.1f}%)"],
            ["Patients with ≥3 prior inpatient visits",
             f"{(df_f['number_inpatient']>=3).sum():,} ({100*(df_f['number_inpatient']>=3).mean():.1f}%)"],
        ]
        st.dataframe(
            pd.DataFrame(summary_rows, columns=["Metric", "Value"]),
            use_container_width=True, hide_index=True,
        )

        st.markdown("---")
        st.markdown(
            "<span style='font-size:12px;color:#57606a'>"
            "<b>Data source:</b> UCI ML Repository — Diabetes 130-US Hospitals (1999–2008). "
            "Strack B, et al. BioMed Research International, 2014. "
            "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 · "
            "License: CC BY 4.0. "
            "<b>Disclaimer:</b> This tool is for operational planning and research only. "
            "It does not provide clinical diagnoses, treatment recommendations, or individualized risk scores."
            "</span>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
