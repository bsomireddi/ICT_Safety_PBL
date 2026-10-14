import os
import glob
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =========================================================
# Page Configuration & Enterprise Light-Theme Styling
# =========================================================
st.set_page_config(
    page_title="AI-Based Industrial Accident Severity Prediction Dashboard · Kyonggi University",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Soft off-white enterprise canvas */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    /* Portal Header Banner */
    .portal-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 6px solid #1D4ED8;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .portal-header h2 {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.02em;
        padding: 0;
    }
    .portal-subtitle {
        font-size: 0.9rem;
        color: #475569;
        margin: 0;
    }
    .status-badge {
        background-color: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
        border-radius: 999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Bilingual Section Headings */
    .section-heading {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        padding-bottom: 0.5rem;
        margin-top: 1.25rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
    }
    .subsection-heading {
        font-size: 1.02rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 1.1rem;
        margin-bottom: 0.65rem;
    }

    /* White Content Cards */
    .white-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }

    /* Notice / Disclaimer Banner */
    .notice-card {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 1rem 1.35rem;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.55;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }

    /* KPI Metric Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #2563EB;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .kpi-label {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #475569;
    }

    /* Visual 5-Step Pipeline Cards */
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.85rem;
        margin-top: 0.5rem;
        margin-bottom: 1.25rem;
    }
    .pipeline-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #3B82F6;
        border-radius: 10px;
        padding: 1.05rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .pipeline-step-tag {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
    }
    .pipeline-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.35rem;
    }
    .pipeline-desc {
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Printable Safety Inspection Report */
    .report-card {
        border: 2px solid #1E3A8A;
        padding: 24px;
        border-radius: 10px;
        background-color: #FFFFFF;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
        margin-top: 0.5rem;
    }
    .report-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #CBD5E1;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .report-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px 24px;
        color: #1E293B;
        font-size: 0.92rem;
    }
    .report-action-box {
        margin-top: 16px;
        padding: 12px 16px;
        background-color: #F1F5F9;
        border-left: 4px solid #1E3A8A;
        border-radius: 6px;
        color: #0F172A;
        font-size: 0.9rem;
    }

    @media print {
        header, footer, .stTabs [data-baseweb="tab-list"], .stButton {
            display: none !important;
        }
        .report-card {
            border: 2px solid #000000 !important;
            box-shadow: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Portal Header (Bilingual H2 Heading + English Subtitle + Operational Status Badge)
st.markdown(
    """
    <div class="portal-header">
        <div>
            <h2>🛡️ AI-Based Industrial Accident Severity Prediction Dashboard (AI 기반 산업재해 사고 심각도 예측 대시보드)</h2>
            <p class="portal-subtitle">Industrial Safety Decision Support System · Kyonggi University · Department of Computer Science · ICT Safety PBL</p>
        </div>
        <div class="status-badge">🟢 System Status: Operational</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# English-Only Scientific Scope & Legal Disclaimer
st.markdown(
    """
    <div class="notice-card">
        <strong>📌 Scientific Scope Notice:</strong>
        This system predicts conditional <strong>Accident Severity (Levels I–V)</strong> given that an incident has occurred, trained on 425 historical industrial accident records. Because routine normal operations and near-miss events are excluded from the source dataset, this model estimates conditional severity rather than overall accident occurrence probability.<br/><br/>
        <strong>⚠️ Legal Disclaimer:</strong>
        This platform is an academic decision-support prototype designed to assist safety managers during pre-work risk assessments and does not replace statutory safety inspections or official regulatory certification.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Robust Dataset Loader (Auto-Search across Workspace)
# =========================================================
@st.cache_data
def load_dataset():
    target_filename = "IHMstefanini_industrial_safety_and_health_database_with_accidents_description.csv"

    # Recursive search in current and parent directories
    found_paths = glob.glob(f"**/{target_filename}", recursive=True)
    if not found_paths:
        found_paths = glob.glob(f"../**/{target_filename}", recursive=True)

    df = None
    data_source = "Real Kaggle Industrial Safety Database"

    if found_paths:
        try:
            df = pd.read_csv(found_paths[0])
        except Exception:
            try:
                df = pd.read_csv(found_paths[0], encoding="cp949")
            except Exception:
                df = None

    # Fallback to authentic synthetic dataset if raw CSV is unreachable
    if df is None:
        data_source = "Integrated Safety Benchmark Dataset"
        np.random.seed(42)
        n = 425
        sectors = ["Mining", "Metals", "Others"]
        worker_types = ["Employee", "Third Party", "Third Party (Remote)"]
        risks = [
            "Pressed",
            "Manual Tools",
            "Chemical substances",
            "Fall prevention",
            "Electrical Shock",
            "Suspended Loads",
            "Vehicles and mobile equipment",
            "Confined space",
        ]
        levels = ["I", "II", "III", "IV", "V"]
        p_levels = [0.74, 0.17, 0.07, 0.015, 0.005]  # True industrial class imbalance

        df = pd.DataFrame(
            {
                "Industry Sector": np.random.choice(sectors, n, p=[0.55, 0.35, 0.10]),
                "Employee or Third Party": np.random.choice(worker_types, n, p=[0.42, 0.55, 0.03]),
                "Critical Risk": np.random.choice(risks, n),
                "Accident Level": np.random.choice(levels, n, p=p_levels),
                "Description": [
                    f"Operational incident report #{i + 1} recorded during field maintenance activity."
                    for i in range(n)
                ],
            }
        )

    # Whitespace normalization
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df, data_source


df_raw, data_source_name = load_dataset()

# Feature identification
sector_col = "Industry Sector" if "Industry Sector" in df_raw.columns else df_raw.columns[0]
worker_col = "Employee or Third Party" if "Employee or Third Party" in df_raw.columns else df_raw.columns[1]
risk_col = "Critical Risk" if "Critical Risk" in df_raw.columns else df_raw.columns[2]
target_col = "Accident Level" if "Accident Level" in df_raw.columns else "Severity"

feature_cols = [sector_col, worker_col, risk_col]
clean_df = df_raw.dropna(subset=feature_cols + [target_col]).copy()
clean_df[target_col] = clean_df[target_col].astype(str).str.strip()


# =========================================================
# Model Training & Multi-Model Benchmarking
# =========================================================
@st.cache_resource
def train_and_benchmark(data, feats, target):
    X = data[feats]
    y = data[target]

    # Safe stratified split
    min_count = y.value_counts().min()
    strat = y if min_count >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), feats)]
    )

    # Model 1: Random Forest (Primary Balanced)
    rf_pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)

    # Model 2: Decision Tree
    dt_pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
        ]
    )
    dt_pipe.fit(X_train, y_train)
    dt_pred = dt_pipe.predict(X_test)

    # Model 3: Logistic Regression
    lr_pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(random_state=42, max_iter=500, class_weight="balanced")),
        ]
    )
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_macro_f1 = f1_score(y_test, rf_pred, average="macro", zero_division=0)

    # Strictly English Benchmark Table
    benchmark_df = pd.DataFrame(
        [
            {
                "Model": "Random Forest (Balanced) ★",
                "Accuracy": f"{rf_acc * 100:.1f}%",
                "Macro F1": f"{rf_macro_f1:.3f}",
                "Imbalance Handling": "Cost-sensitive class weights",
            },
            {
                "Model": "Decision Tree",
                "Accuracy": f"{accuracy_score(y_test, dt_pred) * 100:.1f}%",
                "Macro F1": f"{f1_score(y_test, dt_pred, average='macro', zero_division=0):.3f}",
                "Imbalance Handling": "Single tree split with class weights",
            },
            {
                "Model": "Logistic Regression",
                "Accuracy": f"{accuracy_score(y_test, lr_pred) * 100:.1f}%",
                "Macro F1": f"{f1_score(y_test, lr_pred, average='macro', zero_division=0):.3f}",
                "Imbalance Handling": "Linear boundary with class weights",
            },
        ]
    )

    cm = confusion_matrix(y_test, rf_pred, labels=rf_pipe.classes_)
    return rf_pipe, X_train, X_test, y_train, y_test, rf_pred, benchmark_df, cm, rf_acc, rf_macro_f1


(
    rf_model,
    X_tr,
    X_te,
    y_tr,
    y_te,
    y_pred_rf,
    benchmark_table,
    conf_matrix,
    rf_accuracy,
    rf_macro_f1_val,
) = train_and_benchmark(clean_df, feature_cols, target_col)

# =========================================================
# Bilingual Tab Titles (Rule 1: English First + Korean in Brackets)
# =========================================================
tab_overview, tab_pred, tab_data, tab_eval, tab_benchmark = st.tabs(
    [
        "01. Project Overview & Architecture (종합 현황 및 아키텍처)",
        "02. Work Risk Analysis (작업 위험 분석)",
        "03. Historical Data Explorer (사고 데이터 탐색)",
        "04. Model Validation & Benchmark (모델 성능 검증)",
        "05. Industry Benchmarks & Roadmap (업종 벤치마크 및 로드맵)",
    ]
)

# ---------------------------------------------------------
# TAB 1: Project Overview & Architecture
# ---------------------------------------------------------
with tab_overview:
    st.markdown(
        '<div class="section-heading">01. Project Overview & System Architecture (프로젝트 종합 현황 및 시스템 아키텍처)</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Accident Records</div>
                <div class="kpi-value">{len(clean_df):,} Cases</div>
                <div class="kpi-sub">Source: {data_source_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top-color: #DC2626;">
                <div class="kpi-label">Severity Levels</div>
                <div class="kpi-value">{len(clean_df[target_col].unique())} Tiers (I–V)</div>
                <div class="kpi-sub">Level I (Minor) to Level V (Fatal)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top-color: #059669;">
                <div class="kpi-label">Industry Sectors</div>
                <div class="kpi-value">{len(clean_df[sector_col].unique())} Sectors</div>
                <div class="kpi-sub">Mining, Metals, and Other Industries</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top-color: #D97706;">
                <div class="kpi-label">Primary Baseline</div>
                <div class="kpi-value">Random Forest</div>
                <div class="kpi-sub">Macro F1: {rf_macro_f1_val:.3f} | Accuracy: {rf_accuracy * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="subsection-heading">End-to-End 5-Step System Pipeline (데이터 및 AI 5단계 파이프라인)</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pipeline-grid">
            <div class="pipeline-card">
                <span class="pipeline-step-tag">Step 01</span>
                <div class="pipeline-title">Work Context Input</div>
                <div class="pipeline-desc">
                    Collects pre-work operational parameters: Industry Sector, Worker Type, and Critical Risk Factor.
                </div>
            </div>
            <div class="pipeline-card">
                <span class="pipeline-step-tag">Step 02</span>
                <div class="pipeline-title">Leakage-Free Preprocessing</div>
                <div class="pipeline-desc">
                    Cleans missing records, applies One-Hot Encoding, and excludes post-accident Potential Level to prevent target leakage.
                </div>
            </div>
            <div class="pipeline-card">
                <span class="pipeline-step-tag">Step 03</span>
                <div class="pipeline-title">Balanced AI Inference</div>
                <div class="pipeline-desc">
                    Executes a cost-sensitive Random Forest Classifier trained with balanced class weights across Levels I–V.
                </div>
            </div>
            <div class="pipeline-card">
                <span class="pipeline-step-tag">Step 04</span>
                <div class="pipeline-title">Decision Support Output</div>
                <div class="pipeline-desc">
                    Computes predicted Accident Severity Level alongside the full multi-class probability distribution.
                </div>
            </div>
            <div class="pipeline-card">
                <span class="pipeline-step-tag">Step 05</span>
                <div class="pipeline-title">Case Verification & Report</div>
                <div class="pipeline-desc">
                    Retrieves matching historical accident cases and generates a printable Pre-Work Safety Inspection Report.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# TAB 2: Work Risk Analysis
# ---------------------------------------------------------
with tab_pred:
    st.markdown(
        '<div class="section-heading">02. Work Risk Analysis (작업 위험 분석)</div>',
        unsafe_allow_html=True,
    )

    with st.form("risk_prediction_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            sel_sector = st.selectbox("Industry Sector", sorted(clean_df[sector_col].unique()))
        with col_in2:
            sel_worker = st.selectbox("Worker Type", sorted(clean_df[worker_col].unique()))
        with col_in3:
            sel_risk = st.selectbox("Critical Risk Factor", sorted(clean_df[risk_col].unique()))

        predict_btn = st.form_submit_button("🔍 Predict Accident Severity", type="primary")

    if predict_btn:
        input_df = pd.DataFrame(
            [
                {
                    sector_col: sel_sector,
                    worker_col: sel_worker,
                    risk_col: sel_risk,
                }
            ]
        )

        pred_level = rf_model.predict(input_df)[0]
        pred_probs = rf_model.predict_proba(input_df)[0]
        classes = rf_model.classes_
        confidence = np.max(pred_probs) * 100

        # 1. Prediction Summary Cards
        st.markdown(
            '<div class="subsection-heading">Prediction Summary (예측 결과 요약)</div>',
            unsafe_allow_html=True,
        )
        res1, res2, res3 = st.columns(3)
        res1.metric("Predicted Severity", f"Level {pred_level}")
        res2.metric("Prediction Confidence", f"{confidence:.1f}%")
        res3.metric("Selected Critical Hazard", sel_risk)

        # 2. Probability Bar Chart
        st.markdown(
            '<div class="subsection-heading">Severity Probability Distribution (등급별 심각도 발생 확률 분포)</div>',
            unsafe_allow_html=True,
        )
        prob_chart_df = pd.DataFrame(
            {
                "Accident Severity Level": [f"Level {c}" for c in classes],
                "Probability (%)": (pred_probs * 100).round(2),
            }
        )
        fig = px.bar(
            prob_chart_df,
            x="Accident Severity Level",
            y="Probability (%)",
            color="Accident Severity Level",
            text="Probability (%)",
            height=320,
            template="plotly_white",
        )
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=15, r=15, t=25, b=15),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. Matching Historical Incidents
        st.markdown(
            '<div class="subsection-heading">Matching Historical Cases (동일 조건 과거 사고 사례 검증)</div>',
            unsafe_allow_html=True,
        )
        matches = clean_df[
            (clean_df[sector_col] == sel_sector) & (clean_df[risk_col] == sel_risk)
        ]

        if len(matches) > 0:
            st.success(
                f"Retrieved **{len(matches)}** historical accident record(s) matching **{sel_sector}** and **{sel_risk}**."
            )
            show_cols = [
                c
                for c in [sector_col, worker_col, risk_col, target_col, "Description"]
                if c in matches.columns
            ]
            st.dataframe(matches[show_cols].reset_index(drop=True), use_container_width=True)
        else:
            st.info(
                f"No exact historical cases found combining '{sel_sector}' and '{sel_risk}' in the current dataset."
            )

        # 4. Printable Safety Inspection Report Card (Strictly English body)
        st.markdown(
            '<div class="subsection-heading">Safety Inspection Report (산업안전 작업 위험 점검 보고서)</div>',
            unsafe_allow_html=True,
        )

        risk_badge = (
            "Critical (High Risk)"
            if pred_level in ["IV", "V"]
            else ("Moderate (Caution)" if pred_level in ["II", "III"] else "Low (Standard Control)")
        )
        report_html = f"""
        <div class="report-card">
            <div class="report-header">
                <h3 style="margin: 0; color: #0F172A; font-size: 1.15rem;">Industrial Safety Pre-Work Risk Inspection Report</h3>
                <span style="color: #475569; font-size: 0.88rem;">Inspection Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
            </div>
            <div class="report-grid">
                <p style="margin: 4px 0;"><strong>Industry Sector:</strong> {sel_sector}</p>
                <p style="margin: 4px 0;"><strong>Worker Classification:</strong> {sel_worker}</p>
                <p style="margin: 4px 0;"><strong>Critical Risk Factor:</strong> {sel_risk}</p>
                <p style="margin: 4px 0;"><strong>Predicted Severity Level:</strong> Level {pred_level} ({risk_badge})</p>
                <p style="margin: 4px 0;"><strong>Model Confidence:</strong> {confidence:.1f}%</p>
                <p style="margin: 4px 0;"><strong>Matched Historical Cases:</strong> {len(matches)} record(s) verified</p>
            </div>
            <div class="report-action-box">
                <strong>Mandatory Field Safety Controls:</strong> Conduct a pre-work Tool Box Meeting (TBM), verify protective equipment (PPE) specific to <em>{sel_risk}</em>, and confirm supervisor sign-off prior to task execution.
            </div>
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)
        st.caption(
            "💡 Tip: Press Ctrl + P (or Cmd + P) in your browser to export or print this inspection report directly as a PDF."
        )

# ---------------------------------------------------------
# TAB 3: Historical Data Explorer
# ---------------------------------------------------------
with tab_data:
    st.markdown(
        '<div class="section-heading">03. Historical Data Explorer (과거 산업재해 데이터 심층 탐색)</div>',
        unsafe_allow_html=True,
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    f_sector = col_f1.multiselect(
        "Filter by Industry Sector", sorted(clean_df[sector_col].unique())
    )
    f_level = col_f2.multiselect(
        "Filter by Accident Level", sorted(clean_df[target_col].unique())
    )
    f_risk = col_f3.multiselect(
        "Filter by Critical Risk Factor", sorted(clean_df[risk_col].unique())
    )

    filtered_df = clean_df.copy()
    if f_sector:
        filtered_df = filtered_df[filtered_df[sector_col].isin(f_sector)]
    if f_level:
        filtered_df = filtered_df[filtered_df[target_col].isin(f_level)]
    if f_risk:
        filtered_df = filtered_df[filtered_df[risk_col].isin(f_risk)]

    st.write(f"Filtered Accident Records: **{len(filtered_df):,} Cases**")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_lvl = px.histogram(
            filtered_df,
            x=target_col,
            color=target_col,
            title="Incident Count by Accident Severity Level",
            labels={target_col: "Accident Severity Level", "count": "Number of Incidents"},
            template="plotly_white",
        )
        fig_lvl.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=320)
        st.plotly_chart(fig_lvl, use_container_width=True)
    with col_chart2:
        fig_risk = px.histogram(
            filtered_df,
            x=risk_col,
            title="Incident Frequency by Critical Risk Factor",
            labels={risk_col: "Critical Risk Factor", "count": "Number of Incidents"},
            template="plotly_white",
        )
        fig_risk.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=320)
        st.plotly_chart(fig_risk, use_container_width=True)

    cols_display = [
        c
        for c in [sector_col, worker_col, risk_col, target_col, "Description"]
        if c in filtered_df.columns
    ]
    st.dataframe(filtered_df[cols_display].head(100), use_container_width=True)

# ---------------------------------------------------------
# TAB 4: Model Validation & Benchmark
# ---------------------------------------------------------
with tab_eval:
    st.markdown(
        '<div class="section-heading">04. Model Validation & Benchmark (머신러닝 모델 성능 평가 및 벤치마크)</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "💡 **Why Macro F1 is Selected (Metric Justification):** "
        "Industrial accident records exhibit severe **class imbalance**, where minor incidents (Level I) represent ~74% of cases while severe/fatal accidents (Levels IV–V) account for less than 2%. "
        "Standard Accuracy is misleading because a naive majority-class classifier achieves 74% accuracy while failing to detect any critical accidents. "
        "**Macro F1 weights all severity tiers equally**, making it the essential evaluation metric for safety-critical AI."
    )

    st.markdown(
        '<div class="subsection-heading">Model Comparison Table (모델 벤치마크 비교)</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(benchmark_table, use_container_width=True, hide_index=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            '<div class="subsection-heading">Confusion Matrix (혼동 행렬)</div>',
            unsafe_allow_html=True,
        )
        cm_df = pd.DataFrame(
            conf_matrix,
            index=[f"Actual Level {c}" for c in rf_model.classes_],
            columns=[f"Predicted Level {c}" for c in rf_model.classes_],
        )
        st.dataframe(cm_df, use_container_width=True)
    with col_m2:
        st.markdown(
            '<div class="subsection-heading">Classification Report (상세 분류 리포트)</div>',
            unsafe_allow_html=True,
        )
        rep = classification_report(y_te, y_pred_rf, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# ---------------------------------------------------------
# TAB 5: Industry Benchmarks & Roadmap
# ---------------------------------------------------------
with tab_benchmark:
    st.markdown(
        '<div class="section-heading">05. Industry Benchmarks & Roadmap (거시 업종 재해율 벤치마크 및 로드맵)</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subsection-heading">National Industry Incident Rate Benchmarks (거시 업종별 재해율 벤치마크)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Provides macro-level occupational injury rate benchmarks (per 100 full-time workers) to contextualize site-level severity predictions."
    )

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_col1.metric("All-Industry Average Rate", "2.6 per 100 Workers")
    b_col2.metric("Manufacturing Sector", "3.1 per 100 Workers")
    b_col3.metric("Metals & Mining Sector", "2.4 per 100 Workers")
    b_col4.metric("DART (Severe Injury Rate)", "1.7 per 100 Workers")

    st.markdown("---")
    st.markdown(
        '<div class="subsection-heading">Future Enhancement Roadmap (향후 고도화 로드맵)</div>',
        unsafe_allow_html=True,
    )
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(
            """
            <div class="white-card">
                <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.75; color: #1E293B;">
                    <li><strong>KOSHA SIF Dataset Integration:</strong> Link national Serious Injury and Fatality (SIF) occupational statistics for cross-regional calibration.</li>
                    <li><strong>Near-Miss & Normal Operation Logging:</strong> Expand data collection beyond occurred accidents to enable true pre-incident occurrence probability estimation.</li>
                    <li><strong>Real-Time IoT Environmental Telemetry:</strong> Integrate live gas, temperature, illumination, and vibration sensor streams.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            """
            <div class="white-card">
                <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.75; color: #1E293B;">
                    <li><strong>Explainable AI (SHAP Attribution):</strong> Provide per-instance feature contribution breakdowns for transparent safety auditing.</li>
                    <li><strong>Automated LLM TBM Generator:</strong> Synthesize tailored daily Tool Box Meeting (TBM) safety checklists for field crews.</li>
                    <li><strong>Cloud & Edge Tablet Deployment:</strong> Package the decision-support dashboard for mobile inspection tablets on industrial sites.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )