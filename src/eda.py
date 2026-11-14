"""
HeartGuard AI - Exploratory Data Analysis (EDA) & Interactive Visualization Engine
Builds interactive, clinically meaningful Plotly charts for patient population analytics,
epidemiological distributions, correlation matrices, and risk factor interactions.
"""

import sys
from pathlib import Path
from typing import Dict, Any

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import CLEAN_DATASET_FILE, THEME_COLORS, FEATURE_METADATA


def load_clean_data() -> pd.DataFrame:
    """Loads cleaned Cleveland dataset."""
    return pd.read_csv(CLEAN_DATASET_FILE)


def create_target_donut_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders an interactive donut chart showing class distribution of CAD vs Healthy cohort.
    """
    counts = df["target"].value_counts().sort_index()
    labels = ["Healthy / Low Risk (0)", "Coronary Artery Disease (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    colors_pie = [THEME_COLORS["accent_teal"], THEME_COLORS["accent_red"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors_pie, line=dict(color="#FFFFFF", width=2)),
        textinfo="label+percent+value",
        hoverinfo="label+value+percent",
        textfont=dict(size=13, color="#1E293B")
    )])

    fig.update_layout(
        title=dict(text="<b>Cohort Diagnostic Outcome Distribution</b>", font=dict(size=16, color=THEME_COLORS["primary_dark"])),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=30),
        height=350,
        paper_bgcolor="#FFFFFF"
    )
    return fig


def create_age_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Overlaid histogram comparing patient age distribution stratified by CAD outcome.
    """
    fig = px.histogram(
        df,
        x="age",
        color="target",
        barmode="overlay",
        nbins=20,
        color_discrete_map={0: THEME_COLORS["accent_teal"], 1: THEME_COLORS["accent_red"]},
        labels={"age": "Patient Age (years)", "target": "Diagnosis"},
        opacity=0.75
    )

    fig.for_each_trace(lambda t: t.update(name="Healthy" if t.name == "0" else "CAD Present"))

    fig.update_layout(
        title=dict(text="<b>Patient Age Distribution by CAD Status</b>", font=dict(size=16, color=THEME_COLORS["primary_dark"])),
        xaxis=dict(title="<b>Age (Years)</b>", gridcolor="#E2E8F0"),
        yaxis=dict(title="<b>Patient Count</b>", gridcolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=60, b=40),
        height=350
    )
    return fig


def create_chest_pain_chart(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart showing CAD prevalence across chest pain categories.
    """
    cp_labels = {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-Anginal",
        4: "Asymptomatic"
    }
    df_copy = df.copy()
    df_copy["cp_desc"] = df_copy["cp"].map(cp_labels)
    df_copy["target_desc"] = df_copy["target"].map({0: "Healthy", 1: "CAD Present"})

    ct = pd.crosstab(df_copy["cp_desc"], df_copy["target_desc"]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ct["cp_desc"],
        y=ct.get("Healthy", []),
        name="Healthy",
        marker_color=THEME_COLORS["accent_teal"]
    ))
    fig.add_trace(go.Bar(
        x=ct["cp_desc"],
        y=ct.get("CAD Present", []),
        name="CAD Present",
        marker_color=THEME_COLORS["accent_red"]
    ))

    fig.update_layout(
        title=dict(text="<b>Chest Pain Type vs CAD Prevalence</b>", font=dict(size=16, color=THEME_COLORS["primary_dark"])),
        barmode="group",
        xaxis=dict(title="<b>Chest Pain Classification</b>", gridcolor="#E2E8F0"),
        yaxis=dict(title="<b>Patient Count</b>", gridcolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=60, b=40),
        height=350
    )
    return fig


def create_chol_vs_thalach_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: Serum Cholesterol vs Peak Heart Rate (thalach),
    sized by ST Depression (oldpeak) and color-coded by target.
    """
    df_plot = df.copy()
    df_plot["Diagnosis"] = df_plot["target"].map({0: "Healthy / Low Risk", 1: "CAD Indicated"})

    fig = px.scatter(
        df_plot,
        x="chol",
        y="thalach",
        color="Diagnosis",
        size="oldpeak",
        size_max=16,
        color_discrete_map={"Healthy / Low Risk": THEME_COLORS["accent_teal"], "CAD Indicated": THEME_COLORS["accent_red"]},
        hover_data=["age", "trestbps", "cp"],
        labels={
            "chol": "Serum Cholesterol (mg/dL)",
            "thalach": "Max Heart Rate Achieved (bpm)",
            "oldpeak": "ST Depression (mm)"
        }
    )

    fig.update_layout(
        title=dict(text="<b>Serum Cholesterol vs Max Heart Rate (Sized by ST Depression)</b>", font=dict(size=16, color=THEME_COLORS["primary_dark"])),
        xaxis=dict(title="<b>Total Serum Cholesterol (mg/dL)</b>", gridcolor="#E2E8F0"),
        yaxis=dict(title="<b>Max Heart Rate Achieved (bpm)</b>", gridcolor="#E2E8F0"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=40),
        height=400
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Computes and plots full Pearson correlation heatmap across numerical & ordinal parameters.
    """
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "target"]
    corr = df[numeric_cols].corr()

    labels_map = {
        "age": "Age",
        "trestbps": "Resting BP",
        "chol": "Cholesterol",
        "thalach": "Max HR",
        "oldpeak": "ST Depression",
        "ca": "Major Vessels",
        "target": "CAD Status"
    }
    x_labels = [labels_map.get(c, c) for c in corr.columns]
    y_labels = [labels_map.get(c, c) for c in corr.index]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=x_labels,
        y=y_labels,
        colorscale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="<b>Pearson r</b>")
    ))

    fig.update_layout(
        title=dict(text="<b>Feature Correlation Matrix (Clinical Biomarkers vs CAD)</b>", font=dict(size=16, color=THEME_COLORS["primary_dark"])),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=50, b=40),
        height=400
    )
    return fig


def create_vessels_fluoroscopy_chart(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing disease risk proportion based on fluoroscopy vessel count (0 to 3).
    """
    df_valid = df.dropna(subset=["ca"]).copy()
    df_valid["ca_label"] = df_valid["ca"].astype(int).astype(str) + " Major Vessels"
    ct = pd.crosstab(df_valid["ca_label"], df_valid["target"], normalize="index") * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ct.index,
        y=ct[1],
        name="CAD Rate (%)",
        marker_color=THEME_COLORS["accent_coral"],
        text=[f"{v:.1f}%" for v in ct[1]],
        textposition="outside"
    ))

    fig.update_layout(
        title=dict(text="<b>CAD Prevalence by Fluoroscopy Major Vessels Colored</b>", font=dict(size=16, color=THEME_COLORS["primary_dark"])),
        xaxis=dict(title="<b>Fluoroscopy Major Vessel Count</b>", gridcolor="#E2E8F0"),
        yaxis=dict(title="<b>Disease Probability (%)</b>", range=[0, 105], gridcolor="#E2E8F0"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=50, b=40),
        height=350
    )
    return fig


if __name__ == "__main__":
    df = load_clean_data()
    print("EDA Charts Engine Initialized Successfully!")
    print(f"Data shape: {df.shape}")
