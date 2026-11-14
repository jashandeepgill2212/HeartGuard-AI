"""
HeartGuard AI - Clinical Report Generator Module
Generates professional, publication-grade downloadable PDF clinical risk assessment reports
using ReportLab, including patient metrics, diagnostic prediction, SHAP risk drivers,
clinical recommendations, and academic disclaimers.
"""

import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)

from src.config import FEATURE_METADATA


def generate_pdf_clinical_report(
    patient_info: Dict[str, Any],
    prediction_result: Dict[str, Any],
    xai_explanation: Dict[str, Any]
) -> bytes:
    """
    Creates a styled, multi-section clinical PDF report for a patient prediction.
    Returns raw PDF bytes for Streamlit download button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    c_primary = colors.HexColor("#0B192C")
    c_secondary = colors.HexColor("#1E3E62")
    c_teal = colors.HexColor("#0D9488")
    c_red = colors.HexColor("#E63946")
    c_bg = colors.HexColor("#F8FAFC")
    c_dark = colors.HexColor("#1E293B")
    c_muted = colors.HexColor("#64748B")

    # Typography Styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=c_muted,
        spaceAfter=12
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=c_dark
    )

    body_bold = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=c_dark
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=c_muted
    )

    story = []

    # --- Header Banner ---
    header_data = [
        [
            Paragraph("<b>HEARTGUARD AI CLINICAL ASSESSMENT REPORT</b>", title_style),
            Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/><b>Status:</b> Completed", body_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(header_table)
    story.append(Paragraph("Intelligent Heart Disease Prediction & Risk Analysis System | Clinical Decision Support Prototype", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_teal, spaceAfter=10))

    # --- Patient Summary & Model Outcome ---
    is_positive = prediction_result.get("prediction", 0) == 1
    badge_color = c_red if is_positive else c_teal
    outcome_label = prediction_result.get("prediction_label", "N/A")
    risk_tier = prediction_result.get("risk_level", "N/A")
    prob_disease = prediction_result.get("probability_disease", 0.0) * 100
    confidence = prediction_result.get("confidence_score", 0.0)

    summary_box_data = [
        [
            Paragraph("<b>ASSESSMENT OUTCOME</b>", body_bold),
            Paragraph("<b>ESTIMATED DISEASE PROBABILITY</b>", body_bold),
            Paragraph("<b>RISK STRATIFICATION</b>", body_bold)
        ],
        [
            Paragraph(f"<font color='{badge_color.hexval()}'><b>{outcome_label}</b></font>", section_heading),
            Paragraph(f"<font color='{badge_color.hexval()}'><b>{prob_disease:.1f}%</b></font><br/>Confidence: {confidence:.1f}%", section_heading),
            Paragraph(f"<b>{risk_tier}</b><br/>{prediction_result.get('clinical_urgency', '')}", body_style)
        ]
    ]
    summary_table = Table(summary_box_data, colWidths=[200, 170, 170])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), c_bg),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # --- Patient Demographics & Physiological Inventory ---
    story.append(Paragraph("1. Physiological Baseline & Diagnostic Parameters", section_heading))

    table_data = [
        [
            Paragraph("<b>Parameter</b>", body_bold),
            Paragraph("<b>Patient Value</b>", body_bold),
            Paragraph("<b>Reference / Standard</b>", body_bold),
            Paragraph("<b>Clinical Relevance</b>", body_bold)
        ]
    ]

    # Clinical interpretation mapping
    rows_def = [
        ("age", f"{patient_info.get('age', '')} years", "Age < 45 (M) / < 55 (F)", "Chronological vascular aging"),
        ("sex", "Male" if patient_info.get("sex") == 1 else "Female", "Gender risk index", "Biological sex predisposition"),
        ("cp", f"Type {patient_info.get('cp', '')}", "Type 1-3 vs 4 (Asymptomatic)", "Symptom presentation morphology"),
        ("trestbps", f"{patient_info.get('trestbps', '')} mm Hg", "< 120 mm Hg (Normal)", "Systemic arterial afterload"),
        ("chol", f"{patient_info.get('chol', '')} mg/dL", "< 200 mg/dL (Desirable)", "Circulating atherogenic lipids"),
        ("fbs", "> 120 mg/dL" if patient_info.get("fbs") == 1 else "<= 120 mg/dL", "<= 120 mg/dL", "Glycemic metabolic control"),
        ("restecg", f"Class {patient_info.get('restecg', '')}", "0 (Normal Sinus)", "Resting electrical conduction"),
        ("thalach", f"{patient_info.get('thalach', '')} bpm", ">= 85% Age-predicted max", "Peak chronotropic reserve"),
        ("exang", "Yes" if patient_info.get("exang") == 1 else "No", "No exertional angina", "Exercise ischemia provocation"),
        ("oldpeak", f"{patient_info.get('oldpeak', '')} mm", "< 1.0 mm (Normal)", "Exercise ST segment depression"),
        ("slope", f"Class {patient_info.get('slope', '')}", "1 (Upsloping)", "Maximal workload ST slope"),
        ("ca", f"{patient_info.get('ca', '')} vessels", "0 major calcified vessels", "Fluoroscopic vascular calcification"),
        ("thal", f"Status {patient_info.get('thal', '')}", "3 (Normal Perfusion)", "Nuclear myocardial scintigraphy")
    ]

    for param_key, val_str, ref_str, rel_str in rows_def:
        meta = FEATURE_METADATA.get(param_key, {})
        name = meta.get("name", param_key.upper())
        table_data.append([
            Paragraph(name, body_style),
            Paragraph(f"<b>{val_str}</b>", body_style),
            Paragraph(ref_str, body_style),
            Paragraph(rel_str, body_style)
        ])

    phys_table = Table(table_data, colWidths=[140, 110, 140, 150])
    phys_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story.append(phys_table)
    story.append(Spacer(1, 10))

    # --- Explainable AI (SHAP) Attribution ---
    story.append(Paragraph("2. Explainable AI (SHAP) Factor Attribution", section_heading))
    story.append(Paragraph(
        "SHapley Additive exPlanations quantify how specific individual parameters adjusted the baseline probability "
        f"(Cohort Base: {xai_explanation.get('base_value', 0.46):.2f}) toward the patient's predicted outcome.",
        body_style
    ))
    story.append(Spacer(1, 4))

    drivers = xai_explanation.get("positive_risk_drivers", [])[:3]
    protects = xai_explanation.get("protective_factors", [])[:3]

    xai_data = [
        [
            Paragraph("<b>Primary Risk Drivers (+ SHAP)</b>", body_bold),
            Paragraph("<b>Protective Factors (- SHAP)</b>", body_bold)
        ]
    ]

    driver_text = "<br/>".join([f"• <b>{d.get('display_name', '')}</b> (+{d.get('shap_value', 0.0):.3f})" for d in drivers]) if drivers else "None prominent"
    protect_text = "<br/>".join([f"• <b>{p.get('display_name', '')}</b> ({p.get('shap_value', 0.0):.3f})" for p in protects]) if protects else "None prominent"

    xai_data.append([
        Paragraph(driver_text, body_style),
        Paragraph(protect_text, body_style)
    ])

    xai_table = Table(xai_data, colWidths=[270, 270])
    xai_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FFF1F2")),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#F0FDF4")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))
    story.append(xai_table)
    story.append(Spacer(1, 10))

    # --- Evidence-Based Recommendations ---
    story.append(Paragraph("3. Clinical Guidance & Follow-up Protocols (AHA/ACC / ESC)", section_heading))
    if is_positive:
        recs = [
            "<b>Urgent Specialist Evaluation:</b> Recommend prompt cardiology referral for formal clinical assessment.",
            "<b>Confirmatory Angiography / Imaging:</b> Consider coronary CT angiography (CCTA) or invasive catheterization.",
            "<b>Lipid & Blood Pressure Management:</b> Target LDL-C < 70 mg/dL with high-intensity statin therapy; maintain BP < 130/80 mmHg.",
            "<b>Metabolic Optimization:</b> Comprehensive diabetic glycemic evaluation (HbA1c) and dietary lifestyle modification."
        ]
    else:
        recs = [
            "<b>Primary Prevention:</b> Maintain aerobic cardiovascular exercise (at least 150 minutes/week moderate activity).",
            "<b>Dietary Adherence:</b> Follow Mediterranean or DASH dietary patterns rich in whole grains and healthy fats.",
            "<b>Biomarker Surveillance:</b> Annual lipid profile, fasting plasma glucose, and resting blood pressure monitoring.",
            "<b>Vascular Risk Counseling:</b> Abstain from all tobacco and maintain ideal BMI (18.5 - 24.9 kg/m²)."
        ]

    for r in recs:
        story.append(Paragraph(f"• {r}", body_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94A3B8"), spaceAfter=6))

    # --- Disclaimers ---
    story.append(Paragraph("<b>ACADEMIC & LEGAL DISCLAIMER:</b>", body_bold))
    story.append(Paragraph(
        "HeartGuard AI is an educational and computational research tool developed for B.Tech Capstone evaluation. "
        "It is NOT a medical device, nor does it constitute diagnostic software or clinical medical advice. "
        "All predictions and SHAP attributions are purely statistical models trained on the UCI Heart Disease cohort. "
        "Under no circumstances should clinical therapies, interventions, or medication adjustments be initiated "
        "without independent verification and direction from a board-certified physician.",
        disclaimer_style
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


if __name__ == "__main__":
    sample_pt = {
        "age": 62, "sex": 1, "cp": 4, "trestbps": 158, "chol": 284,
        "fbs": 1, "restecg": 1, "thalach": 118, "exang": 1, "oldpeak": 2.6,
        "slope": 2, "ca": 2, "thal": 7
    }
    sample_pred = {
        "prediction": 1,
        "prediction_label": "Heart Disease Indicated",
        "probability_disease": 0.981,
        "confidence_score": 98.1,
        "risk_level": "High / Critical Risk",
        "clinical_urgency": "Urgent Diagnostic Angiography / Comprehensive Workup"
    }
    sample_xai = {
        "base_value": 0.46,
        "positive_risk_drivers": [{"display_name": "Major Vessels Colored: 2", "shap_value": 0.08}],
        "protective_factors": [{"display_name": "Biological Sex: Female", "shap_value": -0.02}]
    }
    pdf_out = generate_pdf_clinical_report(sample_pt, sample_pred, sample_xai)
    print(f"Generated PDF clinical report: {len(pdf_out)} bytes")
