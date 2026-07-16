"""Export a completed audit report (the dict returned by run_full_audit) to a
polished, presentable PDF using reportlab/Platypus."""
from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

SEVERITY_COLORS = {
    "good": colors.HexColor("#1a7f37"),
    "warning": colors.HexColor("#9a6700"),
    "critical": colors.HexColor("#cf222e"),
}
SEVERITY_LABEL = {"good": "OK", "warning": "WARNING", "critical": "CRITICAL"}


def export_report_pdf(report: dict, output_path: str) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="FindingText", parent=styles["Normal"], fontSize=9.5, leading=13))
    story = []

    story.append(Paragraph("SEO & Website Health Report", styles["Title"]))
    story.append(Paragraph(report.get("url", ""), styles["Heading3"]))
    story.append(Spacer(1, 10))

    score = report.get("overall_score", 0)
    grade = report.get("grade", "")
    score_table = Table(
        [[f"Overall Score: {score}/100", f"Grade: {grade}"]],
        colWidths=[3.2 * inch, 3.2 * inch],
    )
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f3f6")),
        ("FONTSIZE", (0, 0), (-1, -1), 13),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 14))

    trend = report.get("trend")
    if trend:
        delta = trend.get("score_delta", 0)
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
        story.append(Paragraph(
            f"<b>Trend:</b> {arrow} {delta:+g} points since last audit "
            f"(previous score: {trend.get('previous_score')})",
            styles["Normal"],
        ))
        story.append(Spacer(1, 10))

    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(report.get("summary", ""), styles["Normal"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Category Breakdown", styles["Heading2"]))
    for cat in report.get("categories", []):
        story.append(Paragraph(
            f"{cat['name']} &mdash; {cat['score']}/100 (weight {cat['weight']})",
            styles["Heading3"],
        ))
        rows = [["Severity", "Issue", "Recommendation"]]
        for f in cat.get("findings", []):
            sev = f.get("severity", "")
            rows.append([
                Paragraph(f'<font color="{SEVERITY_COLORS.get(sev, colors.black)}"><b>{SEVERITY_LABEL.get(sev, sev)}</b></font>', styles["FindingText"]),
                Paragraph(f.get("issue", ""), styles["FindingText"]),
                Paragraph(f.get("recommendation", ""), styles["FindingText"]),
            ])
        if len(rows) > 1:
            t = Table(rows, colWidths=[0.9 * inch, 2.6 * inch, 3.0 * inch], repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe3e8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d1d9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t)
        story.append(Spacer(1, 12))

    quick_wins = report.get("quick_wins", [])
    if quick_wins:
        story.append(Paragraph("Quick Wins", styles["Heading2"]))
        for qw in quick_wins:
            story.append(Paragraph(f"• {qw}", styles["Normal"]))
        story.append(Spacer(1, 12))

    if report.get("data_limitations"):
        story.append(Paragraph("Data Limitations", styles["Heading2"]))
        story.append(Paragraph(report["data_limitations"], styles["Normal"]))

    doc.build(story)
    return output_path
