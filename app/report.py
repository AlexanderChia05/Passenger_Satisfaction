"""One-page passenger report as PDF (reportlab)."""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


NAVY, AMBER = colors.HexColor("#0b1220"), colors.HexColor("#c98a00")


def _fonts():
    return "Helvetica", "Helvetica-Bold"


def _table(rows, widths, font, bold):
    tb = Table([[str(c) for c in r] for r in rows], colWidths=widths)
    tb.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 0), bold), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f8")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8cfdb")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tb


def build_pdf(passenger: dict, seat: str, cabin: str, verdict: str, good: bool, p_team: float, conformal: str,
              model_rows: list, drivers: list, upgrades: list, ood_pct: float) -> bytes:
    font, bold = _fonts()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    ss = getSampleStyleSheet()
    h = ss["Title"].clone("h", textColor=NAVY, alignment=0, fontSize=18, spaceAfter=2, fontName=bold)
    h2 = ss["Heading3"].clone("h2", textColor=AMBER, spaceBefore=8, spaceAfter=4, fontName=bold)
    body = ss["BodyText"].clone("b", fontSize=9, leading=13, fontName=font)

    color = "#1a8f5c" if good else "#c0392b"
    story = [
        Paragraph("SkyPulse - Passenger Satisfaction Report", h),
        Paragraph("Generated {d}".format(d=f"{datetime.now():%Y-%m-%d %H:%M}"), body),
        Spacer(1, 6),
        Paragraph(("<b>Verdict:</b> {v} &nbsp; P(satisfied) = <b>{p:.3f}</b> &nbsp; Conformal set: <b>{c}</b> &nbsp; "
                    "Seat <b>{s}</b> ({cab})").format(v=f"<font color='{color}'><b>{verdict.upper()}</b></font>", p=p_team, c=conformal, s=seat, cab=cabin), body),
        Paragraph("<b>Data check:</b> passenger is more unusual than {p:.0f}% of calibration passengers".format(p=ood_pct)
                  + (" - <font color='#c0392b'>treat prediction with caution</font>." if ood_pct > 99 else "."), body),
        Paragraph("Passenger profile", h2),
        _table([["Field", "Value"]] + [[k, v] for k, v in passenger.items()], [70 * mm, 100 * mm], font, bold),
        Paragraph("Model predictions", h2),
        _table([["Model", "Role", "P(satisfied)", "Prediction"]] + model_rows,
               [50 * mm, 35 * mm, 35 * mm, 50 * mm], font, bold),
        Paragraph("Top drivers (XGBoost TreeSHAP, log-odds)", h2),
        _table([["Feature", "Contribution", "Effect"]] + drivers, [80 * mm, 40 * mm, 50 * mm], font, bold),
        Paragraph("Recommended service recovery (cost-aware)", h2),
    ]
    if upgrades:
        story.append(_table([["#", "Service", "Change", "Cost (RM)", "P(satisfied)"]] + upgrades,
                            [10 * mm, 70 * mm, 30 * mm, 25 * mm, 35 * mm], font, bold))
    else:
        story.append(Paragraph("No upgrade needed or none available within the search budget.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Upgrade suggestions show associations learned from survey data, not proven causes; "
                             "validate with an A/B test before rollout.", body))
    doc.build(story)
    return buf.getvalue()
