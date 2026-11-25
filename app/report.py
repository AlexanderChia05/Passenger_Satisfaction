"""One-page passenger report as PDF (reportlab).

Print-friendly version of the app's look: white page, graphite ink, hairline tables, sentence case,
and the runway amber only as a thin rule under the title.
"""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK = colors.HexColor("#1b2028")
INK_2 = colors.HexColor("#5b6472")
LINE = colors.HexColor("#d9dde3")
HEAD_BG = colors.HexColor("#f1f3f6")
AMBER = colors.HexColor("#c8921f")
GOOD, BAD = "#1f8a5b", "#b8433a"


def _fonts():
    """Regular and bold font names for the PDF."""
    return "Helvetica", "Helvetica-Bold"


def _table(rows, widths, font, bold, num_cols=()):
    """Hairline table: light header row, no zebra stripes, numeric columns right-aligned."""
    tb = Table([[str(c) for c in r] for r in rows], colWidths=widths, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("TEXTCOLOR", (0, 0), (-1, 0), INK_2),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("FONTNAME", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 0), bold), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    style += [("ALIGN", (c, 0), (c, -1), "RIGHT") for c in num_cols]
    tb.setStyle(TableStyle(style))
    return tb


def build_pdf(passenger: dict, seat: str, cabin: str, verdict: str, good: bool, p_team: float, conformal: str,
              model_rows: list, drivers: list, upgrades: list, ood_pct: float) -> bytes:
    """One-page passenger report (verdict, conformal set, model votes, drivers and the
    cheapest service-recovery plan) returned as PDF bytes.
    """
    font, bold = _fonts()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="SkyPulse passenger report")
    ss = getSampleStyleSheet()
    h = ss["Title"].clone("h", textColor=INK, alignment=0, fontSize=17, leading=21, spaceAfter=2, fontName=bold)
    h2 = ss["Heading3"].clone("h2", textColor=INK, fontSize=10.5, spaceBefore=10, spaceAfter=4, fontName=bold)
    body = ss["BodyText"].clone("b", fontSize=9, leading=13, fontName=font, textColor=INK)
    meta = body.clone("m", textColor=INK_2, fontSize=8.5)

    color = GOOD if good else BAD
    # agreement of the member models with the headline verdict (rows: name, role, p, prediction)
    members = [r for r in model_rows if str(r[1]).startswith("Member")]
    agree = sum(1 for r in members if (r[3] == "Satisfied") == good)
    story = [
        Paragraph("SkyPulse passenger report", h),
        Paragraph("Generated {d} · seat {s}, {cab}".format(d=f"{datetime.now():%Y-%m-%d %H:%M}", s=seat, cab=cabin), meta),
        HRFlowable(width="100%", thickness=1.2, color=AMBER, spaceBefore=6, spaceAfter=8),
        Paragraph(("Verdict (M4 XGBoost): <font color='{c}'><b>{v}</b></font> &nbsp;·&nbsp; P(satisfied) <b>{p:.3f}</b> "
                   "&nbsp;·&nbsp; conformal set <b>{cs}</b>").format(c=color, v=verdict, p=p_team, cs=conformal), body),
        Paragraph("{a} of {n} member models give the same verdict.".format(a=agree, n=len(members))
                  + (" <font color='{b}'>The members disagree: review before acting.</font>".format(b=BAD)
                     if agree < len(members) else ""), body),
        Paragraph("Data check: this passenger is more unusual than {p:.0f}% of calibration passengers".format(p=ood_pct)
                  + (" - <font color='{b}'>treat the prediction with caution</font>.".format(b=BAD) if ood_pct > 99 else "."),
                  body),
        Paragraph("Passenger profile", h2),
        _table([["Field", "Value"]] + [[k, v] for k, v in passenger.items()], [70 * mm, 100 * mm], font, bold),
        Paragraph("Model predictions", h2),
        _table([["Model", "Role", "P(satisfied)", "Prediction"]] + model_rows,
               [50 * mm, 35 * mm, 35 * mm, 50 * mm], font, bold, num_cols=(2,)),
        Paragraph("Top drivers (XGBoost TreeSHAP, log-odds)", h2),
        _table([["Feature", "Contribution", "Effect"]] + drivers, [80 * mm, 40 * mm, 50 * mm], font, bold, num_cols=(1,)),
        Paragraph("Recommended service recovery (cost-aware)", h2),
    ]
    if upgrades:
        story.append(_table([["#", "Service", "Change", "Cost (RM)", "P(satisfied)"]] + upgrades,
                            [10 * mm, 70 * mm, 30 * mm, 25 * mm, 35 * mm], font, bold, num_cols=(0, 3)))
    else:
        story.append(Paragraph("No upgrade needed, or none available within the search budget.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Upgrade suggestions show associations learned from survey data, not proven causes; "
                           "validate with an A/B test before rollout.", meta))
    doc.build(story)
    return buf.getvalue()
