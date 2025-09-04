# WORKING REPORTING SCRIPT

# app/core/reporting.py
import re
from datetime import datetime, timezone

# ======= BRANDING / THEME =======
BRAND_NAME = "Bauhaus Project Management"
REPORT_TITLE = f"{BRAND_NAME} — Tender Analysis Report"
ACCENT = "#3404f5"  # brand violet
STRIPE = "#F3F4F6"  # zebra rows
GRID   = "#E5E7EB"  # grid color
# =================================


# -------------------- shared helpers --------------------
def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _md_inline_to_rl(text: str) -> str:
    esc = _esc(text or "")
    esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)
    esc = re.sub(r"\*(.+?)\*", r"<i>\1</i>", esc)
    return esc

def _truncate_safe(s: str, n: int) -> str:
    """Truncate on word boundary; append ellipsis only if needed."""
    if not s: return ""
    s = s.strip()
    if len(s) <= n: return s
    cut = s[:n].rstrip()
    sp = cut.rfind(" ")
    if sp > int(n * 0.6):  # avoid chopping too early
        cut = cut[:sp]
    return cut + "…"

def _num_from_currency(s: str) -> float:
    """Extract numeric value from 'AED 6,407,718.95' or '6407718.95' strings."""
    if not s: return float("nan")
    m = re.findall(r"[\d.,]+", s.replace(",", ""))
    try:
        return float(m[0]) if m else float("nan")
    except Exception:
        try:
            return float(str(s).replace(",", ""))
        except Exception:
            return float("nan")

def _compute_value_tags(cost_items: list[dict]) -> dict:
    """
    Compute Low/Mid/High vs peer median when 'value_tag' missing.
    Returns dict: bidder -> value_tag
    """
    vals = []
    for it in cost_items or []:
        v = _num_from_currency(it.get("total_cost", ""))
        if v == v:  # not NaN
            vals.append(v)
    if not vals:
        return {}
    vals_sorted = sorted(vals)
    median = vals_sorted[len(vals_sorted)//2]
    out = {}
    for it in cost_items or []:
        b = it.get("bidder", "")
        v = _num_from_currency(it.get("total_cost", ""))
        if v == v:
            if v <= 0.95 * median:
                out[b] = "Low vs median"
            elif v >= 1.05 * median:
                out[b] = "High vs median"
            else:
                out[b] = "Mid vs median"
    return out

def _derive_risk_themes_from_analysis(risk_analysis: dict) -> dict[str, set]:
    """
    Build cross-cutting themes scanning general + by_bidder risks.
    """
    themes = {}
    def add(theme, bidder=None):
        if theme not in themes: themes[theme] = set()
        if bidder: themes[theme].add(bidder)

    if not risk_analysis: return themes

    # General
    for r in risk_analysis.get("general_project_risks", []) or []:
        s = (r or "").lower()
        if any(k in s for k in ("authority", "permit", "approval")):
            add("Authority approvals dependency")
        if any(k in s for k in ("remeasure", "boq", "variation")):
            add("Remeasurement / BOQ variation exposure")
        if any(k in s for k in ("hse", "safety", "work at height", "material handling")):
            add("HSE considerations (work at height / handling)")
        if any(k in s for k in ("interface", "coordination", "other trades", "mep")):
            add("Interfaces & coordination with other trades")

    # By bidder
    for bidder, cats in (risk_analysis.get("by_bidder") or {}).items():
        for cat_list in (cats or {}).values():
            for r in cat_list or []:
                s = (r or "").lower()
                if any(k in s for k in ("authority", "permit", "approval")):
                    add("Authority approvals dependency", bidder)
                if any(k in s for k in ("remeasure", "boq", "variation")):
                    add("Remeasurement / BOQ variation exposure", bidder)
                if any(k in s for k in ("long lead", "joinery", "lead time", "procure")):
                    add("Long-lead procurement (joinery/fixtures)", bidder)
                if any(k in s for k in ("interface", "coordination", "other trades", "mep")):
                    add("Interfaces & coordination with other trades", bidder)
                if any(k in s for k in ("resource", "capacity", "availability")):
                    add("Contractor resourcing / availability", bidder)
                if any(k in s for k in ("hse", "safety", "work at height", "lifting")):
                    add("HSE considerations (work at height / handling)", bidder)
    return themes


# -------------------- On-page (Streamlit) markdown --------------------
def build_markdown(data: dict) -> str:
    """
    Markdown for comprehensive tender analysis report aligned with the new schema.
    """
    md = []
    dt = datetime.now(timezone.utc).strftime('%B %d, %Y at %H:%M')
    md.append("# Comprehensive Tender Analysis Report")
    md.append(f"**Generated:** {dt} UTC\n")
    md.append("---\n")

    # Executive Summary
    md.append("## Executive Summary")
    md.append(data.get("executive_summary", ""))

    # Comparison Snapshot
    snap = data.get("comparison_snapshot", [])
    if snap:
        md.append("\n## Bidder Snapshot")
        md.append("| Bidder | Technical Fit | Team |")
        md.append("| --- | --- | --- |")
        for r in snap:
            md.append(f"| {r.get('bidder','')} | {r.get('technical_fit','')} | {r.get('team','')} |")

    # Technical Analysis (brief table)
    tech = data.get("technical_analysis", {}) or {}
    tech_rows = tech.get("detailed_assessments", []) or []
    if tech_rows:
        md.append("\n## Technical Scores")
        md.append("| Bidder | Technical Score | Compliance | Innovation |")
        md.append("| --- | --- | --- | --- |")
        for r in tech_rows:
            md.append(f"| {r.get('bidder','')} | {r.get('technical_score','')} | {r.get('compliance_with_specs','')} | {r.get('innovation_factor','')} |")
        
        # Add explanatory paragraph
        if tech.get("table_summary"):
            md.append(f"\n{tech.get('table_summary')}")

    # Team & Capacity (brief table)
    team = data.get("team_and_capacity_analysis", {}) or {}
    team_rows = team.get("assessments", []) or []
    if team_rows:
        md.append("\n## Team & Capacity")
        md.append("| Bidder | Team Score | PM (yrs/quals) | UAE Experience |")
        md.append("| --- | --- | --- | --- |")
        for r in team_rows:
            md.append(f"| {r.get('bidder','')} | {r.get('team_score','')} | {r.get('project_manager_credentials','')} | {r.get('relevant_experience','')} |")
        
        # Add explanatory paragraph
        if team.get("table_summary"):
            md.append(f"\n{team.get('table_summary')}")

    # Financial (brief table + value tag)
    fin = data.get("financial_analysis", {}) or {}
    fin_rows = fin.get("cost_breakdown_analysis", []) or []
    if fin_rows:
        md.append("\n## Financial Summary")
        md.append("| Bidder | Total (AED) | Financial Score | Value Tag |")
        md.append("| --- | --- | --- | --- |")
        for r in fin_rows:
            md.append(f"| {r.get('bidder','')} | {r.get('total_cost','')} | {r.get('financial_score','')} | {r.get('value_tag','')} |")
        
        # Add explanatory paragraph
        if fin.get("table_summary"):
            md.append(f"\n{fin.get('table_summary')}")

    # Timeline & Delivery (brief table)
    tl = data.get("timeline_and_delivery_analysis", {}) or {}
    tl_rows = tl.get("schedule_evaluation", []) or []
    if tl_rows:
        md.append("\n## Timeline & Delivery")
        md.append("| Bidder | Proposed Timeline | Timeline Score | Delivery Confidence |")
        md.append("| --- | --- | --- | --- |")
        for r in tl_rows:
            md.append(f"| {r.get('bidder','')} | {r.get('proposed_timeline','')} | {r.get('timeline_score','')} | {r.get('delivery_confidence','')} |")
        
        # Add explanatory paragraph
        if tl.get("table_summary"):
            md.append(f"\n{tl.get('table_summary')}")

    # Compliance (brief table)
    comp = data.get("compliance_and_regulatory_analysis", {}) or {}
    comp_rows = comp.get("assessments", []) or []
    if comp_rows:
        md.append("\n## Compliance")
        md.append("| Bidder | Compliance Score & Details |")
        md.append("| --- | --- |")
        for r in comp_rows:
            md.append(f"| {r.get('bidder','')} | {r.get('compliance_score','')} |")
        
        # Add explanatory paragraph
        if comp.get("table_summary"):
            md.append(f"\n{comp.get('table_summary')}")

    # Cross-Cutting Risk Themes
    risk = data.get("comprehensive_risk_analysis", {}) or {}
    themes = _derive_risk_themes_from_analysis(risk)
    if themes:
        md.append("\n## Cross-Cutting Risk Themes")
        for theme, bidders in sorted(themes.items()):
            suffix = f" (e.g., {', '.join(sorted(b for b in bidders if b))})" if bidders else ""
            md.append(f"- {theme}{suffix}")

    # External Risk Analysis (from NewsAPI) — move earlier, before Bidder Highlights
    ext_risk = data.get("external_risk_analysis", {}) or {}
    if ext_risk:
        md.append("\n## External Risk Analysis")
        md.append("*Based on recent news and media monitoring*")
        
        for company, risk_data in ext_risk.items():
            if isinstance(risk_data, dict):
                risk_analysis = risk_data.get('risk_analysis', '')
                sample_heads = risk_data.get('sample_headlines', [])
                
                md.append(f"\n### {company}")
                if sample_heads:
                    md.append(f"Sample headlines: {', '.join(sample_heads[:3])}")
                
                if risk_analysis:
                    md.append(f"\n{risk_analysis}")
                else:
                    md.append("\nNo adverse media or external red flags identified in recent monitoring.")
            else:
                # Legacy format - just the analysis text
                md.append(f"\n### {company}")
                md.append(f"{risk_data}")

    # Bidder Highlights (build comprehensive bullets by bidder)
    # Use enhanced technical_approach and key_personnel_quality for more detailed descriptions
    if any([tech_rows, team_rows, fin_rows, tl_rows, comp_rows]):
        md.append("\n## Bidder Highlights")
        bidders = {r.get("bidder") for r in (snap or []) if r.get("bidder")}
        # augment with any bidder seen elsewhere
        for coll in (tech_rows, team_rows, fin_rows, tl_rows, comp_rows):
            for r in coll:
                if r.get("bidder"): bidders.add(r.get("bidder"))

        def _find(lst, name): 
            return next((x for x in lst if (x or {}).get("bidder")==name), None)

        for b in sorted(bidders):
            md.append(f"### {b}")
            trow = _find(tech_rows, b) or {}
            teamr = _find(team_rows, b) or {}
            finr = _find(fin_rows, b) or {}
            tlr  = _find(tl_rows, b) or {}
            compr= _find(comp_rows, b) or {}

            bullets = []
            # Use the enhanced technical_approach for comprehensive technical description
            if trow.get("technical_approach"): 
                bullets.append(f"**Technical Approach** — {trow.get('technical_approach')}")
            
            # Use the enhanced key_personnel_quality for comprehensive team description
            if teamr.get("key_personnel_quality"): 
                bullets.append(f"**Team & Leadership** — {teamr.get('key_personnel_quality')}")
            elif teamr.get("project_manager_credentials"):
                bullets.append(f"**Project Management** — {teamr.get('project_manager_credentials')}")
            
            # Enhanced timeline description
            if tlr.get("proposed_timeline") and tlr.get("delivery_confidence"): 
                bullets.append(f"**Timeline & Delivery** — {tlr.get('proposed_timeline')} with {tlr.get('delivery_confidence')} confidence level ({tlr.get('timeline_score','')})")
            elif tlr.get("proposed_timeline"): 
                bullets.append(f"**Timeline** — {tlr.get('proposed_timeline')} ({tlr.get('timeline_score','')})")
            
            # Enhanced commercial description
            if finr.get("total_cost") and finr.get("value_for_money_assessment"): 
                bullets.append(f"**Commercial Position** — {finr.get('total_cost')} ({finr.get('value_tag','')}) - {finr.get('value_for_money_assessment')} ({finr.get('financial_score','')})")
            elif finr.get("total_cost"): 
                bullets.append(f"**Commercial** — {finr.get('total_cost')} ({finr.get('value_tag','')}, {finr.get('financial_score','')})")
            
            for s in bullets:
                md.append(f"- {s}")

    # Strategic Recommendation
    rec = data.get("strategic_recommendation", {}) or {}
    md.append("\n## Strategic Recommendation")
    if rec.get("winner"):
        md.append(f"**Recommended Bidder:** {rec.get('winner')}")
    if rec.get("comprehensive_rationale"):
        md.append("\n### Strategic Rationale")
        md.append(rec["comprehensive_rationale"])

    # Appendix (Implementation Roadmap + KPIs)
    appx = data.get("implementation_roadmap", {}) or {}
    if any(appx.values()) or rec.get("kpis_and_monitoring"):
        md.append("\n## Appendix")
        if appx.get("immediate_actions"):
            md.append("### Implementation — Immediate Actions (30 days)")
            for a in appx["immediate_actions"][:6]:
                md.append(f"- {a}")
        if appx.get("contract_negotiation_priorities"):
            md.append("\n### Contract Negotiation Priorities")
            for a in appx["contract_negotiation_priorities"][:6]:
                md.append(f"- {a}")
        if appx.get("mobilization_requirements"):
            md.append("\n### Mobilization Requirements")
            for a in appx["mobilization_requirements"][:6]:
                md.append(f"- {a}")
        if appx.get("governance_framework"):
            md.append("\n### Governance Framework")
            for a in appx["governance_framework"][:6]:
                md.append(f"- {a}")
        if appx.get("stakeholder_communication"):
            md.append("\n### Stakeholder Communication")
            for a in appx["stakeholder_communication"][:6]:
                md.append(f"- {a}")
        if appx.get("milestone_tracking"):
            md.append("\n### Milestone Tracking")
            for a in appx["milestone_tracking"][:6]:
                md.append(f"- {a}")
        if rec.get("kpis_and_monitoring"):
            md.append("\n### KPIs & Monitoring")
            for k in (rec.get("kpis_and_monitoring") or [])[:6]:
                md.append(f"- {k}")

    return "\n".join(md)


# -------------------- Narrative PDF (board-ready, concise tables + appendix) --------------------
def build_pdf_report(data: dict) -> bytes:
    """
    Board-ready PDF with:
      - Title + timestamp (unchanged)
      - Executive Summary
      - Dashboard tables: Snapshot, Technical, Team, Financial (with Value Tag), Timeline, Compliance
      - Cross-Cutting Risk Themes
      - PageBreak
      - Bidder Highlights (bulleted)
      - Strategic Recommendation
      - Appendix (Implementation Roadmap + KPIs)
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem,
        KeepTogether, Table, TableStyle, LongTable, PageBreak
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        title=REPORT_TITLE,
    )

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, leading=22, spaceBefore=4, spaceAfter=8)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, leading=18, spaceBefore=10, spaceAfter=6, keepWithNext=True)
    H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=12, leading=16, spaceBefore=6, spaceAfter=4, keepWithNext=True)
    Body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=14.5, spaceAfter=6, wordWrap="CJK", splitLongWords=True)
    Bullet = ParagraphStyle("Bullet", parent=Body, leftIndent=14, bulletIndent=6)
    Cell  = ParagraphStyle("Cell", parent=Body, fontSize=9.2, leading=12.5, spaceAfter=2, splitLongWords=True)
    CellH = ParagraphStyle("CellH", parent=Body, textColor=colors.white, fontName="Helvetica-Bold", fontSize=9.4, leading=12.8)

    ts_footer = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    def _footer(canvas, doc_):
        canvas.saveState()
        w, h = doc_.pagesize
        canvas.setFont("Helvetica", 9)
        canvas.drawString(doc_.leftMargin, doc_.bottomMargin - 12, ts_footer)
        canvas.drawRightString(w - doc_.rightMargin, doc_.bottomMargin - 12, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    def _brand_divider():
        tbl = Table([[" "]], colWidths=[doc.width], rowHeights=[3])  # thinner, modern
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor(ACCENT)),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING", (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        return tbl

    def _make_table(rows: list[list[str]], colw=None, header_bg=ACCENT):
        """Generic styled LongTable with zebra and grid."""
        if not rows: return None
        r = [[Paragraph(_md_inline_to_rl(c), CellH if i==0 else Cell) for c in row] for i, row in enumerate(rows)]
        tbl = LongTable(r, colWidths=colw, repeatRows=1, splitByRow=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor(header_bg)),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONT",       (0,0), (-1,0), "Helvetica-Bold", 9),
            ("FONT",       (0,1), (-1,-1), "Helvetica", 9),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor(GRID)),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor(STRIPE)]),
            ("VALIGN",     (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("LEFTPADDING",(0,0),(-1,-1), 6),
            ("RIGHTPADDING",(0,0),(-1,-1), 6),
        ]))
        return tbl

    # ---------- story ----------
    story = []
    story.append(Paragraph(_md_inline_to_rl(f"{BRAND_NAME}"), H1))
    story.append(Paragraph(_md_inline_to_rl(f"Tender Analysis Report — Generated {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M')} UTC"), Body))
    story.append(_brand_divider())
    story.append(Spacer(1, 8))

    # Executive Summary
    story.append(Paragraph(_md_inline_to_rl("Executive Summary"), H2))
    story.append(Paragraph(_md_inline_to_rl(data.get("executive_summary", "")), Body))

    # ---------- Dashboards (concise tables) ----------
    # 1) Snapshot
    snap = data.get("comparison_snapshot", []) or []
    if snap:
        rows = [["Bidder", "Technical Fit", "Team"]]
        for r in snap:
            rows.append([
                _truncate_safe(r.get("bidder",""), 30),
                _truncate_safe(r.get("technical_fit",""), 240),
                _truncate_safe(r.get("team",""), 200),
            ])
        # widen Team column
        colw = [0.22*doc.width, 0.40*doc.width, 0.38*doc.width]
        story.append(Spacer(1, 6))
        story.append(Paragraph(_md_inline_to_rl("Bidder Snapshot"), H2))
        story.append(_make_table(rows, colw=colw))

    # 2) Technical Scores
    tech = data.get("technical_analysis", {}) or {}
    tech_rows = tech.get("detailed_assessments", []) or []
    if tech_rows:
        rows = [["Bidder", "Technical Score", "Compliance", "Innovation"]]
        for r in tech_rows:
            rows.append([
                _truncate_safe(r.get("bidder",""), 30),
                _truncate_safe(r.get("technical_score",""), 60),
                _truncate_safe(r.get("compliance_with_specs",""), 120),
                _truncate_safe(r.get("innovation_factor",""), 100),
            ])
        colw = [0.22*doc.width, 0.18*doc.width, 0.32*doc.width, 0.28*doc.width]
        story.append(Spacer(1, 6))
        story.append(Paragraph(_md_inline_to_rl("Technical Scores"), H2))
        story.append(_make_table(rows, colw=colw))
        
        # Add explanatory paragraph
        if tech.get("table_summary"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(_md_inline_to_rl(tech.get("table_summary")), Body))

    # 3) Team & Capacity
    team = data.get("team_and_capacity_analysis", {}) or {}
    team_rows = team.get("assessments", []) or []
    if team_rows:
        rows = [["Bidder", "Team Score", "PM (yrs/quals)", "UAE Experience"]]
        for r in team_rows:
            rows.append([
                _truncate_safe(r.get("bidder",""), 30),
                _truncate_safe(r.get("team_score",""), 60),
                _truncate_safe(r.get("project_manager_credentials",""), 120),
                _truncate_safe(r.get("relevant_experience",""), 130),
            ])
        colw = [0.22*doc.width, 0.18*doc.width, 0.30*doc.width, 0.30*doc.width]
        story.append(Spacer(1, 6))
        story.append(Paragraph(_md_inline_to_rl("Team & Capacity"), H2))
        story.append(_make_table(rows, colw=colw))
        
        # Add explanatory paragraph
        if team.get("table_summary"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(_md_inline_to_rl(team.get("table_summary")), Body))

    # 4) Financial Summary (Value Tag)
    fin = data.get("financial_analysis", {}) or {}
    fin_rows = fin.get("cost_breakdown_analysis", []) or []
    if fin_rows:
        # compute value tags if missing
        computed_tags = _compute_value_tags(fin_rows)
        rows = [["Bidder", "Total (AED)", "Financial Score", "Value Tag"]]
        for r in fin_rows:
            val_tag = r.get("value_tag","") or computed_tags.get(r.get("bidder",""), "")
            rows.append([
                _truncate_safe(r.get("bidder",""), 30),
                _truncate_safe(r.get("total_cost",""), 35),
                _truncate_safe(r.get("financial_score",""), 60),
                _truncate_safe(val_tag, 40),
            ])
        colw = [0.28*doc.width, 0.22*doc.width, 0.24*doc.width, 0.26*doc.width]
        story.append(Spacer(1, 6))
        story.append(Paragraph(_md_inline_to_rl("Financial Summary"), H2))
        story.append(_make_table(rows, colw=colw))
        
        # Add explanatory paragraph
        if fin.get("table_summary"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(_md_inline_to_rl(fin.get("table_summary")), Body))

    # 5) Timeline & Delivery
    tl = data.get("timeline_and_delivery_analysis", {}) or {}
    tl_rows = tl.get("schedule_evaluation", []) or []
    if tl_rows:
        rows = [["Bidder", "Proposed Timeline", "Timeline Score", "Delivery Confidence"]]
        for r in tl_rows:
            rows.append([
                _truncate_safe(r.get("bidder",""), 30),
                _truncate_safe(r.get("proposed_timeline",""), 120),
                _truncate_safe(r.get("timeline_score",""), 60),
                _truncate_safe(r.get("delivery_confidence",""), 120),
            ])
        colw = [0.22*doc.width, 0.34*doc.width, 0.18*doc.width, 0.26*doc.width]
        story.append(Spacer(1, 6))
        story.append(Paragraph(_md_inline_to_rl("Timeline & Delivery"), H2))
        story.append(_make_table(rows, colw=colw))
        
        # Add explanatory paragraph
        if tl.get("table_summary"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(_md_inline_to_rl(tl.get("table_summary")), Body))

    # 6) Compliance (Score & Details only)
    comp = data.get("compliance_and_regulatory_analysis", {}) or {}
    comp_rows = comp.get("assessments", []) or []
    if comp_rows:
        rows = [["Bidder", "Compliance Score & Details"]]
        for r in comp_rows:
            rows.append([
                _truncate_safe(r.get("bidder",""), 30),
                _truncate_safe(r.get("compliance_score",""), 140),
            ])
        colw = [0.28*doc.width, 0.72*doc.width]
        story.append(Spacer(1, 6))
        story.append(Paragraph(_md_inline_to_rl("Compliance Summary"), H2))
        story.append(_make_table(rows, colw=colw))
        
        # Add explanatory paragraph
        if comp.get("table_summary"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(_md_inline_to_rl(comp.get("table_summary")), Body))

    # Cross-Cutting Risk Themes
    risk = data.get("comprehensive_risk_analysis", {}) or {}
    themes = _derive_risk_themes_from_analysis(risk)
    if themes:
        story.append(Spacer(1, 6))
        story.append(Paragraph(_md_inline_to_rl("Cross-Cutting Risk Themes"), H2))
        items = []
        for theme, bidders in sorted(themes.items(), key=lambda x: x[0]):
            suffix = f" (e.g., {', '.join(sorted(b for b in bidders if b))})" if bidders else ""
            items.append(f"{theme}{suffix}")
        story.append(ListFlowable([ListItem(Paragraph(_md_inline_to_rl(t), Bullet)) for t in items], bulletType="bullet"))

    # External Risk Analysis (from NewsAPI) — move earlier, before Bidder Highlights
    ext_risk = data.get("external_risk_analysis", {}) or {}
    if ext_risk:
        story.append(Spacer(1, 8))
        story.append(Paragraph(_md_inline_to_rl("External Risk Analysis"), H2))
        story.append(Paragraph(_md_inline_to_rl("Based on recent news and media monitoring"), Body))
        story.append(Spacer(1, 6))
        for company, risk_data in ext_risk.items():
            story.append(Paragraph(_md_inline_to_rl(f"{company}"), H3))
            if isinstance(risk_data, dict):
                risk_analysis = risk_data.get('risk_analysis', '')
                sample_heads = risk_data.get('sample_headlines', [])
                if sample_heads:
                    story.append(Paragraph(_md_inline_to_rl("Sample headlines: " + ", ".join(sample_heads[:3])), Body))
                story.append(Spacer(1, 3))
                if risk_analysis:
                    story.append(Paragraph(_md_inline_to_rl(risk_analysis), Body))
                else:
                    story.append(Paragraph(_md_inline_to_rl("No adverse media or external red flags identified in recent monitoring."), Body))
            else:
                story.append(Paragraph(_md_inline_to_rl(str(risk_data)), Body))
            story.append(Spacer(1, 6))

    # ---------- Page break to avoid orphaned 'Bidder Highlights' ----------
    story.append(PageBreak())

    # Bidder Highlights (bulleted, more comprehensive)
    # Build unified bidder set
    if any([snap, tech_rows, team_rows, fin_rows, tl_rows, comp_rows]):
        story.append(Paragraph(_md_inline_to_rl("Bidder Highlights"), H2))

        bidders = {r.get("bidder") for r in (snap or []) if r.get("bidder")}
        for coll in (tech_rows, team_rows, fin_rows, tl_rows, comp_rows):
            for r in coll:
                if r.get("bidder"):
                    bidders.add(r.get("bidder"))

        def _find(lst, name):
            return next((x for x in lst if (x or {}).get("bidder")==name), None)

        for b in sorted(bidders):
            trow = _find(tech_rows, b) or {}
            teamr = _find(team_rows, b) or {}
            finr  = _find(fin_rows, b) or {}
            tlr   = _find(tl_rows, b) or {}
            compr = _find(comp_rows, b) or {}

            bullets = []
            # Use the enhanced technical_approach for comprehensive technical description
            if trow.get("technical_approach"): 
                bullets.append(f"**Technical Approach** — {trow.get('technical_approach')}")
            
            # Use the enhanced key_personnel_quality for comprehensive team description
            if teamr.get("key_personnel_quality"): 
                bullets.append(f"**Team & Leadership** — {teamr.get('key_personnel_quality')}")
            elif teamr.get("project_manager_credentials"):
                bullets.append(f"**Project Management** — {teamr.get('project_manager_credentials')}")
            
            # Enhanced timeline description
            if tlr.get("proposed_timeline") and tlr.get("delivery_confidence"): 
                bullets.append(f"**Timeline & Delivery** — {tlr.get('proposed_timeline')} with {tlr.get('delivery_confidence')} confidence level")
            elif tlr.get("proposed_timeline"): 
                bullets.append(f"**Timeline** — {tlr.get('proposed_timeline')}")
            
            # Enhanced commercial description
            if finr.get("total_cost") and finr.get("value_for_money_assessment"): 
                bullets.append(f"**Commercial Position** — {finr.get('total_cost')} - {finr.get('value_for_money_assessment')}")
            elif finr.get("total_cost"): 
                bullets.append(f"**Commercial** — {finr.get('total_cost')}")
            
            if bullets:
                blk = [Paragraph(_md_inline_to_rl(b), H3)]
                blk.append(ListFlowable([ListItem(Paragraph(_md_inline_to_rl(p), Bullet)) for p in bullets], bulletType="bullet"))
                story.append(KeepTogether(blk))
            story.append(Spacer(1, 6))

    # Strategic Recommendation (last in the main body)
    rec = data.get("strategic_recommendation", {}) or {}
    story.append(Spacer(1, 6))
    story.append(Paragraph(_md_inline_to_rl("Strategic Recommendation"), H2))
    if rec.get("winner"):
        story.append(Paragraph(_md_inline_to_rl(f"Recommended Bidder: {rec.get('winner')}"), H3))
    if rec.get("comprehensive_rationale"):
        story.append(Paragraph(_md_inline_to_rl("Strategic Rationale"), H3))
        story.append(Paragraph(_md_inline_to_rl(rec["comprehensive_rationale"]), Body))

    # Appendix: Implementation Roadmap + KPIs (after main recommendation)
    appx = data.get("implementation_roadmap", {}) or {}
    show_appx = any(appx.values()) or bool(rec.get("kpis_and_monitoring"))
    if show_appx:
        story.append(PageBreak())
        story.append(Paragraph(_md_inline_to_rl("Appendix"), H2))
        if appx.get("immediate_actions"):
            story.append(Paragraph(_md_inline_to_rl("Implementation — Immediate Actions (30 days)"), H3))
            story.append(ListFlowable([ListItem(Paragraph(_md_inline_to_rl(x), Bullet)) for x in appx["immediate_actions"][:6]], bulletType="bullet"))
            story.append(Spacer(1, 4))
        if appx.get("contract_negotiation_priorities"):
            story.append(Paragraph(_md_inline_to_rl("Contract Negotiation Priorities"), H3))
            story.append(ListFlowable([ListItem(Paragraph(_md_inline_to_rl(x), Bullet)) for x in appx["contract_negotiation_priorities"][:6]], bulletType="bullet"))
            story.append(Spacer(1, 4))
        if appx.get("mobilization_requirements"):
            story.append(Paragraph(_md_inline_to_rl("Mobilization Requirements"), H3))
            story.append(ListFlowable([ListItem(Paragraph(_md_inline_to_rl(x), Bullet)) for x in appx["mobilization_requirements"][:6]], bulletType="bullet"))
            story.append(Spacer(1, 4))
        if appx.get("governance_framework"):
            story.append(Paragraph(_md_inline_to_rl("Governance Framework"), H3))
            story.append(ListFlowable([ListItem(Paragraph(_md_inline_to_rl(x), Bullet)) for x in appx["governance_framework"][:6]], bulletType="bullet"))
            story.append(Spacer(1, 4))
        if appx.get("stakeholder_communication"):
            story.append(Paragraph(_md_inline_to_rl("Stakeholder Communication"), H3))
            story.append(ListFlowable([ListItem(Paragraph(_md_inline_to_rl(x), Bullet)) for x in appx["stakeholder_communication"][:6]], bulletType="bullet"))
            story.append(Spacer(1, 4))
        if appx.get("milestone_tracking"):
            story.append(Paragraph(_md_inline_to_rl("Milestone Tracking"), H3))
            story.append(ListFlowable([ListItem(Paragraph(_md_inline_to_rl(x), Bullet)) for x in appx["milestone_tracking"][:6]], bulletType="bullet"))
            story.append(Spacer(1, 4))
        if rec.get("kpis_and_monitoring"):
            story.append(Paragraph(_md_inline_to_rl("KPIs & Monitoring"), H3))
            story.append(ListFlowable([ListItem(Paragraph(_md_inline_to_rl(x), Bullet)) for x in (rec.get("kpis_and_monitoring") or [])[:6]], bulletType="bullet"))

    # Build
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf.getvalue()
