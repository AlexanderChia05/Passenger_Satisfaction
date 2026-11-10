import altair as alt
import streamlit as st

import ui
from ui import C

ui.header("GATE 06 · INSIGHT", "Explain the models",
          "What drives satisfaction for different passenger groups, and the ensemble distilled into "
            "plain IF-THEN rules a manager can read.")



def simplify_rule(conds: list[str]) -> list[str]:
    """Merge tree-path conditions on the same feature for readability:
    'x ≥ 4 AND x ≤ 4' -> 'x = 4', 'x ≥ 4 AND x ≥ 5' -> 'x ≥ 5' (same logic, shorter)."""
    import re
    lo, hi, order = {}, {}, []
    for c in conds:
        m = re.match(r"^(.*?) (≤|≥) (-?\d+(?:\.\d+)?)$", c)
        if not m:
            order.append(("raw", c))
            continue
        subj, op, txt = m.groups()
        v = (float(txt), txt)
        if subj not in lo and subj not in hi:
            order.append(("feat", subj))
        if op == "≥":
            lo[subj] = max(lo.get(subj, v), v)
        else:
            hi[subj] = min(hi.get(subj, v), v)
    out = []
    for kind, x in order:
        if kind == "raw":
            out.append(x)
        elif x in lo and x in hi and lo[x][0] == hi[x][0]:
            out.append(f"{x} = {lo[x][1]}")
        else:
            out += ([f"{x} ≥ {lo[x][1]}"] if x in lo else []) + ([f"{x} ≤ {hi[x][1]}"] if x in hi else [])
    return out


tab_seg, tab_rules = st.tabs(["Segment drivers", "Rules extracted from the ensemble"])

with tab_seg:
    with st.spinner("Computing SHAP explanations for 6,000 hold-out passengers..."):
        shap_df, raw = ui.test_shap()
    seg = raw["Class"].map(lambda c: C.CABIN_NAMES[c]) + " · " + raw["Type of Travel"].astype(str)
    counts = seg.value_counts()
    valid = counts[counts >= 100].index.tolist() or counts.index.tolist()
    a, b, c = st.columns([1, 1, 1])
    s1 = a.selectbox("Segment A", valid, index=0)
    s2 = b.selectbox("Segment B", valid, index=min(1, len(valid) - 1))
    top_n = c.slider("Features shown", 5, 20, 10)
    imp = shap_df.abs().groupby(seg).mean()
    order = imp.loc[[s1, s2]].mean().sort_values(ascending=False).head(top_n).index.tolist()
    long = imp.loc[[s1, s2], order].reset_index(names="segment").melt("segment", var_name="feature", value_name="impact")
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3, height=10).encode(
        x=alt.X("impact:Q", title="Mean |SHAP| (log-odds)"), y=alt.Y("feature:N", sort=order, title=None),
        yOffset="segment:N",
        color=alt.Color("segment:N", scale=alt.Scale(range=[ui.ACCENT, "#5ab0ff"]), legend=ui.bottom_legend()),
        tooltip=["segment", "feature", alt.Tooltip("impact:Q", format=".3f")]).properties(height=30 * top_n)
    st.altair_chart(ui.themed(bars), width="stretch")
    diff = (imp.loc[s1] - imp.loc[s2]).sort_values()
    if s1 != s2:  # comparing a segment with itself has no "more important" driver
        ui.note(("<b>{a}</b> matters much more for {s1}; <b>{b}</b> matters more for {s2}. "
                  "Different segments need different service priorities.").format(a=diff.index[-1], s1=s1, b=diff.index[0], s2=s2))

    st.markdown(f"#### {'Driver heatmap across all segments'}")
    heat_cols = imp.mean().sort_values(ascending=False).head(12).index.tolist()
    hl = imp.loc[valid, heat_cols].reset_index(names="segment").melt("segment", var_name="feature", value_name="impact")
    heat = alt.Chart(hl).mark_rect(cornerRadius=3).encode(
        x=alt.X("feature:N", sort=heat_cols, title=None, axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("segment:N", title=None),
        color=alt.Color("impact:Q", scale=alt.Scale(scheme="goldorange"),
                        legend=alt.Legend(orient="right", title="Impact", gradientLength=120)),
        tooltip=["segment", "feature", alt.Tooltip("impact:Q", format=".3f")]).properties(height=36 * len(valid))
    st.altair_chart(ui.themed(heat), width="stretch")
    st.caption("Darker cell = that feature moves the prediction more for that segment (mean |SHAP|).")

with tab_rules:
    sur = ui.load_trust()["surrogates"]
    depth = st.select_slider("Rule depth (max conditions per rule)", list(sur), value=3)
    s = sur[depth]
    k = st.columns(4)
    k[0].markdown(ui.kpi("Rules", str(s["n_rules"])), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Fidelity to ensemble", f"{s['fidelity']*100:.1f}%", ui.ACCENT), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Accuracy vs truth", f"{s['accuracy']*100:.1f}%", ui.GOOD), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Hand-made rule", f"{ui.load_metrics()['models']['rule']['Accuracy']*100:.1f}%", ui.MUTED),
                  unsafe_allow_html=True)
    ui.note("A shallow decision tree is trained to <b>imitate the team ensemble</b>. Fidelity = how often the rules agree "
              "with the ensemble on hold-out passengers. Deeper = more faithful but harder to read.")
    html = ui.step_head(("Rule #", "min-width:2.4rem"), ("Condition", "flex:1"), ("Predicts", "width:8.5rem;text-align:center"),
                        ("Confidence", "width:7rem;text-align:right"),
                        ("Passengers covered", "width:9rem;text-align:right"))
    for i, r in s["rules"].iterrows():
        sat = r["prediction"] == "Satisfied"
        color = ui.GOOD if sat else ui.BAD
        conds = f" <span style='color:#ffb627'>{'AND'}</span> ".join(
            ui.translate_condition(c) for c in simplify_rule(r["rule"].split(" AND ")))
        html += (f'<div class="step"><span class="n">R{i+1:02d}</span><span style="flex:1"><span style="color:#8a9bb4">'
                 f'{"IF"}</span> {conds}</span>'
                 f'<span style="width:8.5rem;text-align:center"><span class="chip {"good" if sat else "bad"}">'
                 f'{"SATISFIED" if sat else "DISSATISFIED"}</span></span>'
                 f'<span class="mono" style="width:7rem;text-align:right;color:{color}">{r["confidence"]*100:.0f}%</span>'
                 f'<span class="mono" style="width:9rem;text-align:right;color:#8a9bb4">{r["support %"]:.1f}%</span></div>')
    # fixed-width columns: scroll inside the box on phones instead of widening the page
    st.markdown(f'<div style="overflow-x:auto">{html}</div>', unsafe_allow_html=True)
