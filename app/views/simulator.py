import altair as alt
import pandas as pd
import streamlit as st

import ui
from ui import C
from src.whatif import DEFAULT_COSTS, counterfactual

ui.header("GATE 02 · WHAT-IF", "Service improvement simulator",
          "Change one service and watch every model react, or let the system find the cheapest upgrades "
            "that turn this passenger into a satisfied one.")

X = ui.current_passenger()
base = ui.predict(X).iloc[0]
p = X.iloc[0]
ui.note("Passenger from <b>Predict</b>: {age}y · {cabin} · {trip} · {cust} · team P(satisfied) <b class=\"mono\">{p:.3f}</b>".format(age=p["Age"], cabin=C.CABIN_NAMES[p["Class"]], trip=p["Type of Travel"], cust=p["Customer Type"], p=base["team"]))

tab_curve, tab_cf, tab_lift = st.tabs(["Sensitivity curve", "Cheapest path to satisfied", "Biggest single lever"])

with tab_curve:
    a, b = st.columns([2, 1])
    service = a.selectbox("Service to vary", C.RATING_COLS, index=C.RATING_COLS.index("Online boarding"))
    show_mono = b.toggle("Add monotonic XGBoost", value=True,
                         help="Variant trained with the constraint 'more stars can never lower P(satisfied)'.")
    grid = pd.concat([X.assign(**{service: r}) for r in range(1, 6)], ignore_index=True)
    probs = ui.predict(grid)
    keys = ui.MEMBER_MODELS + ["team"]
    if show_mono:
        probs["xgb_mono"] = ui.load_variants()["xgb_mono"].predict_proba(grid)[:, 1]
        keys = keys + ["xgb_mono"]
    long = probs[keys].assign(rating=range(1, 6)).melt("rating", var_name="key", value_name="p")
    long["model"] = long["key"].map(ui.mname)
    lines = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("rating:O", title="{s} rating".format(s=service)),
        y=alt.Y("p:Q", title="P(satisfied)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=ui.color_scale(keys), legend=ui.bottom_legend()),
        strokeDash=alt.condition(alt.datum.key == "xgb_mono", alt.value([6, 3]), alt.value([1, 0])),
        tooltip=["model", "rating", alt.Tooltip("p:Q", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color=ui.MUTED).encode(y="y")
    now = alt.Chart(pd.DataFrame({"rating": [int(p[service])], "lbl": ["current"]}))
    now_rule = now.mark_rule(color=ui.ACCENT, strokeWidth=10, opacity=.12).encode(x="rating:O")
    now_txt = now.mark_text(color=ui.ACCENT, dy=-8, font="JetBrains Mono", fontSize=11).encode(
        x="rating:O", y=alt.value(0), text="lbl:N")
    st.altair_chart(ui.themed((now_rule + lines + rule + now_txt).properties(height=380)), width="stretch")
    mono = ui.load_trust()["mono"]
    st.caption(("Current rating: {r} (shaded column). Dashed line = threshold 0.5. On 1,500 hold-out passengers, +1 star lowered "
                 "P(satisfied) by >0.5 pts in {a:.1f}% of cases for XGBoost vs {b:.1f}% for the monotonic variant.").format(r=ui.rating_label(int(p[service])), a=mono["xgb"] * 100, b=mono.get("xgb_mono", 0) * 100))

with tab_cf:
    c1, c2, c3 = st.columns(3)
    budget = c1.slider("Max rating upgrades (+1 star each)", 1, 15, 6)
    target = c2.slider("Target team P(satisfied)", 0.5, 0.95, 0.6, 0.05)
    objective = c3.radio("Optimise for", ["Lowest cost", "Fewest upgrades"], horizontal=True)
    locked = st.multiselect("Services the airline cannot change", C.RATING_COLS,
                            default=["Gate location", "Departure/Arrival time convenient"])
    with st.expander("Upgrade cost per passenger, per +1 star (RM)"):
        edited = st.data_editor(pd.DataFrame({"service": [s for s in DEFAULT_COSTS],
                                              "cost (RM)": list(DEFAULT_COSTS.values())}),
                                hide_index=True, width="stretch", disabled=["service"], key="cost_editor")
    real_costs = {s: max(float(v), 0.0) if pd.notna(v) else float(DEFAULT_COSTS[s])
                  for s, v in zip(DEFAULT_COSTS, edited["cost (RM)"])}
    costs = real_costs if objective == "Lowest cost" else {s: 1 for s in C.RATING_COLS}

    steps, p_now = counterfactual(lambda d: ui.predict(d)["team"].to_numpy(), X, costs, locked, target, budget)
    if float(base["team"]) >= target:
        ui.note("Passenger already meets the target - no recovery needed.")
    elif not steps:
        ui.note("No upgrade improves this passenger. The drivers are outside service control (e.g. trip type, delay).",
                warn=True)
    else:
        reached = p_now >= target
        total = sum(real_costs.get(s, 0) for s, *_ in steps)
        ui.note(("Target reached" if reached else "Target not reached")
                + " in <b>{n}</b> upgrade(s) costing <b>RM {c:,.0f}</b>: P(satisfied) {a:.3f} → <b>{b:.3f}</b>".format(n=len(steps), c=total, a=base["team"], b=p_now), warn=not reached)
        html = ui.step_head(("Step", "min-width:2.4rem"), ("Service", "flex:1"), ("Rating", "width:9rem;text-align:right"),
                            ("Cost", "width:6rem;text-align:right"),
                            ("P(satisfied)", "width:9rem;text-align:right"))
        html += "".join(f'<div class="step"><span class="n">{i+1:02d}</span><span style="flex:1">{s}</span>'
                       f'<span class="mono" style="color:#8a9bb4;width:9rem;text-align:right">{ui.rating_label(a)} → {ui.rating_label(b)}</span>'
                       f'<span class="mono" style="color:{ui.ACCENT};width:6rem;text-align:right">RM {real_costs.get(s, 0):,.0f}</span>'
                       f'<span class="mono" style="color:{ui.GOOD};width:9rem;text-align:right">{p0:.3f} → {p1:.3f}</span></div>'
                       for i, (s, a, b, _, p0, p1) in enumerate(steps))
        st.markdown(html, unsafe_allow_html=True)

with tab_lift:
    rows = [X.assign(**{s: min(int(p[s]) + 1, 5)}) for s in C.RATING_COLS]
    lift = ui.predict(pd.concat(rows, ignore_index=True))["team"].to_numpy() - base["team"]
    df = pd.DataFrame({"service": [s for s in C.RATING_COLS], "lift": lift}).sort_values("lift", ascending=False)
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=16).encode(
        x=alt.X("lift:Q", title="Change in team P(satisfied) from +1 star", axis=alt.Axis(format="+.2f")),
        y=alt.Y("service:N", sort=None, title=None),
        color=alt.condition("datum.lift > 0", alt.value(ui.GOOD), alt.value(ui.BAD)),
        tooltip=["service", alt.Tooltip("lift:Q", format="+.4f")]).properties(height=420)
    st.altair_chart(ui.themed(bars), width="stretch")
