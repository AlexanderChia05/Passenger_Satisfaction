import altair as alt
import pandas as pd
import streamlit as st

import ui
from ui import C
from src.whatif import DEFAULT_COSTS, plan_budget

ui.header("GATE 03 · INVEST", "Service budget planner",
          "Give the planner a budget and a passenger group. It picks which services to upgrade, step by step, "
            "to turn the most passengers into satisfied ones per ringgit spent.")

test = ui.load_test_predictions()
f1, f2, f3, f4 = st.columns(4)
cls = f1.multiselect("Cabin", C.CLASS_ORDER, default=C.CLASS_ORDER, format_func=lambda c: C.CABIN_NAMES[c])
trip = f2.multiselect("Type of travel", C.CATEGORY_VALUES["Type of Travel"], default=C.CATEGORY_VALUES["Type of Travel"])
cust = f3.multiselect("Customer type", C.CATEGORY_VALUES["Customer Type"], default=C.CATEGORY_VALUES["Customer Type"])
n = f4.select_slider("Passengers to plan for", [500, 1000, 2000, 5000], value=1000)

group = test[test["Class"].isin(cls) & test["Type of Travel"].isin(trip) & test["Customer Type"].isin(cust)]
if group.empty:
    st.warning("No passengers match these filters.")
    st.stop()
group = group.sample(min(n, len(group)), random_state=C.SEED)[C.FEATURE_COLS]

b1, b2, b3 = st.columns([1.2, 1, 1])
budget = b1.number_input("Budget (RM)", 1_000, 1_000_000, 50_000, 5_000)
engine = b2.radio("Scoring model", ["XGBoost (fast)", "Team ensemble (slow)"])
locked = b3.multiselect("Cannot change", C.RATING_COLS,
                        default=["Gate location", "Departure/Arrival time convenient"])
with st.expander("Upgrade cost per passenger, per +1 star (RM)"):
    edited = st.data_editor(pd.DataFrame({"service": [s for s in DEFAULT_COSTS],
                                          "cost (RM)": list(DEFAULT_COSTS.values())}),
                            hide_index=True, width="stretch", disabled=["service"], key="plan_costs")
# a cleared cell falls back to the default cost; negative costs are not meaningful
costs = {s: max(float(v), 0.0) if pd.notna(v) else float(DEFAULT_COSTS[s])
         for s, v in zip(DEFAULT_COSTS, edited["cost (RM)"])}

if engine.startswith("XGBoost"):
    model = ui.load_models()["xgb"]
    predict_fn = lambda d: model.predict_proba(d)[:, 1]  # noqa: E731
else:
    predict_fn = lambda d: ui.predict(d)["team"].to_numpy()  # noqa: E731

# results belong to the inputs they were computed for; changing any input hides the old plan
plan_sig = (tuple(cls), tuple(trip), tuple(cust), n, budget, engine, tuple(locked), tuple(sorted(costs.items())))
if st.button("Run planner", type="primary", key="run_planner"):
    with st.spinner("Evaluating every service upgrade at each step..."):
        before = predict_fn(group)
        steps, after = plan_budget(predict_fn, group, costs, budget, locked)
    st.session_state["plan"] = (steps, float((before >= .5).mean()), float((after >= .5).mean()), len(group))
    st.session_state["plan_sig"] = plan_sig

if "plan" not in st.session_state or st.session_state.get("plan_sig") != plan_sig:
    ui.note("Choose a passenger group and a budget, then press <b>Run planner</b>. "
              "The result shows the satisfied rate before and after, and each upgrade bought in order.")
else:
    steps, r0, r1, size = st.session_state["plan"]
    k = st.columns(4)
    k[0].markdown(ui.kpi("Satisfied before", f"{r0*100:.1f}%"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Satisfied after", f"{r1*100:.1f}%", ui.GOOD), unsafe_allow_html=True)
    spent = steps["cost (RM)"].sum() if len(steps) else 0
    conv = steps["newly satisfied"].sum() if len(steps) else 0
    k[2].markdown(ui.kpi("Spent", f"RM {spent:,.0f}", ui.ACCENT), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("RM per converted", f"RM {spent / conv:,.0f}" if conv else "-"), unsafe_allow_html=True)
    if steps.empty:
        ui.note("No affordable upgrade converts anyone. Raise the budget or unlock more services.", warn=True)
    else:
        st.write("")
        shown = steps
        chart_df = pd.concat([pd.DataFrame({"step": ["start"], "satisfied rate": [r0]}),
                              shown.assign(step=shown["step"].astype(str) + ". " + shown["service"])[["step", "satisfied rate"]]])
        st.markdown(f"#### {'Satisfied rate after each upgrade step'}")
        base = alt.Chart(chart_df).encode(y=alt.Y("step:N", sort=None, title=None))
        bars = base.mark_bar(cornerRadiusEnd=4, height=alt.RelativeBandSize(0.7), color=ui.ACCENT).encode(
            x=alt.X("satisfied rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]),
                    title="satisfied rate"),
            tooltip=["step", alt.Tooltip("satisfied rate:Q", format=".1%")])
        labels = base.mark_text(align="left", dx=4, color="#e9eef6", font="JetBrains Mono", fontSize=11).encode(
            x="satisfied rate:Q", text=alt.Text("satisfied rate:Q", format=".1%"))
        st.altair_chart(ui.themed((bars + labels).properties(height=26 * len(chart_df))), width="stretch")
        st.dataframe(shown.style.format({"cost (RM)": "{:,.0f}", "RM per convert": "{:,.0f}",
                                              "satisfied rate": "{:.1%}", "cumulative spend (RM)": "{:,.0f}"}),
                     hide_index=True, width="stretch")
        st.download_button("Download plan CSV", steps.to_csv(index=False).encode(), "skypulse_service_plan.csv", "text/csv")
    st.caption("Planned on {n:,} hold-out passengers. Associations, not proven causes - pilot before rollout.".format(n=size))
