from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import cabin
import report
import ui
from ui import C
from src.whatif import DEFAULT_COSTS, counterfactual

ui.header("GATE 01 · LIVE PREDICTION", "Will this passenger be satisfied?",
          "Pick the cabin and journey stage, enter what is known so far, and every model scores the passenger "
            "side by side against the baselines.")

# Segmented controls return None when the active option is clicked again; restore the last value.
class_key = ui.key("Class")
if st.session_state.get(class_key) is None:
    st.session_state[class_key] = st.session_state.get("_last_class", "Business")
if st.session_state.get("stage") is None:
    st.session_state["stage"] = st.session_state.get("_last_stage", "Landed")

top = st.container(key="predict_top")
c1, c2 = top.columns([1, 1.25])
c1.segmented_control("Cabin class", C.CLASS_ORDER[::-1], format_func=lambda c: C.CABIN_NAMES[c], key=class_key)
c2.segmented_control("Journey stage - what do we know so far?", C.STAGE_ORDER, key="stage")
stage = st.session_state["stage"]
st.session_state["_last_stage"] = stage
known = set(C.stage_known_ratings(stage))
landed = stage == "Landed"

X = ui.current_passenger()
p_team = ui.predict_stage(X, stage)
good = p_team >= 0.5

# ---- 3D cabin ----
selected = st.session_state[class_key]
prev = st.session_state.get("_last_class", selected)
row = X.iloc[0]
seat = cabin.pick_seat(selected, f"{row['Gender']}{row['Age']}{row['Customer Type']}{row['Flight Distance']}")
cabin.render(selected, prev, seat, good, p_team, ui.cabin_rates(), stage)
st.session_state["_last_class"] = selected

left, right = st.container(key="predict_main").columns([5, 7], gap="large")

with left:
    st.markdown(f"#### {'Quick passengers'}")
    cols = st.columns(2)
    for i, name in enumerate(ui.PRESETS):
        cols[i % 2].button(name, on_click=ui.apply_preset, args=(name,), width="stretch", key=f"preset_{i}")

    with st.expander("Profile & trip", expanded=True):
        a, b = st.columns(2)
        a.selectbox("Gender", C.CATEGORY_VALUES["Gender"], key=ui.key("Gender"))
        b.selectbox("Customer Type", C.CATEGORY_VALUES["Customer Type"], key=ui.key("Customer Type"))
        a.selectbox("Type of Travel", C.CATEGORY_VALUES["Type of Travel"], key=ui.key("Type of Travel"))
        b.number_input("Age", *C.NUM_RANGES["Age"], key=ui.key("Age"))
        a.number_input("Flight Distance (miles)", *C.NUM_RANGES["Flight Distance"], step=50, key=ui.key("Flight Distance"))
        dep_known = stage != "Booked"
        b.number_input("Departure delay (min)" + ("" if dep_known else " - not yet known"),
                       *C.NUM_RANGES["Departure Delay in Minutes"], step=5,
                       key=ui.key("Departure Delay in Minutes"), disabled=not dep_known)
        a.number_input("Arrival delay (min)" + ("" if landed else " - not yet known"),
                       *C.NUM_RANGES["Arrival Delay in Minutes"], step=5,
                       key=ui.key("Arrival Delay in Minutes"), disabled=not landed)

    with st.expander("Service ratings - {n} of 14 known at this stage  (0 = N/A)".format(n=len(known)), expanded=True):
        a, b = st.columns(2)
        for i, col in enumerate(C.RATING_COLS):
            is_known = col in known
            (a if i % 2 == 0 else b).select_slider(col if is_known else col + " - later", options=[0, 1, 2, 3, 4, 5],
                                                   format_func=ui.rating_label, key=ui.key(col), disabled=not is_known)

with right:
    label = "TEAM ENSEMBLE" if landed else "STAGE MODEL ({stage})".format(stage=stage.upper())
    st.markdown(f'<div class="mono" style="color:#8a9bb4;letter-spacing:.2em;font-size:.75rem">'
                f'{label} {"VERDICT"}</div>', unsafe_allow_html=True)
    st.markdown(ui.flaps("Satisfied" if good else "Dissatisfied", good), unsafe_allow_html=True)

    if not landed:
        sm = ui.load_trust()["stage_metrics"]
        acc = float(sm.loc[sm["stage"] == stage, "Accuracy"].iloc[0])
        st.markdown(f'<div class="mono" style="color:#8a9bb4">P(satisfied) = <b style="color:{ui.GOOD if good else ui.BAD}">'
                    f'{p_team:.3f}</b> · {"stage model accuracy"} {acc*100:.1f}%</div>', unsafe_allow_html=True)
        ui.note(("<b>Early warning mode.</b> Only {n} of 14 ratings are known, so a model trained on exactly "
                  "this information is used. Accuracy grows as the journey progresses:").format(n=len(known)))
        smt = sm
        order = [s for s in C.STAGE_ORDER]
        chart = alt.Chart(smt).mark_line(point=alt.OverlayMarkDef(size=90), color=ui.ACCENT, strokeWidth=3).encode(
            x=alt.X("stage:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0.75, 1]), axis=alt.Axis(format="%"), title="Accuracy"),
            tooltip=["stage", alt.Tooltip("Accuracy:Q", format=".2%"), alt.Tooltip("ROC-AUC:Q", format=".3f")])
        now = alt.Chart(smt[sm["stage"] == stage]).mark_point(size=400, color="#ffffff", filled=False, strokeWidth=2).encode(
            x=alt.X("stage:N", sort=order), y="Accuracy:Q")
        st.altair_chart(ui.themed((chart + now).properties(height=220)), width="stretch")
        st.caption("Switch the journey stage to Landed to see all member models, conformal sets and the PDF report.")
        st.stop()

    probs = ui.predict(X).iloc[0]
    level = st.select_slider("Conformal confidence level", [0.80, 0.90, 0.95, 0.99], value=0.95,
                             format_func=lambda v: f"{v:.0%}")
    cset, certain = ui.conformal_label(p_team, level)
    ood = ui.ood_percentile(X)
    empty_set = cset.startswith("{ }")
    k = st.columns(3)
    k[0].markdown(ui.kpi("P(SATISFIED)", f"{p_team:.3f}", ui.GOOD if good else ui.BAD,
                         "Team ensemble probability"), unsafe_allow_html=True)
    shown_set = "{ }" if empty_set else cset.replace("Dissatisfied", "Dissat.").replace(
        "Satisfied" + ",", "Sat." + ",")
    k[1].markdown(ui.kpi("{lv:.0%} prediction set".format(lv=level), shown_set, ui.GOOD if certain else ui.ACCENT,
                         "One label = confident" if certain else
                         ("Empty set = no confident call" if empty_set else "Both labels = not a confident call")),
                  unsafe_allow_html=True)
    pct = round(ood)
    suffix = "th" if 10 <= pct % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(pct % 10, "th")
    pct_txt = f"{pct}{suffix} pct"
    k[2].markdown(ui.kpi("Unusualness", pct_txt,
                         ui.BAD if ood > 99 else ("#e9eef6" if ood < 95 else ui.ACCENT),
                         "vs. training passengers"), unsafe_allow_html=True)
    if empty_set:
        ui.note(("<b>Uncertain passenger.</b> At {lv:.0%} confidence neither outcome is supported strongly enough, "
                  "so the conformal set is empty - a human should review before acting.").format(lv=level), warn=True)
    elif not certain:
        ui.note(("<b>Uncertain passenger.</b> At {lv:.0%} confidence the conformal set contains both outcomes - "
                  "a human should review before acting.").format(lv=level), warn=True)
    if ood > 99:
        ui.note("<b>Out-of-distribution.</b> This passenger is further from the training data than 99% of "
                  "calibration passengers; predictions may be unreliable.", warn=True)
    member_votes = [probs[m] >= 0.5 for m in ui.MEMBER_MODELS]
    if len(set(member_votes)) > 1:
        ui.note("Models disagree: {n} of 4 member models say <b>satisfied</b>.".format(n=sum(member_votes)), warn=True)

    st.markdown(ui.cards_html(probs), unsafe_allow_html=True)

    st.markdown(f"#### {'Why? Top drivers (XGBoost TreeSHAP)'}")
    contrib = ui.xgb_contributions(X).head(10)
    pos, neg = "Pushes to satisfied", "Pushes to dissatisfied"
    contrib["direction"] = contrib["shap"].map(lambda s: pos if s > 0 else neg)
    chart = alt.Chart(contrib).mark_bar(cornerRadiusEnd=3, height=alt.RelativeBandSize(0.7)).encode(
        x=alt.X("shap:Q", title="Contribution (log-odds)"),
        y=alt.Y("label:N", sort=None, title=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=[pos, neg], range=[ui.GOOD, ui.BAD]),
                        legend=None),
        tooltip=["label", alt.Tooltip("shap:Q", format="+.3f")],
    ).properties(height=30 * len(contrib))
    st.altair_chart(ui.themed(chart), width="stretch")
    st.markdown(f'<div class="mono" style="font-size:.78rem;color:#8a9bb4">'
                f'<b style="color:{ui.GOOD}">■</b> {pos} &nbsp;&nbsp; <b style="color:{ui.BAD}">■</b> {neg}</div>',
                unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    # The prepared PDF belongs to one passenger / language / confidence level; never offer a stale one.
    pdf_sig = (tuple(X.iloc[0].tolist()), level)
    if st.session_state.get("pdf_sig") != pdf_sig:
        st.session_state.pop("pdf", None)
    with r1:
        if st.button("Prepare PDF report", width="stretch", key="pdf_btn"):
            with st.spinner("Searching cheapest service recovery..."):
                steps, _ = counterfactual(lambda d: ui.predict(d)["team"].to_numpy(), X, DEFAULT_COSTS,
                                          locked=["Gate location", "Departure/Arrival time convenient"], max_steps=6)
            pdf = report.build_pdf(
                passenger={c: X.iloc[0][c] if isinstance(X.iloc[0][c], str) else X.iloc[0][c] for c in C.FEATURE_COLS},
                seat=seat, cabin=C.CABIN_NAMES[selected],
                verdict="Satisfied" if good else "Dissatisfied", good=good, p_team=p_team, conformal=cset,
                model_rows=[[ui.mname(m), ui.MODEL_INFO[m]["role"], f"{probs[m]:.3f}",
                             "Satisfied" if probs[m] >= .5 else "Dissatisfied"]
                            for m in ["team"] + ui.MEMBER_MODELS + ui.BASELINES],
                drivers=[[r.label, f"{r.shap:+.3f}", r.direction] for r in contrib.itertuples()],
                upgrades=[[i + 1, s, f"{a} -> {b}", c, f"{p0:.3f} -> {p1:.3f}"] for i, (s, a, b, c, p0, p1) in enumerate(steps)],
                ood_pct=ood)
            st.session_state["pdf"], st.session_state["pdf_sig"] = pdf, pdf_sig
        if "pdf" in st.session_state:
            st.download_button("Download report.pdf", st.session_state["pdf"], "skypulse_passenger_report.pdf",
                               "application/pdf", width="stretch")
    with r2:
        with st.popover("Record actual outcome", width="stretch"):
            st.caption("Feedback is logged for the next retraining cycle.")
            actual = st.radio("What did the passenger actually report?", ["satisfied", "neutral or dissatisfied"])
            if st.button("Save feedback", key="fb_btn"):
                row = {**{c: X.iloc[0][c] for c in C.FEATURE_COLS}, C.TARGET: actual, "p_team": round(p_team, 4),
                       "predicted": "satisfied" if good else "neutral or dissatisfied",
                       "logged_at": datetime.now().isoformat(timespec="seconds")}
                path = C.ARTIFACTS / "feedback.csv"
                try:
                    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
                    st.success("Saved. " + ("Model was right." if row["predicted"] == actual
                                                else "Logged as a model error."))
                except OSError as exc:
                    st.error("Could not save feedback: {e}".format(e=exc))
