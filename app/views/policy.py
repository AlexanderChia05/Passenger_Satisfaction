import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve

import ui

ui.header("Decision policy",
          "The airline sends a service-recovery voucher to passengers predicted dissatisfied. Set the costs; the "
          "threshold is chosen on training data (5-fold out-of-fold predictions) and then scored once on the hold-out set.")

preds = ui.load_test_predictions()
y = preds["y_true"].to_numpy()
oof = ui.load_oof()

c1, c2, c3 = st.columns(3)
model = c1.selectbox("Model", ui.MEMBER_MODELS + ["rule"], format_func=ui.mname,
                     index=ui.MEMBER_MODELS.index(ui.HEADLINE))
voucher = c2.number_input("Voucher cost per flagged passenger (RM)", 0, 1000, 50, 10)
churn = c3.number_input("Cost of a missed unhappy passenger (RM)", 0, 5000, 300, 50)

p = preds[model].to_numpy()
ts = np.round(np.arange(0.05, 0.96, 0.01), 2)


def cost_per_passenger(prob, truth):
    """Average cost per passenger at every threshold. Flag = predicted dissatisfied = P(satisfied) < t."""
    return np.array([(voucher * (prob < t).sum() + churn * ((prob >= t) & (truth == 0)).sum()) / len(truth) for t in ts])


hold_cost = cost_per_passenger(p, y)
if oof is not None and model in oof:
    # Choose on out-of-fold training predictions, report on the hold-out set: the hold-out labels never
    # influence the threshold, so the reported cost is an honest estimate for new passengers.
    sel_cost = cost_per_passenger(oof[model].to_numpy(), oof["y_true"].to_numpy())
    chosen_on = "5-fold OOF on 103,904 training passengers"
else:
    sel_cost = hold_cost
    chosen_on = "hold-out set (no OOF predictions in these artifacts)"
    ui.note("Prototype artifacts have no out-of-fold predictions, so the threshold is chosen on the hold-out set "
            "and the reported cost is optimistic.", warn=True)
best_t = float(ts[int(np.argmin(sel_cost))])
th = st.slider("Threshold on P(satisfied): flag passenger if below", 0.05, 0.95, best_t, 0.01)

flag = p < th
tp, fp = int((flag & (y == 0)).sum()), int((flag & (y == 1)).sum())
fn, tn = int((~flag & (y == 0)).sum()), int((~flag & (y == 1)).sum())
cost_now = voucher * flag.sum() + churn * fn
k = st.columns(4)
k[0].markdown(ui.kpi("Chosen threshold", f"{best_t:.2f}", ui.ACCENT, "picked on training OOF predictions"),
              unsafe_allow_html=True)
k[1].markdown(ui.kpi("Hold-out cost at the slider", f"RM {cost_now/1000:,.1f}k",
                     sub=f"RM {cost_now / len(y):,.2f} per passenger"), unsafe_allow_html=True)
k[2].markdown(ui.kpi("Unhappy passengers caught", f"{tp/max(tp+fn, 1)*100:.1f}%", sub="hold-out"),
              unsafe_allow_html=True)
k[3].markdown(ui.kpi("Vouchers sent to satisfied passengers", f"{fp:,}", sub="hold-out"), unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("#### Cost per passenger vs threshold")
    cdf = pd.concat([pd.DataFrame({"threshold": ts, "cost": sel_cost, "data": "Training OOF (chooses)"}),
                     pd.DataFrame({"threshold": ts, "cost": hold_cost, "data": "Hold-out (reports)"})])
    if sel_cost is hold_cost:
        cdf = cdf[cdf["data"] == "Hold-out (reports)"]
    line = alt.Chart(cdf).mark_line(strokeWidth=2).encode(
        x=alt.X("threshold:Q", title="Threshold"), y=alt.Y("cost:Q", title="Average cost per passenger (RM)"),
        color=alt.Color("data:N", scale=alt.Scale(domain=["Training OOF (chooses)", "Hold-out (reports)"],
                                                  range=[ui.MUTED, ui.ACCENT]), legend=ui.bottom_legend()),
        strokeDash=alt.StrokeDash("data:N", scale=alt.Scale(domain=["Training OOF (chooses)", "Hold-out (reports)"],
                                                            range=[[5, 4], [1, 0]]), legend=None),
        tooltip=["data", alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("cost:Q", format=",.2f")])
    marks = alt.Chart(pd.DataFrame({"x": [best_t, th], "kind": ["chosen", "slider"]})).mark_rule(
        strokeDash=[2, 3], color=ui.TEXT2).encode(x="x:Q", tooltip=["kind", alt.Tooltip("x:Q", format=".2f")])
    st.altair_chart(ui.themed((line + marks).properties(height=320)), width="stretch")
    hold_best = float(ts[int(np.argmin(hold_cost))])
    gap = (hold_cost[list(ts).index(best_t)] - hold_cost.min())
    verdict = ("so the choice transfers well to unseen passengers" if gap <= 0.25 * max(hold_cost.min(), 1e-9) / 10
               else "a noticeable gap, so treat the cost estimate with care")
    st.caption(f"Threshold chosen on: {chosen_on}. Tuned on the hold-out labels instead, it would be {hold_best:.2f}, "
               f"RM {gap:,.2f} per passenger cheaper on the hold-out set: {verdict}. The threshold is never fitted "
               "to the data it is scored on.")

with right:
    st.markdown("#### Outcome at the slider threshold (rows = actual)")
    # Colour encodes what the cell MEANS (right call / wasted voucher / missed unhappy), not its size,
    # so the large correct cells never look alarming.
    cm = pd.DataFrame({"Actual": ["Dissatisfied", "Dissatisfied", "Satisfied", "Satisfied"],
                       "Action": ["Voucher", "No voucher", "Voucher", "No voucher"],
                       "n": [tp, fn, fp, tn],
                       "kind": ["Right call", "Missed unhappy", "Wasted voucher", "Right call"]})
    kinds_cm = ["Right call", "Wasted voucher", "Missed unhappy"]
    heat = alt.Chart(cm).mark_rect(cornerRadius=8, stroke=ui.BG, strokeWidth=4).encode(
        x=alt.X("Action:N", title=None, axis=alt.Axis(orient="top"), sort=["Voucher", "No voucher"]),
        y=alt.Y("Actual:N", title=None, sort=["Dissatisfied", "Satisfied"], axis=alt.Axis(labelPadding=8)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=kinds_cm, range=["#1f2630", "#4a3a1c", "#4a2422"]),
                        legend=None),
        tooltip=["Actual", "Action", "kind", alt.Tooltip("n:Q", format=",")])
    pos = dict(x=alt.X("Action:N", sort=["Voucher", "No voucher"]), y=alt.Y("Actual:N", sort=["Dissatisfied", "Satisfied"]))
    text = alt.Chart(cm).mark_text(font="JetBrains Mono", fontSize=16, fontWeight=500, color=ui.TEXT, dy=-8).encode(
        **pos, text=alt.Text("n:Q", format=","))
    kind = alt.Chart(cm).mark_text(fontSize=11, color=ui.TEXT2, dy=12).encode(**pos, text="kind:N")
    st.altair_chart(ui.themed((heat + text + kind).properties(height=200)), width="stretch")

    st.markdown("#### Calibration")
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    cal = pd.DataFrame({"predicted": mean, "observed": frac})
    diag = alt.Chart(pd.DataFrame({"a": [0, 1]})).mark_line(strokeDash=[4, 4], color=ui.MUTED).encode(x="a:Q", y="a:Q")
    pts = alt.Chart(cal).mark_line(point=True, color=ui.MODEL_COLORS.get(model, ui.ACCENT)).encode(
        x=alt.X("predicted:Q", title="Predicted P(satisfied)"), y=alt.Y("observed:Q", title="Observed rate"))
    st.altair_chart(ui.themed((diag + pts).properties(height=220)), width="stretch")
