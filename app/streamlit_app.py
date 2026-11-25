"""SkyPulse - airline passenger satisfaction system.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="SkyPulse · Passenger Satisfaction", page_icon="✈", layout="wide")

import ui  # noqa: E402

# Streamlit drops widget state for widgets not on the current page; re-assigning
# the passenger keys keeps the same passenger across pages.
for k in [k for k in st.session_state if str(k).startswith("p_")]:
    st.session_state[k] = st.session_state[k]

ui.setup_page()
ui.init_passenger()
# First visit after a server start loads ~all models from disk; show progress instead of a blank page.
with st.spinner("Loading the trained models..."):
    ui.load_models()
    ui.load_variants()
    ui.load_trust()
    ui.load_test_predictions()

m = ui.load_metrics()
full = m.get("mode") == "full"

pages = [
    st.Page("views/overview.py", title="Project overview", icon=":material/dashboard:", default=True),
    st.Page("views/predict.py", title="Predict passenger", icon=":material/flight:"),
    st.Page("views/simulator.py", title="Service simulator", icon=":material/tune:"),
    st.Page("views/planner.py", title="Budget planner", icon=":material/savings:"),
    st.Page("views/policy.py", title="Decision policy", icon=":material/rule:"),
    st.Page("views/trust.py", title="Trust & fairness", icon=":material/verified_user:"),
    st.Page("views/explain.py", title="Explain the models", icon=":material/insights:"),
    st.Page("views/scoreboard.py", title="Model scoreboard", icon=":material/leaderboard:"),
    st.Page("views/lab.py", title="Experiment lab", icon=":material/science:"),
]
try:
    st.navigation(pages).run()
except Exception as exc:  # st.stop / st.rerun are BaseException subclasses, so they still pass through
    st.error(f"This page hit an unexpected error: {type(exc).__name__}: {exc}. "
             "The other pages still work; please report the details below.")
    with st.expander("Technical details"):
        st.exception(exc)

with st.sidebar:
    st.markdown('<div class="brand"><div class="name">SkyPulse</div><div class="meta">BMDS2114 · {mode}<br>'
                '{n:,} train · {t:,} hold-out</div></div>'.format(
                    mode="tuned models" if full else "prototype models", n=m["trained_on_rows"], t=m["test_rows"]),
                unsafe_allow_html=True)
    if not full:
        st.caption("Run `python train_full.py` for the full system.")
