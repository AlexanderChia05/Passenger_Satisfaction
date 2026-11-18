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
with st.sidebar:
    st.markdown(f'<div class="board" style="border:none;margin:0"><div><div class="code">BMDS2114 · '
                f'{"TUNED MODELS" if full else "PROTOTYPE"}</div>'
                '<h1 style="font-size:2rem">SkyPulse</h1></div></div>', unsafe_allow_html=True)

pages = [
    st.Page("views/overview.py", title="Project overview", icon="🗺️", default=True),
    st.Page("views/predict.py", title="Predict passenger", icon="✈️"),
    st.Page("views/simulator.py", title="Service simulator", icon="🎚️"),
    st.Page("views/planner.py", title="Budget planner", icon="💰"),
    st.Page("views/policy.py", title="Decision policy", icon="⚖️"),
    st.Page("views/trust.py", title="Trust & fairness", icon="🛡️"),
    st.Page("views/explain.py", title="Explain the models", icon="🧭"),
    st.Page("views/scoreboard.py", title="Model scoreboard", icon="🏁"),
    st.Page("views/lab.py", title="Experiment lab", icon="🧪"),
]
try:
    st.navigation(pages).run()
except Exception as exc:  # st.stop / st.rerun are BaseException subclasses, so they still pass through
    st.error(f"This page hit an unexpected error: {type(exc).__name__}: {exc}. "
             "The other pages still work; please report the details below.")
    with st.expander("Technical details"):
        st.exception(exc)

with st.sidebar:
    if full:
        st.caption("Tuned models trained on {n:,} passengers · scored on {m:,} hold-out passengers.".format(n=m["trained_on_rows"], m=m["test_rows"]))
    else:
        st.caption(("Prototype models: untuned, trained on {n:,} rows · scored on {m:,} hold-out passengers. "
                     "Run `python train_full.py` for the full system.").format(n=m["trained_on_rows"], m=m["test_rows"]))
