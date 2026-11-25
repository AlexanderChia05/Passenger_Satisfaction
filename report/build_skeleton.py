"""Build the report SKELETON on the faculty cover template.

The skeleton holds structure only: chapter headings, page budgets, owners, the questions each
section must answer, the evidence (numbers / files) to cite, and figure placeholders. It contains
no drafted report prose: under the assignment's Yellow (Limited AI) category the report text,
arguments and critical analysis must be written by the team.

    python report/build_skeleton.py  ->  report/SkyPulse_Report_Skeleton.docx
"""
import json
import shutil
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(r"C:\Users\Alex\Downloads\ML_Assignment.docx")
OUT = ROOT / "report" / "SkyPulse_Report_Skeleton.docx"
EXP = ROOT / "artifacts" / "full" / "experiments"

GREY = RGBColor(0x59, 0x59, 0x59)
BLUE = RGBColor(0x1F, 0x4E, 0x79)

# ------------------------------------------------------------------ numbers from the artifacts
metrics = json.loads((ROOT / "artifacts" / "full" / "metrics.json").read_text())["models"]
partA = pd.read_csv(EXP / "cv_partA.csv").set_index("model")
final = pd.read_csv(EXP / "cv_final.csv").set_index("model")
trials = pd.read_csv(EXP / "tuning_trials.csv")
DEFAULT_ROW = {"knn": "1. Bagging-KNN (default)", "svm": "2. Bagging-SVM (default)",
               "mlp": "3. Seed-bagged MLP (default)", "xgb": "4. XGBoost (default)"}
BASE_ROW = {"knn": "1. KNN", "svm": "2. SVM (RBF, 20k rows)", "mlp": "3. MLP", "xgb": "4. Decision Tree"}


def f(x, d=4):
    return f"{x:.{d}f}"


def member_numbers(k):
    t = trials[trials.member == k]
    return {
        "base_cv": f(partA.loc[BASE_ROW[k], "F1"]), "ens_default_cv": f(partA.loc[DEFAULT_ROW[k], "F1"]),
        "ens_tuned_cv": f(final.loc[k, "F1"]), "ens_tuned_std": f(final.loc[k, "F1_std"]),
        "trial0": f(float(t.loc[t.number == 0, "value"].iloc[0])), "best": f(t.loc[t.state == "COMPLETE", "value"].max()),
        "best_trial": int(t.loc[t.value == t.loc[t.state == "COMPLETE", "value"].max(), "number"].iloc[0]),
        "pruned": int((t.state == "PRUNED").sum()),
        "base_hold": f(metrics[f"{k}_base"]["F1"]), "ens_hold": f(metrics[k]["F1"]),
        "auc_hold": f(metrics[k]["ROC-AUC"]), "mcc_hold": f(metrics[k]["MCC"]),
    }


# ------------------------------------------------------------------ helpers
doc = None


def para(text="", style=None, bold=False, italic=False, color=None, size=None):
    p = doc.add_paragraph(style=style)
    if text:
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        if color is not None:
            r.font.color.rgb = color
        if size:
            r.font.size = Pt(size)
    return p


def heading(text, level):
    return doc.add_heading(text, level=level)


def meta(owner, pages, rubric):
    p = para()
    for label, val in (("Owner: ", owner), ("   Budget: ", pages), ("   Rubric: ", rubric)):
        r = p.add_run(label)
        r.bold = True
        r.font.color.rgb = BLUE
        r.font.size = Pt(9)
        r = p.add_run(val)
        r.font.color.rgb = BLUE
        r.font.size = Pt(9)


def block(title, items, color=GREY):
    p = para(title, bold=True, color=color, size=10)
    p.paragraph_format.space_after = Pt(0)
    for it in items:
        q = doc.add_paragraph(style="List Bullet") if "List Bullet" in [s.name for s in doc.styles] else para()
        r = q.add_run(it if q.style.name == "List Bullet" else "• " + it)
        r.italic = True
        r.font.color.rgb = color
        r.font.size = Pt(10)


def write_about(items):
    block("Answer in your own words:", items)


def evidence(items):
    block("Evidence to cite (computed by the project, verify before use):", items, BLUE)


def borders(table):
    """Single grey borders on every cell (the template has no 'Table Grid' style)."""
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "0"), ("w:color", "8C8C8C")):
            e.set(qn(k), v)
        b.append(e)
    tblPr.append(b)


def figure(label, source):
    t = doc.add_table(rows=1, cols=1)
    borders(t)
    c = t.rows[0].cells[0]
    c.text = ""
    r = c.paragraphs[0].add_run(f"[ FIGURE: {label} ]")
    r.bold = True
    r.font.color.rgb = GREY
    r2 = c.add_paragraph().add_run(f"Insert: {source}")
    r2.italic = True
    r2.font.size = Pt(9)
    r2.font.color.rgb = GREY
    para(f"Figure X: {label}  (caption, then one or two sentences interpreting it)", italic=True, color=GREY, size=9)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def toc():
    """Word TOC field: right-click > Update Field in Word to fill it."""
    p = doc.add_paragraph()
    run = p.add_run()
    for tag, txt in (("begin", None), (None, 'TOC \\h \\z \\u'), ("separate", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        run._r.append(el)
    run.add_text("Right-click here and choose Update Field to build the table of contents.")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(end)


def set_fonts():
    """Spec: Times New Roman, 11 pt, single spacing."""
    for s in doc.styles:
        if s.type != 1:
            continue
        s.font.name = "Times New Roman"
        rpr = s.element.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.append(fonts)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            fonts.set(qn(a), "Times New Roman")
        if s.name.lower() == "normal":
            s.font.size = Pt(11)
            s.paragraph_format.line_spacing = 1.0
            s.paragraph_format.space_after = Pt(4)


# ------------------------------------------------------------------ build
def build():
    global doc
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, OUT)
    doc = Document(OUT)
    # The template (exported from Google Docs) stores page margins as decimal twips, e.g. "1411.2";
    # Word's schema expects integers, so round them before any layout call reads them.
    for pg in doc.element.body.iter(qn("w:pgMar")):
        for attr, val in list(pg.attrib.items()):
            try:
                pg.set(attr, str(round(float(val))))
            except ValueError:
                pass
    set_fonts()
    # Cover-page headings keep their look but leave the TOC: copy the heading's font size / bold onto
    # the runs, then switch the paragraph to the body style.
    for cp in doc.paragraphs:
        if cp.style.name.startswith("Heading") and cp.text.strip() in ("BMDS2114 Machine Learning", "Assignment Title"):
            size = cp.style.font.size or Pt(16)
            for r in cp.runs:
                r.bold = True
                r.font.size = r.font.size or size
            cp.style = doc.styles["normal"]

    # the template already ends with a page break after the cover, so no extra break here
    para("HOW TO USE THIS SKELETON", bold=True, size=12)
    for line in [
        "Grey italic text = questions your section must answer. Delete it once the section is written.",
        "Blue text = evidence produced by the project (numbers, files). Cite it; re-check numbers in the app.",
        "Boxes = figure placeholders with the exact file or app page to screenshot.",
        "Assignment AI category is Yellow (Limited AI): the report text, arguments and critical analysis must be "
        "your own. Declare all AI use in the AI Usage Disclosure Form.",
        "Format: Times New Roman 11 pt, single spacing, around 40 pages (below 50). Rubric is submitted as a separate file.",
    ]:
        para("• " + line, size=10)
    page_break()

    para("Table of Contents", bold=True, size=16)
    toc()
    page_break()

    # 1 -------------------------------------------------------------
    heading("1. Introduction", 1)
    meta("Team (lead: ___)", "3-4 pages", "Part A 1 Problem statement (10, team)")
    heading("1.1 Background", 2)
    write_about(["Why passenger satisfaction matters to an airline (cost of churn, loyalty, reputation).",
                 "Where survey data is used today and what it cannot tell an airline in time."])
    heading("1.2 Problem Statement", 2)
    write_about(["What exact problem does the airline face? Who is affected and what does it cost?",
                 "Why is this a machine-learning problem (binary classification) rather than a report or rule?"])
    heading("1.3 Objectives", 2)
    write_about(["3-5 measurable objectives (e.g. target F1 on unseen passengers, explainability, decision support).",
                 "Link each objective to a page or result you will show later."])
    heading("1.4 Scope and Limitations", 2)
    write_about(["What is in scope (data, four algorithm families, Streamlit system) and what is not.",
                 "Known limits: survey labels, one dataset, no live airline data."])
    heading("1.5 Proposed Solution and Project Title", 2)
    write_about(["Your project title and a one-paragraph summary of the solution.",
                 "What makes it different from existing work (link to Chapter 2's gap)."])
    evidence(["App: Project overview page lists what the system adds beyond a standard classifier."])
    page_break()

    # 2 -------------------------------------------------------------
    heading("2. Review of Related Systems and Technologies", 1)
    meta("Team (lead: ___)", "3-4 pages", "Part A 1 Problem statement: 'evaluation of the related system'")
    heading("2.1 Existing Approaches", 2)
    write_about(["Find and cite 4-6 real sources: published Kaggle notebooks on this dataset, academic papers on "
                 "airline satisfaction prediction, commercial customer-experience tools.",
                 "For each: method, reported result, and what it does NOT provide."])
    heading("2.2 Critical Evaluation", 2)
    write_about(["Comparison table: source | method | metric reported | strengths | weaknesses.",
                 "Your own judgement: common gaps (single metric, no uncertainty, no decision support, black box...)."])
    heading("2.3 Research Gap and Contribution", 2)
    write_about(["Which gaps does SkyPulse close? Tie each to a feature and a chapter."])
    page_break()

    # 3 -------------------------------------------------------------
    heading("3. Data Understanding and Preparation", 1)
    meta("Team (lead: ___)", "4-5 pages", "Checklist steps 2-4; Part A 3 'analyse patterns'")
    heading("3.1 Dataset Description", 2)
    evidence(["Kaggle teejmahal20/airline-passenger-satisfaction: train 103,904 rows, test 25,976 rows, 25 raw columns.",
              "4 categorical, 4 numeric, 14 service ratings (0-5), binary target; 'Unnamed: 0' and 'id' dropped.",
              "Table: reports/eda/01_columns.csv"])
    heading("3.2 Exploratory Data Analysis", 2)
    write_about(["Walk through the 15-item EDA checklist; for each finding say what it means for modelling."])
    evidence(["Target: 43.3% satisfied train, 43.9% test (mild imbalance).",
              "Missing: Arrival Delay NaN 310 train + 83 test (0.30%); rating 0 = 'not applicable' (up to 5.10%).",
              "Delays skew ~6.7; departure vs arrival delay r = 0.965; all train/test PSI < 0.1.",
              "Top mutual information: Online boarding, Inflight wifi service, Class, Type of Travel.",
              "Full list: reports/eda/eda_summary.md"])
    figure("Target class balance", "reports/eda/03_target_balance.png")
    figure("Correlation heatmap of service ratings", "reports/eda/10_correlation_heatmap.png")
    figure("Satisfaction by passenger segment", "reports/eda/11_categorical_vs_target.png")
    figure("Mutual information ranking", "reports/eda/12_mutual_information.png")
    heading("3.3 Data Preparation Decisions", 2)
    write_about(["Explain each preprocessing decision and the EDA finding that justifies it.",
                 "Why all fitting happens inside each CV fold (no leakage)."])
    evidence(["Findings -> decisions table: reports/eda/eda_summary.md (last section).",
              "Code: src/features.py AirlineFeatures (25 engineered features)."])
    page_break()

    # 4 -------------------------------------------------------------
    heading("4. Methodology and System Design", 1)
    meta("Team (lead: ___)", "3-4 pages", "Part B 2 System implementation (15, team)")
    heading("4.1 Machine Learning Workflow", 2)
    write_about(["Map your work onto the 8-step ML project checklist from the specification."])
    heading("4.2 System Architecture", 2)
    write_about(["Diagram: data -> feature pipeline -> 4 member models -> evaluation -> artifacts -> Streamlit pages."])
    figure("System architecture diagram", "draw it yourselves (draw.io / PowerPoint)")
    heading("4.3 Validation Design", 2)
    evidence(["Official split: 103,904 train (80%) / 25,976 hold-out test (20%), test used once at the end.",
              "Tuning: stratified 30,000-row subsample, 3-fold inner CV, 60 Optuna TPE trials per member, median pruning.",
              "Final CV: stratified 5-fold on all 103,904 rows (about 83,124 train / 20,781 validation per fold).",
              "Studies (missing values, features, learning curves): 30,000-row subsample, 3-fold."])
    heading("4.4 Evaluation Metrics and Baselines", 2)
    write_about(["Why each of the six metrics (Accuracy, Recall, Precision, F1, ROC-AUC, MCC) is needed.",
                 "Why a Dummy and a business-rule baseline."])
    evidence([f"Dummy accuracy {f(metrics['dummy']['Accuracy'])}; rule baseline F1 {f(metrics['rule']['F1'])}."])
    heading("4.5 Equal-Workload Design", 2)
    write_about(["How the four members received identical data, budget and CV so results are comparable."])
    page_break()

    # 5 -------------------------------------------------------------
    heading("5. Individual Model Development", 1)
    para("One sub-chapter per member, written by that member (individual marks: Programming 20, Model optimisation 10, "
         "Results 10).", italic=True, color=GREY, size=10)
    members = [
        ("knn", "Member 1", "K-Nearest Neighbours", "Bagging-KNN (random subspace)", "NCA metric learning",
         "src/members/m1_knn.py, src/estimators.py SubsampledNCA, src/trust.py NeighbourOOD"),
        ("svm", "Member 2", "Support Vector Machine", "Bagging-SVM with probability calibration",
         "Nystroem kernel approximation + LinearSVC; Platt vs isotonic calibration",
         "src/members/m2_svm.py, src/estimators.py SubsampleFit"),
        ("mlp", "Member 3", "Multi-layer Perceptron", "Seed-bagged MLP soft vote", "Bootstrap-bagged MLP, architecture search",
         "src/members/m3_mlp.py"),
        ("xgb", "Member 4", "Decision Tree", "XGBoost (gradient boosting, early stopping)",
         "Cost-complexity pruning, monotone constraints, AdaBoost",
         "src/members/m4_tree.py, src/estimators.py XGBEarlyStop / PrunedTree"),
    ]
    for i, (k, owner, base, ens, adv, code) in enumerate(members, 1):
        n = member_numbers(k)
        heading(f"5.{i} {owner}: {base} and {ens}", 2)
        meta(owner, "4 pages", "Programming 20 / Model optimisation 10 / Results 10 (individual)")
        heading(f"5.{i}.1 Algorithm Theory", 3)
        write_about([f"How {base} learns and predicts; its key hyperparameters; strengths and weaknesses on this data."])
        heading(f"5.{i}.2 Part A: Base Model and Default Ensemble", 3)
        write_about([f"Why {ens} suits this algorithm (bias or variance? what problem does it fix?).",
                     "Interpret the Part A numbers: did the ensemble help with default settings, and why?"])
        evidence([f"5-fold CV F1: base {n['base_cv']} -> default ensemble {n['ens_default_cv']}.",
                  "App: Experiment lab -> Part A."])
        heading(f"5.{i}.3 Part B: Hyperparameter Tuning", 3)
        write_about(["Search space and why each range was chosen; why Optuna TPE instead of grid search.",
                     "Read the convergence curve: when did it plateau, and is 60 trials enough?"])
        evidence([f"Inner-CV F1 trial 0 (defaults) {n['trial0']} -> best {n['best']} at trial {n['best_trial']}/60; "
                  f"{n['pruned']} trials pruned.",
                  f"5-fold CV on all rows: default {n['ens_default_cv']} -> tuned {n['ens_tuned_cv']} "
                  f"(std {n['ens_tuned_std']}).",
                  "Tuned values: artifacts/full/experiments/tuned_params_table.csv"])
        figure("Before vs after tuning, and tuning curve",
               "screenshot App -> Experiment lab -> Tuning (Focus = your member)")
        heading(f"5.{i}.4 Advanced Technique", 3)
        write_about([f"What {adv} does, why you tried it, and whether it helped (with numbers)."])
        evidence(["artifacts/full/experiments/cv_final.csv (rows for your member); App -> Experiment lab -> Part B final."])
        heading(f"5.{i}.5 Results on the Hold-out Set", 3)
        write_about(["Interpret all six metrics, the confusion matrix and where your model makes errors.",
                     "Compare with the baselines and the other members."])
        evidence([f"Hold-out F1: base {n['base_hold']} -> tuned ensemble {n['ens_hold']}; "
                  f"ROC-AUC {n['auc_hold']}; MCC {n['mcc_hold']}.",
                  "App: Model scoreboard -> Overview and -> Error analysis (select your model)."])
        figure("Confusion matrix and errors by segment", "screenshot App -> Model scoreboard -> Error analysis")
        heading(f"5.{i}.6 Implementation Walkthrough", 3)
        write_about(["Explain your key functions in your own words (what, why, how errors are handled).",
                     "Show short code excerpts only where they support the explanation."])
        evidence([f"Code: {code}"])
        page_break()

    # 6 -------------------------------------------------------------
    heading("6. Model Comparison, Selection and Combination", 1)
    meta("Team (lead: ___)", "3-4 pages", "Part A 7 shortlist; Part B 2 combine models; Results")
    write_about(["Rank the models; are the differences statistically significant?",
                 "Which model does the system use and why (accuracy, speed, explainability)?",
                 "Combining different models: what the team soft vote / stacking achieved and why it was or was not "
                 "chosen for the app.",
                 "Learning curves: bias or variance for each member?"])
    evidence([f"Hold-out F1: KNN {f(metrics['knn']['F1'])}, SVM {f(metrics['svm']['F1'])}, MLP {f(metrics['mlp']['F1'])}, "
              f"XGBoost {f(metrics['xgb']['F1'])}, team soft vote {f(metrics['team']['F1'])}.",
              "Statistics: artifacts/full/experiments/stat_mcnemar.csv, stat_5x2cv.csv, stat_bootstrap.csv.",
              "Team weights: artifacts/full/experiments/team.json; error overlap in Scoreboard -> Error analysis."])
    figure("Learning curves of the four member ensembles", "reports/training/learning_curves.png")
    figure("Bootstrap 95% confidence intervals", "screenshot App -> Experiment lab -> Statistics")
    page_break()

    # 7 -------------------------------------------------------------
    heading("7. System Implementation", 1)
    meta("Team (lead: ___)", "4-5 pages", "Part A 3 Degree of completion; Part B 2 System implementation; Output")
    write_about(["One short section per page: purpose, inputs, outputs, how a user acts on it.",
                 "Differences from the original proposal and the reason for each change."])
    for page in ["Project overview", "Predict passenger", "Service simulator", "Budget planner", "Decision policy",
                 "Trust & fairness", "Explain the models", "Model scoreboard", "Experiment lab"]:
        figure(f"{page} page", f"screenshot App -> {page}")
    page_break()

    # 8 -------------------------------------------------------------
    heading("8. Reliability, Fairness and Error Analysis", 1)
    meta("Team (lead: ___)", "2-3 pages", "Part A 2 robustness, Part A 4 errors, Part B 3 reliability")
    write_about(["Conformal coverage: does the guarantee hold?", "Fairness gaps by group and what you would do about them.",
                 "Robustness to noise and hidden ratings.", "Error analysis: which passengers are hard and why."])
    evidence(["App: Trust & fairness (all tabs); Experiment lab -> Robustness; Scoreboard -> Error analysis."])
    page_break()

    # 9-10 ----------------------------------------------------------
    heading("9. Conclusion and Future Work", 1)
    meta("Team", "1-2 pages", "Organization (5)")
    write_about(["Answer each objective from 1.3 with evidence.", "Limitations and concrete next steps."])
    heading("10. Task Allocation and Project Plan", 1)
    meta("Team", "1-2 pages", "Required by the specification")
    t = doc.add_table(rows=5, cols=4)
    borders(t)
    for j, h in enumerate(["Member", "Algorithm / ensemble", "Shared modules owned", "Report sections"]):
        t.rows[0].cells[j].text = h
    for r, (k, owner, base, ens, *_rest) in enumerate(members, 1):
        t.rows[r].cells[0].text = owner
        t.rows[r].cells[1].text = f"{base} / {ens}"
    para()
    write_about(["Gantt chart of the project weeks (proposal, EDA, Part A, Part B, system, report, presentation)."])
    figure("Project plan (Gantt chart)", "draw it yourselves")

    heading("References", 1)
    write_about(["APA or IEEE style, consistently. Cite the dataset, libraries (scikit-learn, XGBoost, Optuna, "
                 "Streamlit) and every source used in Chapter 2."])
    heading("Appendix A: How to Run the System", 1)
    evidence(["README.md: pip install -r requirements.txt; python -m streamlit run app/streamlit_app.py; "
              "retraining with python train_full.py."])
    heading("Appendix B: Tuned Hyperparameters", 1)
    evidence(["artifacts/full/experiments/tuned_params_table.csv"])
    heading("Appendix C: AI Usage Declaration", 1)
    write_about(["Summarise which AI tools were used and for what; the full AI Usage Disclosure Form is submitted "
                 "as required by the specification."])

    doc.save(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    build()
