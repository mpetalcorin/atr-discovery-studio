"""
ATR Discovery Studio
Advanced Streamlit app for ATR inhibitor data cleaning, chemical descriptor analysis,
classification, explainability, and candidate prioritisation.

Input CSV formats supported:
1) Training set: SMILES column plus activity/class column, for example:
   Ligand SMILES, IC50 (nM), class
2) Screening set: SMILES only or SMILES plus metadata, for example:
   Generated_SMILES

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, QED, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="ATR Discovery Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .metric-card {border: 1px solid rgba(120,120,120,0.25); border-radius: 18px; padding: 18px;}
    .small-note {font-size: 0.86rem; color: #6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)

SMILES_CANDIDATES = ["Ligand SMILES", "SMILES", "smiles", "Generated_SMILES", "canonical_smiles"]
ACTIVITY_CANDIDATES = ["IC50 (nM)", "activity_nM", "Ki (nM)", "Kd (nM)", "Activity", "activity"]
CLASS_CANDIDATES = ["class", "activity_label", "label", "Active", "active"]

SAMPLE_SMILES = [
    ("CC1=NC=C(C=C1)C(=O)NCC2=CC=CC=C2", 120, 1),
    ("COC1=CC=C(C=C1)C(=O)NC2=NC=CC=C2", 450, 1),
    ("CCOC(=O)N1CCC(CC1)NC2=NC=NC3=CC=CC=C23", 35, 1),
    ("CC(C)NC(=O)C1=CC=C(NC2=NC=NC=C2)C=C1", 820, 1),
    ("CCCCCCCCCCCC", 15000, 0),
    ("CCOC(=O)C1=CC=CC=C1", 7000, 0),
    ("CCN(CC)CCOC(=O)C1=CC=CC=C1", 3200, 0),
    ("O=C(NCC1=CC=CC=C1)C2=CC=CC=C2", 2100, 0),
    ("CC1=C(C=CC=N1)NC(=O)C2=CN=CC=C2", 260, 1),
    ("CC(C)(C)OC(=O)NCC1=NC=CC=C1", 5000, 0),
    ("CN1CCN(CC1)C(=O)C2=CC=CC=C2", 1400, 0),
    ("COC1=CC2=C(C=C1)N=CN=C2NCC3=CC=CC=C3", 95, 1),
]


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(SAMPLE_SMILES, columns=["Ligand SMILES", "IC50 (nM)", "class"])


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def mol_from_smiles(smiles: str):
    if not RDKIT_AVAILABLE or not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def basic_descriptors(smiles: str) -> Dict[str, float]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {
            "valid_molecule": 0, "MolWt": np.nan, "LogP": np.nan, "TPSA": np.nan,
            "HBD": np.nan, "HBA": np.nan, "RotBonds": np.nan, "AromaticRings": np.nan,
            "QED": np.nan, "HeavyAtoms": np.nan, "FractionCSP3": np.nan,
            "Lipinski_violations": np.nan, "Druglike": 0,
        }
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    aro = Lipinski.NumAromaticRings(mol)
    qed = QED.qed(mol)
    heavy = mol.GetNumHeavyAtoms()
    csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return {
        "valid_molecule": 1, "MolWt": mw, "LogP": logp, "TPSA": tpsa,
        "HBD": hbd, "HBA": hba, "RotBonds": rot, "AromaticRings": aro,
        "QED": qed, "HeavyAtoms": heavy, "FractionCSP3": csp3,
        "Lipinski_violations": violations, "Druglike": int(violations == 0 and qed >= 0.45),
    }


@st.cache_data(show_spinner=False)
def add_descriptors(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    if not RDKIT_AVAILABLE:
        return df.copy()
    desc = [basic_descriptors(s) for s in df[smiles_col].astype(str).tolist()]
    desc_df = pd.DataFrame(desc)
    out = pd.concat([df.reset_index(drop=True), desc_df], axis=1)
    return out[out["valid_molecule"] == 1].reset_index(drop=True)


def normalise_activity(df: pd.DataFrame, activity_col: Optional[str], class_col: Optional[str], threshold: float) -> pd.DataFrame:
    out = df.copy()
    if activity_col and activity_col in out.columns:
        out["activity_nM"] = pd.to_numeric(out[activity_col], errors="coerce")
        out["pActivity"] = -np.log10(out["activity_nM"] * 1e-9)
        if not class_col:
            out["class"] = (out["activity_nM"] < threshold).astype(int)
    if class_col and class_col in out.columns:
        out["class"] = pd.to_numeric(out[class_col], errors="coerce").fillna(0).astype(int)
    return out


def descriptor_columns(df: pd.DataFrame) -> List[str]:
    preferred = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "AromaticRings", "QED", "HeavyAtoms", "FractionCSP3", "Lipinski_violations", "Druglike"]
    return [c for c in preferred if c in df.columns]


def model_dictionary() -> Dict[str, object]:
    models = {
        "Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random forest": RandomForestClassifier(n_estimators=350, random_state=42, class_weight="balanced", max_depth=None),
        "Gradient boosting": GradientBoostingClassifier(random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, eval_metric="logloss", random_state=42,
        )
    return models


def build_pipeline(model, k_best: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k_best)),
        ("model", model),
    ])


def evaluate_models(df: pd.DataFrame, features: List[str], label_col: str = "class"):
    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[label_col].astype(int)
    k_best = min(len(features), max(2, len(features)))
    rows = []
    trained = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    for name, model in model_dictionary().items():
        pipe = build_pipeline(model, k_best=k_best)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        if hasattr(pipe[-1], "predict_proba"):
            prob = pipe.predict_proba(X_test)[:, 1]
        else:
            prob = pred
        rows.append({
            "model": name,
            "ROC_AUC": roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else np.nan,
            "PR_AUC": average_precision_score(y_test, prob) if len(np.unique(y_test)) > 1 else np.nan,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
        })
        trained[name] = (pipe, X_test, y_test, prob, pred)
    return pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False), trained


def molecule_grid(smiles: List[str], legends: List[str], mols_per_row: int = 4) -> Optional[bytes]:
    if not RDKIT_AVAILABLE:
        return None
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [m for m in mols if m is not None]
    if not mols:
        return None
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, legends=legends[:len(mols)], subImgSize=(260, 190))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def download_csv_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")


with st.sidebar:
    st.title("🧬 ATR Discovery Studio")
    st.caption("Advanced analysis app for ATR inhibitor datasets, ChemGPT/SELFIES candidates, and oncology drug-discovery triage.")
    uploaded = st.file_uploader("Upload ATR CSV", type=["csv", "tsv"])
    sep = st.selectbox("File separator", [",", "\t"], index=0)
    threshold = st.number_input("Active threshold, nM", min_value=1.0, value=1000.0, step=50.0)
    use_sample = st.checkbox("Use built-in sample dataset", value=uploaded is None)

if uploaded is not None:
    raw_df = pd.read_csv(uploaded, sep=sep)
elif use_sample:
    raw_df = sample_data()
else:
    st.info("Upload a CSV file or enable the sample dataset.")
    st.stop()

smiles_col = first_existing_column(raw_df, SMILES_CANDIDATES)
activity_col = first_existing_column(raw_df, ACTIVITY_CANDIDATES)
class_col = first_existing_column(raw_df, CLASS_CANDIDATES)

st.title("ATR Inhibitor Discovery, Analysis and Visualisation App")
st.markdown(
    "This app turns an ATR inhibitor dataset into a decision-support workflow: data cleaning, drug-likeness analysis, chemical-space mapping, model comparison, explainability, and candidate prioritisation."
)

if smiles_col is None:
    st.error("No SMILES column found. Please include a column such as 'Ligand SMILES', 'SMILES', or 'Generated_SMILES'.")
    st.stop()

if not RDKIT_AVAILABLE:
    st.error("RDKit is not installed. Install the requirements file, then rerun the app.")
    st.stop()

raw_df = raw_df.dropna(subset=[smiles_col]).copy()
raw_df = normalise_activity(raw_df, activity_col, class_col, threshold)
with st.spinner("Computing molecular descriptors..."):
    df = add_descriptors(raw_df, smiles_col)

features = descriptor_columns(df)
has_labels = "class" in df.columns and df["class"].nunique() == 2
has_activity = "activity_nM" in df.columns

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows loaded", f"{len(raw_df):,}")
m2.metric("Valid molecules", f"{len(df):,}")
m3.metric("Drug-like", f"{int(df.get('Druglike', pd.Series(dtype=int)).sum()):,}")
m4.metric("Labelled classes", "Yes" if has_labels else "No")

if len(df) == 0:
    st.error("No valid molecules remained after SMILES parsing.")
    st.stop()

tabs = st.tabs([
    "Dataset overview", "Drug-likeness", "Chemical space", "Model analysis", "Candidate prioritisation", "Molecule viewer", "Export"
])

with tabs[0]:
    st.subheader("Cleaned dataset")
    st.dataframe(df.head(1000), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        if has_labels:
            counts = df["class"].map({1: "Active", 0: "Inactive"}).value_counts().reset_index()
            counts.columns = ["Class", "Count"]
            fig = px.bar(counts, x="Class", y="Count", text="Count", title="Active versus inactive molecules")
            st.plotly_chart(fig, use_container_width=True)
        elif has_activity:
            st.plotly_chart(px.histogram(df, x="activity_nM", nbins=40, title="Activity distribution, nM", log_y=True), use_container_width=True)
    with c2:
        if has_activity:
            st.plotly_chart(px.histogram(df, x="pActivity", nbins=40, title="pActivity distribution"), use_container_width=True)
        else:
            st.plotly_chart(px.histogram(df, x="MolWt", nbins=40, title="Molecular-weight distribution"), use_container_width=True)

with tabs[1]:
    st.subheader("Drug-likeness and developability profile")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            df, x="MolWt", y="LogP", color="class" if has_labels else "Druglike",
            size="QED", hover_data=[smiles_col, "QED", "TPSA", "Lipinski_violations"],
            title="Molecular weight versus LogP"
        )
        fig.add_vline(x=500, line_dash="dash")
        fig.add_hline(y=5, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            df, x="TPSA", y="QED", color="Lipinski_violations",
            hover_data=[smiles_col, "MolWt", "LogP", "HBD", "HBA"],
            title="Polar surface area versus QED"
        )
        st.plotly_chart(fig, use_container_width=True)
    props = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "QED"]
    summary = df[props].describe().T.reset_index().rename(columns={"index":"Property"})
    st.dataframe(summary, use_container_width=True)

with tabs[2]:
    st.subheader("Chemical-space map")
    X = df[features].replace([np.inf, -np.inf], np.nan)
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    map_df = df.copy()
    map_df["PC1"] = coords[:, 0]
    map_df["PC2"] = coords[:, 1]
    colour = "class" if has_labels else ("ATR_probability" if "ATR_probability" in df.columns else "QED")
    fig = px.scatter(
        map_df, x="PC1", y="PC2", color=colour, size="QED",
        hover_data=[smiles_col, "MolWt", "LogP", "QED", "Lipinski_violations"],
        title=f"PCA map, variance explained: PC1 {pca.explained_variance_ratio_[0]:.1%}, PC2 {pca.explained_variance_ratio_[1]:.1%}"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Use this map to spot chemical clusters, outliers, active regions, and candidate molecules that sit near known active chemistry.")

with tabs[3]:
    st.subheader("Machine-learning model comparison")
    if not has_labels:
        st.info("This tab needs a binary class column, for example class = 1 for active and class = 0 for inactive. If IC50/Ki/Kd is present, the app creates this label from the active threshold.")
    elif df["class"].value_counts().min() < 2:
        st.warning("Each class needs at least two molecules for model evaluation.")
    else:
        metrics, trained = evaluate_models(df, features)
        st.dataframe(metrics, use_container_width=True)
        st.plotly_chart(px.bar(metrics, x="model", y=["ROC_AUC", "PR_AUC", "F1", "Recall"], barmode="group", title="Model performance comparison"), use_container_width=True)
        best_name = metrics.iloc[0]["model"]
        pipe, X_test, y_test, prob, pred = trained[best_name]
        c1, c2 = st.columns(2)
        with c1:
            fpr, tpr, _ = roc_curve(y_test, prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=best_name))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
            fig.update_layout(title=f"ROC curve, {best_name}", xaxis_title="False positive rate", yaxis_title="True positive rate")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            prec, rec, _ = precision_recall_curve(y_test, prob)
            fig = go.Figure(go.Scatter(x=rec, y=prec, mode="lines", name=best_name))
            fig.update_layout(title=f"Precision-recall curve, {best_name}", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig, use_container_width=True)
        cm = confusion_matrix(y_test, pred)
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="True"), title=f"Confusion matrix, {best_name}")
        st.plotly_chart(fig, use_container_width=True)
        selector = pipe.named_steps["select"]
        selected = np.array(features)[selector.get_support()]
        model = pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({"Feature": selected, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        elif hasattr(model, "coef_"):
            imp = pd.DataFrame({"Feature": selected, "Importance": np.abs(model.coef_[0])}).sort_values("Importance", ascending=False)
        else:
            imp = pd.DataFrame({"Feature": selected, "Importance": np.nan})
        st.plotly_chart(px.bar(imp.head(20), x="Importance", y="Feature", orientation="h", title="Top model features"), use_container_width=True)

with tabs[4]:
    st.subheader("Candidate prioritisation")
    score_df = df.copy()
    score_df["developability_score"] = (
        0.35 * score_df["QED"].fillna(0) +
        0.25 * (score_df["Lipinski_violations"].fillna(4).rsub(4) / 4) +
        0.20 * (1 - np.clip(np.abs(score_df["LogP"].fillna(2.5) - 2.5) / 5, 0, 1)) +
        0.20 * (1 - np.clip(np.abs(score_df["MolWt"].fillna(350) - 350) / 350, 0, 1))
    )
    if has_activity:
        score_df["potency_score"] = 1 / (1 + np.log10(score_df["activity_nM"].clip(lower=1)))
        score_df["combined_score"] = 0.55 * score_df["potency_score"] + 0.45 * score_df["developability_score"]
    else:
        score_df["combined_score"] = score_df["developability_score"]
    top = score_df.sort_values("combined_score", ascending=False).head(12)
    st.dataframe(top[[smiles_col, "combined_score", "QED", "MolWt", "LogP", "TPSA", "Lipinski_violations"] + (["activity_nM"] if has_activity else [])], use_container_width=True)
    st.plotly_chart(px.bar(top, x="combined_score", y=smiles_col, orientation="h", title="Top prioritised molecules"), use_container_width=True)
    img = molecule_grid(top[smiles_col].astype(str).tolist(), [f"score={s:.2f}" for s in top["combined_score"]])
    if img:
        st.image(img, caption="Top prioritised chemical structures")

with tabs[5]:
    st.subheader("Molecule viewer")
    n = min(24, len(df))
    sort_by = st.selectbox("Sort molecules by", ["QED", "MolWt", "LogP", "TPSA", "Lipinski_violations"] + (["activity_nM"] if has_activity else []))
    ascending = st.checkbox("Ascending", value=sort_by == "activity_nM")
    view_df = df.sort_values(sort_by, ascending=ascending).head(n)
    legends = []
    for _, r in view_df.iterrows():
        label = f"QED {r['QED']:.2f}"
        if has_activity and not pd.isna(r.get("activity_nM")):
            label += f" | {r['activity_nM']:.0f} nM"
        legends.append(label)
    img = molecule_grid(view_df[smiles_col].astype(str).tolist(), legends, mols_per_row=4)
    if img:
        st.image(img)

with tabs[6]:
    st.subheader("Export results")
    download_csv_button(df, "Download cleaned descriptor table", "atr_cleaned_descriptor_table.csv")
    if 'top' in locals():
        download_csv_button(top, "Download top prioritised molecules", "atr_top_prioritised_molecules.csv")
    st.markdown("""
    **Suggested next wet-lab validation:** confirm hit identity and purity, test ATR biochemical inhibition, measure cell viability in ATR-dependent backgrounds, compare sensitive versus resistant genetic contexts, then assess replication-stress markers such as γH2AX, pCHK1, and RAD51 foci.
    """)
