# ATR Discovery Studio

A Streamlit app for ATR inhibitor discovery analysis, based on the uploaded ATR drug discovery notebook workflow.

## What the app does

- Loads ATR inhibitor training data or ChemGPT/SELFIES generated candidate molecules.
- Detects SMILES, activity and class columns automatically.
- Computes RDKit descriptors, including molecular weight, LogP, TPSA, hydrogen bond donors/acceptors, rotatable bonds, aromatic rings, QED and Lipinski violations.
- Visualises active versus inactive compounds, potency distributions, drug-likeness, chemical space and candidate molecules.
- Trains and compares machine-learning models, including logistic regression, random forest, gradient boosting and XGBoost if installed.
- Shows ROC curves, precision-recall curves, confusion matrix and feature importance.
- Prioritises molecules using potency and developability scores.
- Exports cleaned descriptor tables and prioritised molecules.

## Expected input columns

The app can detect common column names automatically.

Training dataset examples:

```csv
Ligand SMILES,IC50 (nM),class
CC1=NC=C(C=C1)C(=O)NCC2=CC=CC=C2,120,1
CCCCCCCCCCCC,15000,0
```

Candidate screening examples:

```csv
Generated_SMILES
CCOC(=O)N1CCC(CC1)NC2=NC=NC3=CC=CC=C23
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

This app is for research prioritisation only. Predictions should be followed by orthogonal experimental validation, including biochemical ATR inhibition assays, cell-based replication-stress assays, genetic-context selectivity testing and medicinal chemistry review.
