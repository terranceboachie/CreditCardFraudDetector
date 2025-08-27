from google.colab import drive
drive.mount('/content/drive')

# 👇 Change this to your preferred folder in Drive
BASE_DIR = "/content/drive/MyDrive/fraud-detector"
DATA_PATH = f"{BASE_DIR}/data/creditcard.csv"

import os
os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
os.makedirs(f"{BASE_DIR}/reports", exist_ok=True)

print("Using BASE_DIR =", BASE_DIR)


!pip -q install pandas numpy scikit-learn matplotlib joblib gradio


import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, precision_recall_curve, auc,
                             average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib, os

BASE_DIR = Path(BASE_DIR)
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Load
df = pd.read_csv(DATA_PATH)
assert "Class" in df.columns, "CSV must contain a 'Class' column (0 legit, 1 fraud)."
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

# --- Splits (train/val/test)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

# --- Preprocessor
scale_cols = [c for c in ["Time", "Amount"] if c in X.columns]
pre = ColumnTransformer([("scale", RobustScaler(), scale_cols)], remainder="passthrough", verbose_feature_names_out=False)

# --- Class weights (severe imbalance)
classes = np.array([0,1])
cw = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {int(c): w for c, w in zip(classes, cw)}

# --- Candidates
candidates = {
    "logreg": LogisticRegression(max_iter=2000, class_weight=class_weight, n_jobs=None),
    "rf": RandomForestClassifier(
        n_estimators=400, class_weight="balanced_subsample", n_jobs=-1, random_state=42
    ),
}

def pr_auc(y_true, scores):
    p, r, _ = precision_recall_curve(y_true, scores)
    return auc(r, p)

best_name, best_score, best_pipe = None, -1, None
val_scores_map = {}

for name, clf in candidates.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    scores_val = pipe.predict_proba(X_val)[:, 1]
    score = pr_auc(y_val, scores_val)
    val_scores_map[name] = float(score)
    if score > best_score:
        best_name, best_score, best_pipe = name, score, pipe

# --- Threshold tuning (maximize F1 on validation)
p, r, thr = precision_recall_curve(y_val, best_pipe.predict_proba(X_val)[:, 1])
f1 = 2*(p*r)/(p+r+1e-12)
idx = int(np.nanargmax(f1))
threshold = float(np.append(thr, 1.0)[idx])  # align lengths

# --- Evaluate on test
test_scores = best_pipe.predict_proba(X_test)[:, 1]
y_pred = (test_scores >= threshold).astype(int)

cr = classification_report(y_test, y_pred, digits=3)
cm = confusion_matrix(y_test, y_pred)
ap = average_precision_score(y_test, test_scores)

print(f"Best model: {best_name} | Val PR-AUC: {best_score:.3f}")
print(f"Chosen threshold (max F1 on val): {threshold:.4f}")
print("\nClassification Report (test):\n", cr)
print("\nConfusion Matrix (test):\n", cm)
print(f"\nAverage Precision (test): {ap:.3f}")

# --- Plots saved to Drive (great for LinkedIn carousel)
fpr, tpr, _ = roc_curve(y_test, test_scores)
plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - Test")
plt.savefig(REPORTS_DIR / "roc_test.png", bbox_inches="tight"); plt.close()

prec, rec, _ = precision_recall_curve(y_test, test_scores)
plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - Test (AP={ap:.3f})")
plt.savefig(REPORTS_DIR / "pr_test.png", bbox_inches="tight"); plt.close()

plt.figure(); im = plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix - Test"); plt.colorbar(im)
plt.xticks([0,1], ["Legit","Fraud"]); plt.yticks([0,1], ["Legit","Fraud"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.savefig(REPORTS_DIR / "cm_test.png", bbox_inches="tight"); plt.close()

# --- Save artifacts to Drive
joblib.dump(best_pipe, MODELS_DIR / "best_pipeline.joblib")
with open(MODELS_DIR / "threshold.json", "w") as f:
    json.dump({"threshold": threshold, "chosen_model": best_name, "validation_pr_auc": best_score, "test_ap": float(ap)}, f, indent=2)

# Optional: feature importances/coefficients
try:
    clf = best_pipe.named_steps["clf"]
    pre = best_pipe.named_steps["pre"]
    feat_names = pre.get_feature_names_out()
    import pandas as pd
    if hasattr(clf, "feature_importances_"):
        pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False).head(30)\
            .to_csv(REPORTS_DIR / "top_features.csv")
    elif hasattr(clf, "coef_"):
        coef = pd.Series(clf.coef_[0], index=feat_names).sort_values(ascending=False)
        pd.concat([coef.head(15), coef.tail(15)]).to_csv(REPORTS_DIR / "top_features.csv")
except Exception as e:
    print("Feature importances not available:", e)


import json, joblib, pandas as pd, numpy as np, gradio as gr
from pathlib import Path

pipe = joblib.load(MODELS_DIR / "best_pipeline.joblib")
thr = json.load(open(MODELS_DIR / "threshold.json"))["threshold"]

def score_csv(file, threshold=thr):
    df = pd.read_csv(file.name)
    X = df.drop(columns=["Class"]) if "Class" in df.columns else df.copy()
    probs = pipe.predict_proba(X)[:,1]
    preds = (probs >= threshold).astype(int)
    out = X.copy()
    out["fraud_prob"] = probs
    out["pred"] = preds
    if "Class" in df.columns:
        y = df["Class"].astype(int)
        tp = int(((preds==1)&(y==1)).sum()); fp = int(((preds==1)&(y==0)).sum())
        fn = int(((preds==0)&(y==1)).sum()); tn = int(((preds==0)&(y==0)).sum())
        summary = f"Rows: {len(df)} | PredFraud: {int(preds.sum())} | Precision={tp/max(tp+fp,1):.2%} | Recall={tp/max(tp+fn,1):.2%}"
    else:
        summary = f"Rows: {len(df)} | PredFraud: {int(preds.sum())}"
    return summary, out.sort_values("fraud_prob", ascending=False).head(25)

demo = gr.Interface(
    fn=score_csv,
    inputs=[gr.File(label="Upload CSV"), gr.Slider(0.01, 0.99, value=float(round(thr,2)), step=0.01, label="Decision threshold")],
    outputs=[gr.Textbox(label="Summary"), gr.Dataframe(label="Top suspicious rows")],
    title="Credit Card Fraud Detector",
    description="Upload transactions; model returns fraud probabilities and predictions."
)
demo.launch(share=True)
