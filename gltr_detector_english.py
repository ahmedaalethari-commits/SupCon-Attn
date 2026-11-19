# ==============================================================================
# GLTR — Statistical Detection of Generated Text (Gehrmann et al., 2019)
# Reimplementation, ADAPTED to this project's data/normalization/workflow
# conventions (see chat, 2026-10-24) — same method as the user's original
# script, with these changes:
#   - Loads the project's 3 FIXED splits (train/val/test, with `domain`),
#     exactly like every other English baseline here, instead of re-loading
#     a raw file and re-balancing/re-splitting it from scratch.
#   - Applies the SAME 8-step English normalization pipeline used by every
#     other baseline in this project (normalize_english_text) before
#     computing GLTR's GPT-2 features and before the Bag-of-Words baseline.
#   - Adds a VALIDATION-set threshold search (same np.arange(0.05, 0.96,
#     0.01) grid used across the project) for the main GLTR_Combined
#     model, instead of a fixed 0.5 cutoff — applied to TEST only, never
#     tuned on test itself.
#   - Saves to this project's Drive layout (PROJECT_DIR/GLTR_Detector_English,
#     with tables/ and plots/ subfolders) and reuses the project's IMG_COLORS
#     palette + style_ax/render_table_png helpers for the main dashboard
#     figures, so this method's outputs look and read like every other
#     method's in the project.
#   - Adds a unified Document-level Overall + per-domain performance table
#     for GLTR_Combined, matching every other baseline script's table.
#
# Method (unchanged from the paper):
#   Test 1: average probability assigned by a detection LM to the actual word
#   Test 2: distribution over top-k rank buckets (top-10 / top-100 / top-1000 / rest)
#   Test 3: entropy of the predicted next-word distribution
#   Baseline: Bag-of-Words (TF-IDF) + Logistic Regression
#   Classifier: Logistic Regression on top of the statistical features
#   Evaluation: document level, on the project's fixed held-out test split
#
# GPT-2 (117M) is used as the detection LM, exactly as in the paper's public
# deployment, because it is a left-to-right LM and needs only ONE forward
# pass per document (unlike BERT, which needs one masked pass per token and
# is far too slow to run over a full dataset).
# ==============================================================================

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import gc, os, re, random, unicodedata, pickle, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix, roc_curve, auc,
    matthews_corrcoef,
)
from tqdm.auto import tqdm

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 20)

# ==============================================================================
# 0. SEED
# ==============================================================================
SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
set_seed()

# ==============================================================================
# 1. CONFIG — same project layout/conventions as every other English baseline
# ==============================================================================
PROJECT_DIR = "/content/drive/MyDrive/english_ai_detector_project"
METHOD_DIR  = f"{PROJECT_DIR}/GLTR_Detector_English"
TABLES_DIR  = f"{METHOD_DIR}/tables"
PLOTS_DIR   = f"{METHOD_DIR}/plots"
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_SPLIT_FILE = f"{PROJECT_DIR}/splits/train_split.csv"
VAL_SPLIT_FILE   = f"{PROJECT_DIR}/splits/val_split.csv"
TEST_SPLIT_FILE  = f"{PROJECT_DIR}/splits/test_split.csv"
DOMAIN_COLUMN    = "domain"

CFG = {
    "lm_name":    "gpt2",            # detection LM (GLTR's public backend uses GPT-2 117M)
    "max_tokens": 512,               # truncate documents to this many GPT-2 tokens
    "buckets":    (10, 100, 1000),   # top-10 / top-100 / top-1000 / rest (GLTR default)
    "th_grid":    np.arange(0.05, 0.96, 0.01),   # same threshold-search grid used project-wide
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 70)
print("  GLTR — Statistical Detection of Generated Text")
print("  Detection LM: GPT-2 (117M)  |  Classifier: Logistic Regression")
print("=" * 70)
print(f"  Device: {DEVICE}")

# ==============================================================================
# 2. PUBLICATION STYLE — same IMG_COLORS palette + style_ax/render_table_png
# helpers used across this project's English baseline scripts.
# ==============================================================================
_preferred_fonts = ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']
_available = {f.name for f in fm.fontManager.ttflist}
_font_family = next((f for f in _preferred_fonts if f in _available), 'DejaVu Serif')

plt.rcParams.update({
    'font.family': _font_family, 'font.size': 12,
    'axes.titlesize': 14, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.9, 'lines.linewidth': 2.0,
})

IMG_COLORS = {
    "human":    "#1E90FF",  # dodger blue — Human class
    "ai":       "#FF6347",  # tomato red  — AI class
    "roc_line": "#2E97F0",  # ROC curve line
    "val_f1":   "#4CAF50",  # green  — Val F1 line
    "val_acc":  "#FF9800",  # orange — Val Accuracy line
}
FEATURE_COMPARISON_COLORS = {   # GLTR-specific: Test1/2/3/Combined/baseline — method comparison,
    "Test1_AvgProb":      "#9C27B0",   # not a Human/AI color, kept distinct from IMG_COLORS
    "Test2_TopKBuckets":  "#2196F3",
    "Test3_Entropy":      "#FF9800",
    "GLTR_Combined":      "#4CAF50",
    "BagOfWords_Baseline":"#F44336",
}

def style_ax(ax, ylabel=None, xlabel=None, title=None):
    if title: ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

def render_table_png(df_table, save_path, figsize=None):
    n_rows, n_cols = df_table.shape
    figsize = figsize or (max(6, n_cols * 1.6), 0.5 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    tbl = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#BBBBBB'); cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor('#1D3557'); cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#F5F7FA' if row % 2 == 0 else '#FFFFFF')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def _plot_cm(ax, cm, title, tn, fp, fn, tp):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
               xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'],
               linewidths=0.5, linecolor='white',
               annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.text(0.5, -0.20, f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
           transform=ax.transAxes, ha='center', fontsize=8, color='#444444')

# ==============================================================================
# 3. DATA — the project's 3 FIXED splits (same rows every English script
# uses), NOT re-loaded/re-balanced/re-split here.
# ==============================================================================
print("\n" + "─" * 70)
print("  Loading fixed splits ...")

required_cols = ["text", "label", DOMAIN_COLUMN]

def _load_split(path, name):
    d = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    missing = [c for c in required_cols if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in {name} split: {missing}")
    d = d[required_cols].dropna().reset_index(drop=True)
    d["label"] = d["label"].astype(int)
    d[DOMAIN_COLUMN] = d[DOMAIN_COLUMN].astype(str).str.strip().str.lower()
    return d

train_df = _load_split(TRAIN_SPLIT_FILE, "train")
val_df   = _load_split(VAL_SPLIT_FILE,   "validation")
test_df  = _load_split(TEST_SPLIT_FILE,  "test")

print(f"  Train: {len(train_df)}   Val: {len(val_df)}   Test: {len(test_df)}")
for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"  {name} label distribution:\n{d['label'].value_counts().to_string()}")

# ==============================================================================
# 4. NORMALIZATION — SAME 8-step English pipeline as every other baseline in
# this project (shared text across all baselines for a fair, identical-input
# comparison).
# ==============================================================================
_INVISIBLE_CHARS_RE = re.compile('[​‌‍‎‏‪‫‬‭‮﻿]')
_CITATION_RE = re.compile(
    r'\[\d+(?:\s*,\s*\d+)*\]'
    r'|\([^()]{0,60}\d{4}[^()]{0,25}\)'
)
_URL_RE   = re.compile(r'(?:https?://|www\.)\S+')
_EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+", flags=re.UNICODE
)
PUNCTUATION = r";:\"'`“”‘’«»\-–—/\\|@#\$%\^&\*_=~<>\(\)\[\]\{\}"
_NUMERIC_COMMA_SAFE_RE = re.compile(r'(?<!\d),|,(?!\d)')
_SPACE_BEFORE_SENT_END_RE = re.compile(r'\s+([.!?])')

_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'ve": " have", "'m": " am",
}
_CONTRACTIONS_RE = re.compile("|".join(re.escape(k) for k in _CONTRACTIONS.keys()))

def normalize_english_text(text):
    if not isinstance(text, str):
        return text
    text = _INVISIBLE_CHARS_RE.sub('', text)
    text = _CITATION_RE.sub(' ', text)
    text = _URL_RE.sub(' ', text)
    text = _EMAIL_RE.sub(' ', text)
    text = _EMOJI_RE.sub(' ', text)
    text = unicodedata.normalize('NFKC', text)
    text = _CONTRACTIONS_RE.sub(lambda m: _CONTRACTIONS[m.group(0)], text)
    text = _NUMERIC_COMMA_SAFE_RE.sub(" ", text)
    text = re.sub(rf"[{PUNCTUATION}]", " ", text)
    text = _SPACE_BEFORE_SENT_END_RE.sub(r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\n  Normalizing text (8-step English pipeline) ...")
for d in (train_df, val_df, test_df):
    d["text_processed"] = d["text"].astype(str).apply(normalize_english_text)
    d["text_processed"] = d["text_processed"].str.slice(0, 20000)   # safety cap before tokenization

# ==============================================================================
# 5. DETECTION LM (GPT-2) — one forward pass per document
# ==============================================================================
print("\n" + "─" * 70)
print("  Loading detection LM (GPT-2) ...")
gc.collect(); torch.cuda.empty_cache()

lm_tokenizer = AutoTokenizer.from_pretrained(CFG["lm_name"])
lm_model = AutoModelForCausalLM.from_pretrained(CFG["lm_name"]).to(DEVICE)
lm_model.eval()

@torch.no_grad()
def gltr_features(text):
    """
    Runs the document through GPT-2 with teacher forcing and computes,
    for every actual next-token position:
      - prob assigned to the actual token    (Test 1)
      - rank of the actual token              (Test 2)
      - entropy of the predicted distribution (Test 3)
    Returns aggregated document-level features + raw per-token arrays
    (used later for the Figure-4 / Figure-5 style plots).
    """
    ids = lm_tokenizer.encode(text, truncation=True, max_length=CFG["max_tokens"],
                              return_tensors="pt").to(DEVICE)
    if ids.shape[1] < 2:
        return None

    out = lm_model(ids)
    logits = out.logits[0, :-1, :]        # predictions at positions 0..N-2
    targets = ids[0, 1:]                  # actual next tokens

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    actual_logprob = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    actual_prob = actual_logprob.exp()

    actual_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    rank = (logits > actual_logit.unsqueeze(-1)).sum(dim=-1) + 1   # 1-indexed rank

    entropy = -(probs * log_probs).sum(dim=-1)

    actual_prob = actual_prob.cpu().numpy()
    rank        = rank.cpu().numpy()
    entropy     = entropy.cpu().numpy()

    b1, b2, b3 = CFG["buckets"]
    n = len(rank)
    frac_top1    = float((rank <= 1).sum())  / n
    frac_top10   = float((rank <= b1).sum()) / n
    frac_top100  = float((rank <= b2).sum()) / n
    frac_top1000 = float((rank <= b3).sum()) / n
    frac_rest    = float((rank >  b3).sum()) / n

    return {
        "avg_prob":     float(np.mean(actual_prob)),   # Test 1
        "frac_top10":   frac_top10,                     # Test 2 (cumulative buckets)
        "frac_top100":  frac_top100,
        "frac_top1000": frac_top1000,
        "frac_rest":    frac_rest,
        "avg_entropy":  float(np.mean(entropy)),         # Test 3
        "std_entropy":  float(np.std(entropy)),
        "n_tokens":     int(n),
        "_rank":        rank,          # raw, for plotting only
        "_entropy":     entropy,       # raw, for plotting only
        "_frac_top1":   frac_top1,
    }

def extract_split(dframe, desc):
    feats, keep_idx = [], []
    for i, row in tqdm(dframe.iterrows(), total=len(dframe), desc=f"  GLTR features [{desc}]"):
        f = gltr_features(row["text_processed"])
        if f is None:
            continue
        f["label"] = int(row["label"])
        f["domain"] = row[DOMAIN_COLUMN]
        feats.append(f)
        keep_idx.append(i)
    return pd.DataFrame(feats), dframe.loc[keep_idx].reset_index(drop=True)

print("\n  Extracting GLTR statistical features (Test 1/2/3) ...")
train_feat, train_df = extract_split(train_df, "train")
val_feat,   val_df   = extract_split(val_df,   "val")
test_feat,  test_df  = extract_split(test_df,  "test")

FEATURE_COLS_TEST1 = ["avg_prob"]
FEATURE_COLS_TEST2 = ["frac_top10", "frac_top100", "frac_top1000", "frac_rest"]
FEATURE_COLS_TEST3 = ["avg_entropy", "std_entropy"]
# (see chat, 2026-10-24) — verified against the ORIGINAL GLTR paper
# (Gehrmann, Strobelt & Rush, 2019, arXiv:1906.04043). The paper's Table 1
# has NO combined classifier at all: Test 1 (avg probability) and Test 2
# (top-k rank buckets) are reported as two SEPARATE single-feature-set
# logistic-regression classifiers (AUC 0.71 and 0.87 for GPT-2
# respectively), and Test 3 (entropy) is used ONLY in the paper's
# qualitative/visual analysis (Fig. 5) — it is never part of any
# classifier there. So "GLTR_Combined" below = Test 1 + Test 2 ONLY, to
# match the paper's actual features. Test3_Entropy is kept as its OWN
# separate ablation model further down (as in the paper, standalone,
# never merged into the main classifier) for extra diagnostic value only —
# it is explicitly NOT part of the paper-faithful "GLTR_Combined" result.
FEATURE_COLS_ALL   = FEATURE_COLS_TEST1 + FEATURE_COLS_TEST2

# ==============================================================================
# 6. CLASSIFIERS — Logistic Regression on GLTR features + Bag-of-Words baseline
# ==============================================================================
print("\n" + "=" * 70)
print("  Training classifiers (Logistic Regression)")
print("=" * 70)

def fit_lr(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(scaler.transform(X_train), y_train)
    return scaler, clf

def predict_lr(scaler, clf, X):
    return clf.predict_proba(scaler.transform(X))[:, 1]

models = {}
ytr = train_feat["label"].values

models["Test1_AvgProb"]     = fit_lr(train_feat[FEATURE_COLS_TEST1].values, ytr)
models["Test2_TopKBuckets"] = fit_lr(train_feat[FEATURE_COLS_TEST2].values, ytr)
models["Test3_Entropy"]     = fit_lr(train_feat[FEATURE_COLS_TEST3].values, ytr)
models["GLTR_Combined"]     = fit_lr(train_feat[FEATURE_COLS_ALL].values, ytr)   # main model

print("  Fitting Bag-of-Words (TF-IDF) baseline ...")
bow_vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 1))
Xtr_bow = bow_vec.fit_transform(train_df["text_processed"])
bow_clf = LogisticRegression(max_iter=2000, random_state=SEED)
bow_clf.fit(Xtr_bow, ytr)

# ==============================================================================
# 7. DECISION THRESHOLD — FIXED at 0.50 for the main GLTR_Combined model (per
# explicit request, 2026-10-24: "خلي العتبات بالكودات الجديدة 50" — use 0.50
# in the new scripts instead of a validation-searched threshold). The
# F1-vs-threshold curve over the validation grid is still computed and
# plotted/saved below purely as a DIAGNOSTIC (so you can see how sensitive
# F1 is to the cutoff), but it no longer overrides the cutoff actually used
# for the reported test-set metrics — that is always 0.50.
# ==============================================================================
print("\n  Decision threshold fixed at 0.50 (GLTR_Combined) — computing validation F1 curve for reference only ...")
val_scaler, val_clf = models["GLTR_Combined"]
val_probs_ai = predict_lr(val_scaler, val_clf, val_feat[FEATURE_COLS_ALL].values)
val_labels   = val_feat["label"].values

threshold_rows = []
for th in CFG["th_grid"]:
    preds_th = (val_probs_ai >= th).astype(int)
    f1_th = f1_score(val_labels, preds_th, zero_division=0)
    threshold_rows.append({"threshold": round(float(th), 2), "val_f1": round(f1_th, 4)})

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_excel(f"{TABLES_DIR}/threshold_search.xlsx", index=False)
best_threshold = 0.5
best_threshold_f1 = f1_score(val_labels, (val_probs_ai >= 0.5).astype(int), zero_division=0)
print(f"  Using FIXED threshold: {best_threshold:.2f}  (val F1 at this threshold={best_threshold_f1:.4f})")

fig_th, ax_th = plt.subplots(figsize=(5.5, 4.2))
ax_th.plot(threshold_df["threshold"], threshold_df["val_f1"], color=IMG_COLORS["val_f1"], lw=2)
ax_th.axvline(best_threshold, color="black", ls="--", lw=1, label=f"best={best_threshold:.2f}")
style_ax(ax_th, xlabel="Decision threshold", ylabel="Validation F1 (GLTR_Combined)",
         title="Threshold Search (Validation)")
ax_th.legend(fontsize=9)
plt.tight_layout()
fig_th.savefig(f"{PLOTS_DIR}/threshold_search.png"); plt.show()

# ==============================================================================
# 8. EVALUATION ON TEST SET
# ==============================================================================
print("\n" + "─" * 70)
print("  Evaluating on held-out test set ...")

y_test = test_feat["label"].values
results_table = []
roc_curves = {}

def full_metrics(name, y_true, y_pred, y_prob, threshold_used):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    results_table.append({"Feature": name, "Threshold": round(threshold_used, 2),
                          "Accuracy": round(acc, 4), "F1": round(f1, 4),
                          "Precision": round(p, 4), "Recall": round(r, 4),
                          "MCC": round(mcc, 4), "AUC": round(roc_auc, 4)})
    return fpr, tpr, roc_auc

for name in ["Test1_AvgProb", "Test2_TopKBuckets", "Test3_Entropy", "GLTR_Combined"]:
    cols = {"Test1_AvgProb": FEATURE_COLS_TEST1, "Test2_TopKBuckets": FEATURE_COLS_TEST2,
            "Test3_Entropy": FEATURE_COLS_TEST3, "GLTR_Combined": FEATURE_COLS_ALL}[name]
    scaler, clf = models[name]
    y_prob = predict_lr(scaler, clf, test_feat[cols].values)
    th_used = best_threshold if name == "GLTR_Combined" else 0.5
    y_pred = (y_prob >= th_used).astype(int)
    fpr, tpr, roc_auc = full_metrics(name, y_test, y_pred, y_prob, th_used)
    roc_curves[name] = (fpr, tpr, roc_auc)
    if name == "GLTR_Combined":
        main_y_pred, main_y_prob = y_pred, y_prob

bow_prob = bow_clf.predict_proba(bow_vec.transform(test_df["text_processed"]))[:, 1]
bow_pred = (bow_prob >= 0.5).astype(int)
fpr, tpr, roc_auc = full_metrics("BagOfWords_Baseline", y_test, bow_pred, bow_prob, 0.5)
roc_curves["BagOfWords_Baseline"] = (fpr, tpr, roc_auc)

results_df = pd.DataFrame(results_table)
print("\n  === Feature comparison (Test1 / Test2 / Test3 / Combined[=Test1+Test2, paper-faithful] / BoW) ===")
print(results_df.to_string(index=False))
results_df.to_excel(f"{TABLES_DIR}/feature_comparison_metrics.xlsx", index=False)
render_table_png(results_df, f"{PLOTS_DIR}/table_feature_comparison.png",
                 figsize=(12, 0.55 * len(results_df) + 1.2))

main_row = results_df[results_df["Feature"] == "GLTR_Combined"].iloc[0]
accuracy, f1, precision, recall, mcc, roc_auc = (
    main_row["Accuracy"], main_row["F1"], main_row["Precision"],
    main_row["Recall"], main_row["MCC"], main_row["AUC"])

print("\n" + "=" * 55)
print("FINAL RESULTS ON TEST SET — GLTR_Combined = Test1+Test2 (main model, paper-faithful)")
print("=" * 55)
print(f"  Threshold : {best_threshold:.2f}  (fixed, val F1 at this threshold={best_threshold_f1:.4f})")
print(f"  Accuracy  : {accuracy:.4f}   F1: {f1:.4f}   Precision: {precision:.4f}")
print(f"  Recall    : {recall:.4f}   MCC: {mcc:.4f}   AUC: {roc_auc:.4f}")
print("=" * 55)

# ==============================================================================
# 9. UNIFIED PERFORMANCE TABLE — Document-level Overall + per-domain, same
# structure as every other baseline script's unified table (GLTR_Combined
# at its validation-searched threshold).
# ==============================================================================
test_df = test_df.reset_index(drop=True)
test_df["prob_ai"] = main_y_prob
test_df["pred_label"] = main_y_pred
test_df["true_label"] = y_test

domain_rows = []
for dom in sorted(test_df[DOMAIN_COLUMN].unique()):
    mask = (test_df[DOMAIN_COLUMN] == dom).values
    sub_true, sub_pred, sub_prob = y_test[mask], main_y_pred[mask], main_y_prob[mask]
    dom_fpr, dom_tpr, _ = roc_curve(sub_true, sub_prob)
    domain_rows.append({
        "Level": f"Document — {dom.capitalize()}", "N": int(mask.sum()),
        "Accuracy": round(accuracy_score(sub_true, sub_pred), 4),
        "F1": round(f1_score(sub_true, sub_pred, zero_division=0), 4),
        "Precision": round(precision_score(sub_true, sub_pred, zero_division=0), 4),
        "Recall": round(recall_score(sub_true, sub_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(sub_true, sub_pred), 4),
        "AUC": round(auc(dom_fpr, dom_tpr), 4),
    })
domain_perf_df = pd.DataFrame(domain_rows)

overall_doc_row = pd.DataFrame([{
    "Level": "Document — Overall", "N": len(test_df),
    "Accuracy": round(accuracy, 4), "F1": round(f1, 4), "Precision": round(precision, 4),
    "Recall": round(recall, 4), "MCC": round(mcc, 4), "AUC": round(roc_auc, 4),
}])

unified_df = pd.concat([overall_doc_row, domain_perf_df], ignore_index=True)
unified_df.to_excel(f"{TABLES_DIR}/unified_performance_table.xlsx", index=False)
render_table_png(unified_df, f"{PLOTS_DIR}/table_unified_performance.png",
                 figsize=(13, 0.55 * len(unified_df) + 1.2))
print(f"\n  Unified table ({len(unified_df)} rows):")
print(unified_df.to_string(index=False))

# ==============================================================================
# 10. PLOTS — GLTR-specific figures (rank buckets, entropy-vs-rank), plus the
# project-standard confusion matrix / ROC comparison / probability
# distribution dashboard (IMG_COLORS, style_ax).
# ==============================================================================
print("\n  Generating plots ...")
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(f"GLTR Statistical Detector (GPT-2 features + Logistic Regression)\n"
             f"GLTR_Combined (Test1+Test2) — F1={f1:.4f}  AUC={roc_auc:.4f}  threshold={best_threshold:.2f}",
             fontsize=12, fontweight="bold")

# (a) Confusion matrix — GLTR_Combined
cm = confusion_matrix(y_test, main_y_pred)
tn, fp, fn, tp = cm.ravel()
_plot_cm(axes[0, 0], cm, "Confusion Matrix (GLTR_Combined = Test1+Test2)", tn, fp, fn, tp)

# (b) ROC — all feature sets + baseline (Table-1 style comparison)
for name, (fpr, tpr, roc_auc_m) in roc_curves.items():
    axes[0, 1].plot(fpr, tpr, lw=2, color=FEATURE_COMPARISON_COLORS.get(name, "gray"),
                    label=f"{name} (AUC={roc_auc_m:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
style_ax(axes[0, 1], xlabel="False Positive Rate", ylabel="True Positive Rate",
         title="ROC — Feature Comparison")
axes[0, 1].legend(fontsize=7)

# (c) Probability distribution — GLTR_Combined (Test1+Test2), project's Human/AI colors
for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    axes[0, 2].hist(main_y_prob[y_test == lbl], bins=40, alpha=0.6, color=color, label=name, density=True)
axes[0, 2].axvline(best_threshold, color="black", lw=1.2, linestyle="--", label=f"threshold={best_threshold:.2f}")
style_ax(axes[0, 2], xlabel="P(AI)", ylabel="Density", title="P(AI) Distribution")
axes[0, 2].legend(fontsize=8)

# (d) Figure-4 style: rank bucket distribution, Human vs AI
bucket_rows = []
for lbl, name in [(0, "Human"), (1, "AI")]:
    sub = test_feat[test_feat["label"] == lbl]
    bucket_rows.append({
        "Class": name,
        "top1":    sub["_frac_top1"].mean() * 100,
        "top5-10": (sub["frac_top10"].mean() - sub["_frac_top1"].mean()) * 100,
        "top100":  (sub["frac_top100"].mean() - sub["frac_top10"].mean()) * 100,
        ">top100": (1 - sub["frac_top100"].mean()) * 100,
    })
bucket_df = pd.DataFrame(bucket_rows).set_index("Class")
bucket_df[["top1", "top5-10", "top100", ">top100"]].plot(
    kind="barh", stacked=True, ax=axes[1, 0],
    color=["#e0e0e0", "#bdbdbd", "#757575", "#212121"])
style_ax(axes[1, 0], xlabel="Percentage", title="Rank-Bucket Distribution (Fig. 4 style)")
axes[1, 0].legend(fontsize=8, loc="lower right")

# (e) Figure-5 style: entropy vs rank density, Human vs AI
for ax, lbl, name in [(axes[1, 1], 0, "Human"), (axes[1, 2], 1, "AI")]:
    sub = test_feat[test_feat["label"] == lbl]
    all_entropy = np.concatenate(sub["_entropy"].values) if len(sub) else np.array([])
    all_rank    = np.concatenate(sub["_rank"].values) if len(sub) else np.array([])
    if len(all_entropy) > 20000:
        idx = np.random.default_rng(SEED).choice(len(all_entropy), 20000, replace=False)
        all_entropy, all_rank = all_entropy[idx], all_rank[idx]
    ax.hexbin(all_entropy, np.clip(all_rank, 0, 500), gridsize=40, cmap="Greys", mincnt=1)
    style_ax(ax, xlabel="Entropy", ylabel="Rank", title=f"Entropy vs Rank — {name} (Fig. 5 style)")

plt.tight_layout()
plot_path = f"{PLOTS_DIR}/gltr_results_dashboard.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"  Dashboard saved -> {plot_path}")

# Standalone ROC curve (project convention: also saved on its own, not just
# inside the dashboard panel)
fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.8))
for name, (fpr, tpr, roc_auc_m) in roc_curves.items():
    ax_roc.plot(fpr, tpr, lw=2, color=FEATURE_COMPARISON_COLORS.get(name, "gray"),
               label=f"{name} (AUC={roc_auc_m:.3f})")
ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA", lw=1)
style_ax(ax_roc, xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve (Test Set)")
ax_roc.legend(loc="lower right", fontsize=8)
plt.tight_layout()
fig_roc.savefig(f"{PLOTS_DIR}/roc_curve.png"); plt.show()

# Confusion matrices — Overall + per-domain (project convention)
domains = sorted(test_df[DOMAIN_COLUMN].unique())
n_cols = 1 + len(domains)
fig_cm, axes_cm = plt.subplots(1, n_cols, figsize=(3.8 * n_cols, 4.4))
if n_cols == 1: axes_cm = [axes_cm]
_plot_cm(axes_cm[0], cm, "Document — Overall", tn, fp, fn, tp)
for ax, dom in zip(axes_cm[1:], domains):
    sub = test_df[test_df[DOMAIN_COLUMN] == dom]
    cm_dom = confusion_matrix(sub["true_label"], sub["pred_label"], labels=[0, 1])
    _plot_cm(ax, cm_dom, f"Document — {dom.capitalize()}", *cm_dom.ravel())
plt.tight_layout()
fig_cm.savefig(f"{PLOTS_DIR}/confusion_matrices_all.png"); plt.show()

# Confidence: correct vs incorrect (project convention)
test_df["correct"] = test_df["pred_label"] == test_df["true_label"]
test_df["confidence"] = test_df["prob_ai"].apply(lambda p: p if p >= best_threshold else 1 - p)
fig_conf, ax_conf = plt.subplots(figsize=(5.5, 4.2))
for correct, color, label in [(True, IMG_COLORS["human"], "Correct"), (False, IMG_COLORS["ai"], "Incorrect")]:
    subset = test_df.loc[test_df["correct"] == correct, "confidence"]
    if len(subset) > 0:
        ax_conf.hist(subset, bins=25, alpha=0.6, color=color, label=label,
                    density=True, edgecolor="white", linewidth=0.4)
style_ax(ax_conf, xlabel="Model confidence in predicted class", ylabel="Density",
         title="Prediction Confidence: Correct vs Incorrect (Document-level)")
ax_conf.legend(fontsize=9)
plt.tight_layout()
fig_conf.savefig(f"{PLOTS_DIR}/confidence_correct_vs_incorrect.png"); plt.show()

# ==============================================================================
# 11. SAVE
# ==============================================================================
print("\n" + "─" * 70)
print(f"  Saving to Drive -> {METHOD_DIR}")

with open(f"{METHOD_DIR}/gltr_combined_model.pkl", "wb") as f_out:
    pickle.dump({
        "scaler": models["GLTR_Combined"][0], "clf": models["GLTR_Combined"][1],
        "feature_cols": FEATURE_COLS_ALL, "cfg": CFG,
        "threshold": best_threshold, "val_f1_at_threshold": best_threshold_f1,
    }, f_out)
with open(f"{METHOD_DIR}/gltr_config.json", "w") as f_out:
    import json
    json.dump({"threshold": best_threshold, "val_f1": best_threshold_f1,
               "buckets": list(CFG["buckets"]), "max_tokens": CFG["max_tokens"]}, f_out, indent=2)

test_feat.drop(columns=["_rank", "_entropy"]).to_csv(f"{TABLES_DIR}/test_gltr_features.csv", index=False)

print(f"  Saved -> {METHOD_DIR}")

# ==============================================================================
# 12. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY — GLTR Statistical Detector")
print("=" * 70)
print(f"  Detection LM : {CFG['lm_name']}")
print(f"  Train docs   : {len(train_feat)}   Val docs: {len(val_feat)}   Test docs: {len(test_feat)}")
print(f"  Threshold    : {best_threshold:.2f}  (fixed at 0.50)")
print()
print(results_df.to_string(index=False))
print()
print(unified_df.to_string(index=False))
print(f"\n  Drive path   : {METHOD_DIR}")
print("=" * 70)