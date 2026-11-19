# ==============================================================================
# Fast-DetectGPT — Zero-Shot Detection via Conditional Probability Curvature
# (Bao et al., ICLR 2024, arXiv:2310.05130)
# Reimplementation, ADAPTED to this project's data/normalization/workflow
# conventions (see chat, 2026-10-24) — same method/formula as the user's
# original script (VERIFIED against the paper's Appendix B, Eq. 5 — the
# analytical curvature formula below is an EXACT match), with these changes:
#   - Loads the project's 3 FIXED splits (train/val/test, with `domain`),
#     exactly like every other English baseline here, instead of re-loading
#     a raw file and re-balancing/re-splitting it from scratch.
#   - Applies the SAME 8-step English normalization pipeline used by every
#     other baseline in this project (normalize_english_text) before
#     computing curvature/likelihood/logrank/entropy.
#   - Scoring/sampling LM upgraded from gpt2-medium (355M) to gpt2-xl
#     (1.5B) — the ACTUAL GPT-2 size the paper uses in its white-box
#     (same-model) setting (Tables 4/5), loaded in fp16 on GPU to keep
#     memory manageable; falls back to fp32 on CPU.
#   - Adds a VALIDATION-set threshold search (same np.arange(0.05, 0.96,
#     0.01) grid used across the project) for the main Fast-DetectGPT
#     (curvature) model's calibrated LR, instead of a fixed 0.5 cutoff —
#     applied to TEST only, never tuned on test itself.
#   - Saves to this project's Drive layout
#     (PROJECT_DIR/FastDetectGPT_Detector_English, with tables/ and plots/
#     subfolders) and reuses the project's IMG_COLORS palette +
#     style_ax/render_table_png helpers, so this method's outputs look and
#     read like every other method's in the project.
#   - Adds a unified Document-level Overall + per-domain performance table,
#     matching every other baseline script's table.
#
# Method (unchanged from the paper, white-box / same-model setting,
# analytical closed-form solution — Appendix B, no sampling needed, ONE
# forward pass per document):
#
#   curvature(x) = ( log p(x) - mu~ ) / sigma~
#
#   per token j:  mu~_j = E_v[ log p(v | x<j) ] = -entropy_j
#                 sigma~_j^2 = E_v[ (log p(v | x<j))^2 ] - mu~_j^2
#   summed over tokens (independence assumption): mu~ = sum_j mu~_j ,
#   sigma~ = sqrt( sum_j sigma~_j^2 )
#
# A positive curvature => machine-generated; close to 0 => human-written.
# Genuinely zero-shot (AUROC needs no training), but since we have labels,
# we also fit a 1-feature Logistic Regression on top of it for a
# calibrated Accuracy/F1/Precision/Recall/MCC, and compare against the
# Likelihood and LogRank/Entropy zero-shot baselines, exactly as in the
# paper's Table 2 (a faithful SUBSET of its baselines — LRR/DNA-GPT/NPR/
# DetectGPT need perturbations or extra models and are out of scope here).
# ==============================================================================

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import gc, os, re, random, unicodedata, pickle, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
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
METHOD_DIR  = f"{PROJECT_DIR}/FastDetectGPT_Detector_English"
TABLES_DIR  = f"{METHOD_DIR}/tables"
PLOTS_DIR   = f"{METHOD_DIR}/plots"
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_SPLIT_FILE = f"{PROJECT_DIR}/splits/train_split.csv"
VAL_SPLIT_FILE   = f"{PROJECT_DIR}/splits/val_split.csv"
TEST_SPLIT_FILE  = f"{PROJECT_DIR}/splits/test_split.csv"
DOMAIN_COLUMN    = "domain"

CFG = {
    # (see chat, 2026-10-24) — gpt2-xl (1.5B), matching the paper's
    # white-box setting where GPT-2 is used as sampling=scoring model
    # (Tables 4/5). Set back to "gpt2-medium" below if this OOMs on your
    # GPU — the curvature formula itself is unaffected by model size,
    # only its sensitivity/AUROC will differ from the paper's reported one.
    "lm_name":    "gpt2-xl",
    "max_tokens": 512,
    "th_grid":    np.arange(0.05, 0.96, 0.01),   # same threshold-search grid used project-wide
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 70)
print("  Fast-DetectGPT — Conditional Probability Curvature (Zero-Shot)")
print(f"  Scoring/Sampling LM: {CFG['lm_name']} (white-box, same-model setting)")
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
METHOD_COMPARISON_COLORS = {   # method comparison (not Human/AI), kept distinct from IMG_COLORS
    "Likelihood":      "#9C27B0",
    "Entropy":         "#FF9800",
    "LogRank":         "#2196F3",
    "Fast-DetectGPT":  "#4CAF50",
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
# 5. SCORING LM — one forward pass per document (analytical curvature, Appx. B)
# ==============================================================================
print("\n" + "─" * 70)
print("  Loading scoring LM ...")
gc.collect(); torch.cuda.empty_cache()

LM_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
lm_tokenizer = AutoTokenizer.from_pretrained(CFG["lm_name"])
lm_model = AutoModelForCausalLM.from_pretrained(CFG["lm_name"], torch_dtype=LM_DTYPE).to(DEVICE)
lm_model.eval()

@torch.no_grad()
def fast_detect_features(text):
    """
    Single forward pass -> per-token log p(actual), analytical mu~_j, sigma~_j^2
    (Eq. 6-7, Appendix B), then aggregate into the conditional probability
    curvature (Eq. 5), plus the classic Likelihood, LogRank and Entropy
    zero-shot baselines (Table 2 of the paper).
    """
    ids = lm_tokenizer.encode(text, truncation=True, max_length=CFG["max_tokens"],
                              return_tensors="pt").to(DEVICE)
    if ids.shape[1] < 2:
        return None

    out = lm_model(ids)
    logits = out.logits[0, :-1, :].float()   # upcast to fp32 for the log/softmax math below
    targets = ids[0, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # log p(actual token)  -> sequence log-likelihood terms
    actual_logprob = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # rank of actual token (for LogRank baseline)
    actual_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    rank = (logits > actual_logit.unsqueeze(-1)).sum(dim=-1) + 1

    # analytical mu~_j = E_v[log p(v)] = -entropy_j ;  E_v[(log p(v))^2]
    mu_j     = (probs * log_probs).sum(dim=-1)                # = -entropy_j
    m2_j     = (probs * log_probs.pow(2)).sum(dim=-1)          # E[(log p)^2]
    var_j    = (m2_j - mu_j.pow(2)).clamp(min=1e-8)

    total_logp = actual_logprob.sum().item()
    total_mu   = mu_j.sum().item()
    total_var  = var_j.sum().item()

    curvature = (total_logp - total_mu) / (total_var ** 0.5 + 1e-8)

    n = len(rank)
    return {
        "curvature":  float(curvature),                          # Fast-DetectGPT main score
        "likelihood": float(total_logp / n),                     # baseline: mean log-prob
        "logrank":    float(-np.log(rank.cpu().numpy()).mean()), # baseline: -mean log(rank)
        "entropy":    float(-total_mu / n),                      # baseline: mean entropy
        "n_tokens":   int(n),
    }

def extract_split(dframe, desc):
    feats, keep_idx = [], []
    for i, row in tqdm(dframe.iterrows(), total=len(dframe), desc=f"  Fast-DetectGPT features [{desc}]"):
        f = fast_detect_features(row["text_processed"])
        if f is None:
            continue
        f["label"] = int(row["label"])
        f["domain"] = row[DOMAIN_COLUMN]
        feats.append(f)
        keep_idx.append(i)
    return pd.DataFrame(feats), dframe.loc[keep_idx].reset_index(drop=True)

print("\n  Extracting conditional probability curvature (+ baselines) ...")
train_feat, train_df = extract_split(train_df, "train")
val_feat,   val_df   = extract_split(val_df,   "val")
test_feat,  test_df  = extract_split(test_df,  "test")

FEATURES = ["likelihood", "entropy", "logrank", "curvature"]
DISPLAY_NAME = {"likelihood": "Likelihood", "entropy": "Entropy", "logrank": "LogRank",
                "curvature": "Fast-DetectGPT"}

# ==============================================================================
# 6. CLASSIFIERS — 1-feature Logistic Regression per method (for a calibrated
# threshold + full metrics); raw zero-shot AUROC is also reported directly
# from the scores (no training needed), exactly as in the paper.
# ==============================================================================
print("\n" + "=" * 70)
print("  Calibrating decision threshold (Logistic Regression, 1 feature each)")
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
for feat in FEATURES:
    models[feat] = fit_lr(train_feat[[feat]].values, ytr)

# ==============================================================================
# 7. DECISION THRESHOLD — FIXED at 0.50 for the main Fast-DetectGPT
# (curvature) model (per explicit request, 2026-10-24: "خلي العتبات
# بالكودات الجديدة 50" — use 0.50 in the new scripts instead of a
# validation-searched threshold). The F1-vs-threshold curve over the
# validation grid is still computed and plotted/saved below purely as a
# DIAGNOSTIC (so you can see how sensitive F1 is to the cutoff), but it no
# longer overrides the cutoff actually used for the reported test-set
# metrics — that is always 0.50.
# ==============================================================================
print("\n  Decision threshold fixed at 0.50 (Fast-DetectGPT) — computing validation F1 curve for reference only ...")
val_scaler, val_clf = models["curvature"]
val_probs_ai = predict_lr(val_scaler, val_clf, val_feat[["curvature"]].values)
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
style_ax(ax_th, xlabel="Decision threshold", ylabel="Validation F1 (Fast-DetectGPT)",
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
    results_table.append({"Method": name, "Threshold": round(threshold_used, 2),
                          "AUROC": round(roc_auc, 4), "Accuracy": round(acc, 4),
                          "F1": round(f1, 4), "Precision": round(p, 4), "Recall": round(r, 4),
                          "MCC": round(mcc, 4)})
    roc_curves[name] = (fpr, tpr, roc_auc)

for feat in FEATURES:
    scaler, clf = models[feat]
    y_prob = predict_lr(scaler, clf, test_feat[[feat]].values)
    th_used = best_threshold if feat == "curvature" else 0.5
    y_pred = (y_prob >= th_used).astype(int)
    name = DISPLAY_NAME[feat]
    full_metrics(name, y_test, y_pred, y_prob, th_used)
    if feat == "curvature":
        main_y_pred, main_y_prob = y_pred, y_prob

# Pure zero-shot AUROC (no LR calibration at all) — matches the paper's protocol exactly
fpr_zs, tpr_zs, _ = roc_curve(y_test, test_feat["curvature"].values)
zeroshot_auroc = auc(fpr_zs, tpr_zs)

results_df = pd.DataFrame(results_table)
print("\n  === Method comparison (Table-2 style) ===")
print(results_df.to_string(index=False))
print(f"\n  Pure zero-shot AUROC (raw curvature, no LR calibration): {zeroshot_auroc:.4f}")
results_df.to_excel(f"{TABLES_DIR}/method_comparison_metrics.xlsx", index=False)
render_table_png(results_df, f"{PLOTS_DIR}/table_method_comparison.png",
                 figsize=(12, 0.55 * len(results_df) + 1.2))

main_row = results_df[results_df["Method"] == "Fast-DetectGPT"].iloc[0]
accuracy, f1, precision, recall, mcc, roc_auc = (
    main_row["Accuracy"], main_row["F1"], main_row["Precision"],
    main_row["Recall"], main_row["MCC"], main_row["AUROC"])

print("\n" + "=" * 55)
print("FINAL RESULTS ON TEST SET — Fast-DetectGPT (curvature, main model)")
print("=" * 55)
print(f"  Threshold : {best_threshold:.2f}  (fixed, val F1 at this threshold={best_threshold_f1:.4f})")
print(f"  Accuracy  : {accuracy:.4f}   F1: {f1:.4f}   Precision: {precision:.4f}")
print(f"  Recall    : {recall:.4f}   MCC: {mcc:.4f}   AUROC: {roc_auc:.4f}")
print(f"  Pure zero-shot AUROC (no calibration): {zeroshot_auroc:.4f}")
print("=" * 55)

# ==============================================================================
# 9. UNIFIED PERFORMANCE TABLE — Document-level Overall + per-domain, same
# structure as every other baseline script's unified table (Fast-DetectGPT
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
# 10. PLOTS — Fast-DetectGPT-specific figures (raw curvature histogram,
# curvature vs doc length, AUROC bar comparison), plus the project-standard
# confusion matrix / ROC comparison / probability distribution dashboard
# (IMG_COLORS, style_ax).
# ==============================================================================
print("\n  Generating plots ...")
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(f"Fast-DetectGPT — Conditional Probability Curvature ({CFG['lm_name']})\n"
             f"F1={f1:.4f}  AUROC={roc_auc:.4f}  threshold={best_threshold:.2f}  "
             f"(zero-shot AUROC={zeroshot_auroc:.4f})",
             fontsize=12, fontweight="bold")

# (a) Confusion matrix
cm = confusion_matrix(y_test, main_y_pred)
tn, fp, fn, tp = cm.ravel()
_plot_cm(axes[0, 0], cm, "Confusion Matrix (Fast-DetectGPT)", tn, fp, fn, tp)

# (b) ROC — all methods
for name, (fpr, tpr, roc_auc_m) in roc_curves.items():
    axes[0, 1].plot(fpr, tpr, lw=2, color=METHOD_COMPARISON_COLORS.get(name, "gray"),
                    label=f"{name} (AUC={roc_auc_m:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
style_ax(axes[0, 1], xlabel="False Positive Rate", ylabel="True Positive Rate",
         title="ROC — Method Comparison")
axes[0, 1].legend(fontsize=8)

# (c) Probability distribution — Fast-DetectGPT, project's Human/AI colors
for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    axes[0, 2].hist(main_y_prob[y_test == lbl], bins=40, alpha=0.6, color=color, label=name, density=True)
axes[0, 2].axvline(best_threshold, color="black", lw=1.2, linestyle="--", label=f"threshold={best_threshold:.2f}")
style_ax(axes[0, 2], xlabel="P(AI)", ylabel="Density", title="P(AI) Distribution")
axes[0, 2].legend(fontsize=8)

# (d) Figure-1 style: raw curvature histogram, Human vs AI
for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    axes[1, 0].hist(test_feat.loc[test_feat["label"] == lbl, "curvature"], bins=40, alpha=0.6,
                    color=color, label=name, density=True)
style_ax(axes[1, 0], xlabel="Curvature", title="Conditional Probability Curvature (Fig. 1 style)")
axes[1, 0].legend(fontsize=9)

# (e) Curvature vs document length
for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    sub = test_feat[test_feat["label"] == lbl]
    axes[1, 1].scatter(sub["n_tokens"], sub["curvature"], s=10, alpha=0.4, color=color, label=name)
style_ax(axes[1, 1], xlabel="Tokens", ylabel="Curvature", title="Curvature vs. Document Length")
axes[1, 1].legend(fontsize=9)

# (f) AUROC bar comparison (Table-2 style)
bar_series = results_df.set_index("Method")["AUROC"]
axes[1, 2].bar(bar_series.index, bar_series.values,
               color=[METHOD_COMPARISON_COLORS.get(m, "gray") for m in bar_series.index])
style_ax(axes[1, 2], ylabel="AUROC", title="AUROC by Method")
axes[1, 2].set_ylim(0, 1.05)
axes[1, 2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plot_path = f"{PLOTS_DIR}/fast_detectgpt_results_dashboard.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"  Dashboard saved -> {plot_path}")

# Standalone ROC curve (project convention: also saved on its own)
fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.8))
for name, (fpr, tpr, roc_auc_m) in roc_curves.items():
    ax_roc.plot(fpr, tpr, lw=2, color=METHOD_COMPARISON_COLORS.get(name, "gray"),
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

with open(f"{METHOD_DIR}/fast_detectgpt_model.pkl", "wb") as f_out:
    pickle.dump({
        "scaler": models["curvature"][0], "clf": models["curvature"][1],
        "cfg": CFG, "threshold": best_threshold, "val_f1_at_threshold": best_threshold_f1,
    }, f_out)
with open(f"{METHOD_DIR}/fast_detectgpt_config.json", "w") as f_out:
    json.dump({"threshold": best_threshold, "val_f1": best_threshold_f1,
               "lm_name": CFG["lm_name"], "max_tokens": CFG["max_tokens"]}, f_out, indent=2)

test_feat.to_csv(f"{TABLES_DIR}/test_fastdetect_features.csv", index=False)
print(f"  Saved -> {METHOD_DIR}")

# ==============================================================================
# 12. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY — Fast-DetectGPT (Zero-Shot)")
print("=" * 70)
print(f"  Scoring LM   : {CFG['lm_name']}")
print(f"  Train docs   : {len(train_feat)}   Val docs: {len(val_feat)}   Test docs: {len(test_feat)}")
print(f"  Threshold    : {best_threshold:.2f}  (fixed at 0.50)")
print()
print(results_df.to_string(index=False))
print()
print(unified_df.to_string(index=False))
print(f"\n  Pure zero-shot AUROC : {zeroshot_auroc:.4f}")
print(f"  Drive path   : {METHOD_DIR}")
print("=" * 70)