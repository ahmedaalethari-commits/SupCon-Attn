# ==============================================================================
# Detection of AI-Generated Texts: A Bi-LSTM and Attention-Based Approach
# (Blake, Miah, Kredens, Shin — IEEE Access 2025)
# Reimplementation, ADAPTED to this project's data/normalization/workflow
# conventions (see chat, 2026-10-24).
#
# PAPER-FIDELITY CHECK — findings:
#   The architecture's core, checkable hyperparameters MATCH the paper
#   exactly: embedding dim 32, Conv1D(128 filters, kernel=3, ReLU) +
#   MaxPooling "Initial Module", Bi-LSTM(64 units, recurrent_activation=
#   tanh, glorot_uniform init), dropout 0.3, one ReLU Dense(64) +
#   Dropout(0.3) classification head, single sigmoid output, Nadam
#   optimizer with gradient clipvalue=1.0, POS mapping NN:1/VB:2/JJ:3/
#   RB:4 (0=pad). The paper itself, however, is internally inconsistent /
#   underspecified on several points, which the original script already
#   documented and reasonably resolved: it offers 3 untitled attention
#   score variants (dot/general/concat) without saying which was used, it
#   gives two conflicting learning rates (0.0001 vs 0.000005 — the script
#   keeps 0.0001, from the hyperparameter section), it describes the model
#   input ambiguously as either POS tags or raw word-token indices (kept
#   as POS tags, consistent with the paper's Section IV.A framing and
#   Fig. 5), max_len=5000/epochs=1000 are impractical for this dataset's
#   actual (much shorter) documents (kept as a data-driven max_len +
#   EarlyStopping, as in the original script), and the reported train/test
#   split counts don't even add up internally in the paper's own tables.
#   None of this needed a fresh decision from you — the prior script's
#   documented choices are kept as-is.
#
#   ONE real code-level inconsistency WAS found and fixed: the script's own
#   header describes the attention mechanism as "two dense layers + tanh +
#   softmax", but the implementation actually used THREE Dense layers
#   (W1, W2, V) with W1(h)+W2(h) applied to the SAME input tensor — which
#   is mathematically identical to a single combined Dense layer (W1+W2
#   collapse into one matrix), so the second matrix added no expressive
#   power, just redundant parameters. Simplified to a single projection +
#   V, matching the header's own stated "two dense layers" design (see the
#   AdditiveAttention class below) — this does NOT reduce model capacity,
#   since the dropped matrix was redundant to begin with.
#
# Architecture (Section IV, Fig. 5-6):
#   POS-tag sequence -> Embedding
#     -> Initial Module: Conv1D + MaxPooling      (local contextual features)
#     -> Bi-LSTM(64)  -> Attention module          (long-range dependencies)
#   -> Concatenate(Initial, Attention)  [residual connection, per Eq. 6]
#   -> Dense(ReLU) + Dropout -> Dense(sigmoid)     (classification)
# ==============================================================================

import gc, os, re, random, unicodedata, pickle, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import nltk
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk import pos_tag, word_tokenize

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix, roc_curve, auc,
    matthews_corrcoef,
)
from tqdm.auto import tqdm

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 20)

# ==============================================================================
# 0. SEED
# ==============================================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ==============================================================================
# 1. CONFIG — same project layout/conventions as every other English baseline
# ==============================================================================
PROJECT_DIR = "/content/drive/MyDrive/english_ai_detector_project"
METHOD_DIR  = f"{PROJECT_DIR}/BiLSTM_Attention_Detector_English"
TABLES_DIR  = f"{METHOD_DIR}/tables"
PLOTS_DIR   = f"{METHOD_DIR}/plots"
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_SPLIT_FILE = f"{PROJECT_DIR}/splits/train_split.csv"
VAL_SPLIT_FILE   = f"{PROJECT_DIR}/splits/val_split.csv"
TEST_SPLIT_FILE  = f"{PROJECT_DIR}/splits/test_split.csv"
DOMAIN_COLUMN    = "domain"

CFG = {
    "max_len":          400,     # paper uses 5000; adapted (see header note) — refined below from data
    "embedding_dim":    32,
    "conv_filters":     128,
    "conv_kernel":      3,
    "lstm_units":       64,
    "attention_units":  64,
    "dense_units":      64,
    "dropout":          0.3,
    "learning_rate":    1e-4,
    "batch_size":        32,
    "max_epochs":        30,     # paper uses 1000; adapted via EarlyStopping
    "patience":           8,
    "th_grid":           np.arange(0.05, 0.96, 0.01),   # kept only for the reference diagnostic curve
}

print("=" * 70)
print("  Bi-LSTM + Attention (POS-tagging based) — AI-Generated Text Detection")
print("=" * 70)
print(f"  GPUs available: {tf.config.list_physical_devices('GPU')}")

# ==============================================================================
# 2. DATA — the project's 3 FIXED splits (same rows every English script
# uses), NOT re-loaded/re-balanced/re-split here.
# ==============================================================================
print("\n" + "─" * 70)
print("  Loading fixed splits ...")
gc.collect()

required_cols = ["text", "label", DOMAIN_COLUMN]

def _load_split(path, name):
    d = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    missing = [c for c in required_cols if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in {name} split: {missing}")
    d = d[required_cols].dropna().reset_index(drop=True)
    d["label"] = d["label"].astype(int)
    d[DOMAIN_COLUMN] = d[DOMAIN_COLUMN].astype(str).str.strip().str.lower()
    d = d[d["text"].astype(str).str.len() > 20].reset_index(drop=True)
    return d

train_df = _load_split(TRAIN_SPLIT_FILE, "train")
val_df   = _load_split(VAL_SPLIT_FILE,   "validation")
test_df  = _load_split(TEST_SPLIT_FILE,  "test")

print(f"  Train: {len(train_df)}   Val: {len(val_df)}   Test: {len(test_df)}")
for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"  {name} label distribution:\n{d['label'].value_counts().to_string()}")

# ==============================================================================
# 3. NORMALIZATION — SAME 8-step English pipeline as every other baseline in
# this project (shared text across all baselines for a fair, identical-input
# comparison). POS-tag CATEGORY sequences (NN/VB/JJ/RB/other) are not tied to
# exact punctuation surface forms, so there is no fidelity conflict here
# (unlike StyloAI's raw punctuation-count features).
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

# ==============================================================================
# 4. PREPROCESSING — POS tagging  (Section IV.A)
# ==============================================================================
print("\n" + "─" * 70)
print("  POS tagging and building the POS-category vocabulary ...")

# Paper's explicit mapping: NN:1, VB:2, JJ:3, RB:4. We extend with an
# "other" category (5) instead of silently collapsing all other tags to 0,
# which would be indistinguishable from padding.
POS_MAP = {"NN": 1, "VB": 2, "JJ": 3, "RB": 4}
OTHER_TAG = 5
PAD = 0
VOCAB_SIZE = 6  # 0=pad, 1-4=NN/VB/JJ/RB, 5=other

def pos_category(tag):
    for prefix, idx in POS_MAP.items():
        if tag.startswith(prefix):
            return idx
    return OTHER_TAG

def text_to_pos_sequence(text, max_len):
    try:
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
    except Exception:
        return [PAD] * max_len
    seq = [pos_category(t) for _, t in tags][:max_len]
    if len(seq) < max_len:
        seq = seq + [PAD] * (max_len - len(seq))
    return seq

# Determine a practical max_len from the data if not already reasonable
sample_lengths = []
for t in train_df["text_processed"].sample(min(300, len(train_df)), random_state=SEED):
    try:
        sample_lengths.append(len(word_tokenize(t)))
    except Exception:
        pass
p95 = int(np.percentile(sample_lengths, 95)) if sample_lengths else CFG["max_len"]
CFG["max_len"] = min(max(CFG["max_len"], 50), max(p95, 50))
print(f"  Using max_len = {CFG['max_len']} tokens (95th percentile of sampled train lengths: {p95})")

def build_pos_matrix(dframe, desc):
    seqs = []
    for t in tqdm(dframe["text_processed"], desc=f"  POS tagging [{desc}]"):
        seqs.append(text_to_pos_sequence(t, CFG["max_len"]))
    return np.array(seqs, dtype=np.int32)

X_train = build_pos_matrix(train_df, "train")
X_val   = build_pos_matrix(val_df,   "val")
X_test  = build_pos_matrix(test_df,  "test")
y_train = train_df["label"].values.astype(np.float32)
y_val   = val_df["label"].values.astype(np.float32)
y_test  = test_df["label"].values.astype(np.float32)

print(f"  X_train shape: {X_train.shape}")

# ==============================================================================
# 5. MODEL — Initial Module (CNN) + Bi-LSTM + Attention + Residual Concat
# ==============================================================================
print("\n" + "=" * 70)
print("  Building the Bi-LSTM + Attention model")
print("=" * 70)

class AdditiveAttention(layers.Layer):
    """Additive (Bahdanau-style) self-attention over a sequence, producing a
    single context vector per document (Eq. 4-5 / Fig. 6 of the paper).
    CORRECTED this round (see header note): a single projection + V, i.e.
    TWO dense layers total, matching the script's own stated "two dense
    layers + tanh + softmax" design — the previous version's extra W2
    matrix (summed with W1 on the same input) was mathematically redundant
    (W1(h)+W2(h) == (W1+W2)(h), a single combined matrix), so dropping it
    loses no expressive capacity."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, hidden_states):
        # hidden_states: (batch, timesteps, features)
        score = self.V(tf.nn.tanh(self.W(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)            # (batch, timesteps, 1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # (batch, features)
        return context, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


def build_model(cfg, vocab_size, max_len):
    inp = layers.Input(shape=(max_len,), name="pos_sequence")
    emb = layers.Embedding(input_dim=vocab_size, output_dim=cfg["embedding_dim"],
                           mask_zero=False, name="embedding")(inp)

    # --- Initial Module: Conv1D + MaxPooling (local contextual features) ---
    conv = layers.Conv1D(cfg["conv_filters"], cfg["conv_kernel"], activation="relu",
                         padding="same", name="conv1d")(emb)
    pooled = layers.MaxPooling1D(pool_size=2, name="maxpool")(conv)
    initial_flat = layers.Flatten(name="initial_flatten")(pooled)

    # --- Bi-LSTM + Attention (long-range dependencies) ---
    bilstm = layers.Bidirectional(
        layers.LSTM(cfg["lstm_units"], return_sequences=True,
                   recurrent_activation="tanh",
                   kernel_initializer="glorot_uniform"),
        name="bilstm")(emb)
    context, _ = AdditiveAttention(cfg["attention_units"], name="attention")(bilstm)

    # --- Residual Concatenation (Eq. 6: F_Initial (+) F_Att) ---
    concat = layers.Concatenate(name="residual_concat")([initial_flat, context])

    # --- Classification Module ---
    dense = layers.Dense(cfg["dense_units"], activation="relu", name="dense_1")(concat)
    dense = layers.Dropout(cfg["dropout"], name="dropout")(dense)
    out = layers.Dense(1, activation="sigmoid", name="output")(dense)

    model = models.Model(inputs=inp, outputs=out, name="BiLSTM_Attention_Detector")
    return model

model = build_model(CFG, VOCAB_SIZE, CFG["max_len"])

optimizer = optimizers.Nadam(learning_rate=CFG["learning_rate"], clipvalue=1.0)  # Adam + Nesterov momentum
model.compile(optimizer=optimizer, loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])
model.summary()

total_params = model.count_params()
print(f"\n  Total trainable parameters: {total_params/1e6:.2f}M")

# ==============================================================================
# 6. TRAINING
# ==============================================================================
print("\n" + "─" * 70)
print("  Training ...")

cb = [
    callbacks.EarlyStopping(monitor="val_loss", patience=CFG["patience"],
                            restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=CFG["batch_size"],
    epochs=CFG["max_epochs"],
    callbacks=cb,
    verbose=1,
)

# ==============================================================================
# 7. PUBLICATION STYLE — same IMG_COLORS palette + style_ax/render_table_png
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
    "human":    "#1E90FF",
    "ai":       "#FF6347",
    "roc_line": "#2E97F0",
    "val_f1":   "#4CAF50",
    "val_acc":  "#FF9800",
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
# 8. DECISION THRESHOLD — FIXED at 0.50 (per explicit request, 2026-10-24:
# "خلي العتبات بالكودات الجديدة 50"). The F1-vs-threshold curve over the
# validation grid is still computed and plotted/saved below purely as a
# DIAGNOSTIC, but it no longer overrides the cutoff used for the reported
# test-set metrics — that is always 0.50.
# ==============================================================================
print("\n  Decision threshold fixed at 0.50 — computing validation F1 curve for reference only ...")
val_probs_ai = model.predict(X_val, batch_size=CFG["batch_size"]).flatten()

threshold_rows = []
for th in CFG["th_grid"]:
    preds_th = (val_probs_ai >= th).astype(int)
    f1_th = f1_score(y_val, preds_th, zero_division=0)
    threshold_rows.append({"threshold": round(float(th), 2), "val_f1": round(f1_th, 4)})

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_excel(f"{TABLES_DIR}/threshold_search.xlsx", index=False)
best_threshold = 0.5
best_threshold_f1 = f1_score(y_val, (val_probs_ai >= 0.5).astype(int), zero_division=0)
print(f"  Using FIXED threshold: {best_threshold:.2f}  (val F1 at this threshold={best_threshold_f1:.4f})")

fig_th, ax_th = plt.subplots(figsize=(5.5, 4.2))
ax_th.plot(threshold_df["threshold"], threshold_df["val_f1"], color=IMG_COLORS["val_f1"], lw=2)
ax_th.axvline(best_threshold, color="black", ls="--", lw=1, label=f"used={best_threshold:.2f}")
style_ax(ax_th, xlabel="Decision threshold", ylabel="Validation F1", title="Threshold Diagnostic (Validation)")
ax_th.legend(fontsize=9)
plt.tight_layout()
fig_th.savefig(f"{PLOTS_DIR}/threshold_search.png"); plt.show()

# ==============================================================================
# 9. EVALUATION ON TEST SET
# ==============================================================================
print("\n" + "─" * 70)
print("  Evaluating on held-out test set ...")

y_prob = model.predict(X_test, batch_size=CFG["batch_size"]).flatten()
y_pred = (y_prob >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

metrics_df = pd.DataFrame([{
    "Method": "Bi-LSTM+Attention (POS-based)", "Threshold": round(best_threshold, 2),
    "Accuracy": round(accuracy, 4), "F1": round(f1, 4), "Precision": round(precision, 4),
    "Recall": round(recall, 4), "MCC": round(mcc, 4), "AUC": round(roc_auc, 4),
}])
print("\n  === Test metrics ===")
print(metrics_df.to_string(index=False))
metrics_df.to_excel(f"{TABLES_DIR}/test_metrics.xlsx", index=False)
render_table_png(metrics_df, f"{PLOTS_DIR}/table_test_metrics.png", figsize=(13, 2.0))

# ==============================================================================
# 10. UNIFIED PERFORMANCE TABLE — Document-level Overall + per-domain, same
# structure as every other baseline script's unified table.
# ==============================================================================
test_df = test_df.reset_index(drop=True)
test_df["prob_ai"] = y_prob
test_df["pred_label"] = y_pred
test_df["true_label"] = y_test

domain_rows = []
for dom in sorted(test_df[DOMAIN_COLUMN].unique()):
    mask = (test_df[DOMAIN_COLUMN] == dom).values
    sub_true, sub_pred, sub_prob = y_test[mask], y_pred[mask], y_prob[mask]
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
# 11. PLOTS
# ==============================================================================
print("\n  Generating plots ...")
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(f"Bi-LSTM + Attention (POS-tagging based)\n"
             f"F1={f1:.4f}  AUC={roc_auc:.4f}  threshold={best_threshold:.2f}  Params={total_params/1e6:.1f}M",
             fontsize=12, fontweight="bold")

# (a) Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
_plot_cm(axes[0, 0], cm, "Confusion Matrix", tn, fp, fn, tp)

# (b) ROC
axes[0, 1].plot(fpr, tpr, color=IMG_COLORS["roc_line"], lw=2, label=f"Bi-LSTM+Attn (AUC={roc_auc:.4f})")
axes[0, 1].plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
style_ax(axes[0, 1], xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
axes[0, 1].legend(fontsize=9)

# (c) P(AI) distribution
for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    axes[0, 2].hist(y_prob[y_test == lbl], bins=40, alpha=0.6, color=color, label=name, density=True)
axes[0, 2].axvline(best_threshold, color="black", lw=1.2, linestyle="--", label=f"threshold={best_threshold:.2f}")
style_ax(axes[0, 2], xlabel="P(AI)", ylabel="Density", title="P(AI) Distribution")
axes[0, 2].legend(fontsize=8)

# (d) Training loss
ep = range(1, len(history.history["loss"]) + 1)
axes[1, 0].plot(ep, history.history["loss"], color=IMG_COLORS["human"], lw=2, label="Train Loss")
axes[1, 0].plot(ep, history.history["val_loss"], color=IMG_COLORS["ai"], lw=2, marker="o", label="Val Loss")
style_ax(axes[1, 0], xlabel="Epoch", title="Training Loss")
axes[1, 0].legend(fontsize=9)

# (e) Accuracy
axes[1, 1].plot(ep, history.history["accuracy"], color=IMG_COLORS["val_f1"], lw=2, label="Train Acc")
axes[1, 1].plot(ep, history.history["val_accuracy"], color=IMG_COLORS["val_acc"], lw=2, marker="s", label="Val Acc")
style_ax(axes[1, 1], xlabel="Epoch", title="Accuracy")
axes[1, 1].set_ylim(0, 1.05); axes[1, 1].legend(fontsize=9)

# (f) Precision / Recall (train)
axes[1, 2].plot(ep, history.history["precision"], color="#9C27B0", lw=2, label="Train Precision")
axes[1, 2].plot(ep, history.history["recall"], color="#00BCD4", lw=2, label="Train Recall")
style_ax(axes[1, 2], xlabel="Epoch", title="Precision / Recall (train)")
axes[1, 2].set_ylim(0, 1.05); axes[1, 2].legend(fontsize=9)

plt.tight_layout()
plot_path = f"{PLOTS_DIR}/bilstm_attention_results_dashboard.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"  Dashboard saved -> {plot_path}")

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
# 12. SAVE
# ==============================================================================
print("\n" + "─" * 70)
print(f"  Saving to Drive -> {METHOD_DIR}")

model.save(f"{METHOD_DIR}/bilstm_attention_model.keras")
with open(f"{METHOD_DIR}/bilstm_attention_config.json", "w") as f_out:
    json.dump({"threshold": best_threshold, "val_f1": best_threshold_f1,
               "max_len": CFG["max_len"], "total_params": int(total_params)}, f_out, indent=2)

pd.DataFrame(history.history).to_csv(f"{TABLES_DIR}/training_history.csv", index=False)
print(f"  Saved -> {METHOD_DIR}")

# ==============================================================================
# 13. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY — Bi-LSTM + Attention (POS-tagging based)")
print("=" * 70)
print(f"  max_len      : {CFG['max_len']}   Params: {total_params/1e6:.2f}M")
print(f"  Threshold    : {best_threshold:.2f}  (fixed)")
print(f"  Train docs   : {len(train_df)}   Val docs: {len(val_df)}   Test docs: {len(test_df)}")
print()
print(metrics_df.to_string(index=False))
print()
print(unified_df.to_string(index=False))
print(f"\n  Drive path   : {METHOD_DIR}")
print("=" * 70)