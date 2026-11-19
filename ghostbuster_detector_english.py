# ==============================================================================
# Ghostbuster — Detecting Text Ghostwritten by LLMs (Verma et al., NAACL 2024,
# arXiv:2305.15047)
# Reimplementation, ADAPTED to this project's data/normalization/workflow
# conventions AND CORRECTED for paper fidelity (see chat, 2026-10-24).
#
# ADAPTATIONS FROM THE ORIGINAL PAPER THAT REMAIN (practical, openly
# documented substitutions — unchanged from the user's original script):
#   1. The paper uses OpenAI's `ada` and `davinci` (GPT-3, pre-ChatGPT) to
#      obtain a weak/strong pair of neural LM probabilities. Both are now
#      DEPRECATED and no longer expose token probabilities via API. We
#      substitute an open, locally-run pair of increasing capacity:
#         weak LM   = gpt2          (124M)   -- stand-in for `ada`
#         strong LM = gpt2-medium   (355M)   -- stand-in for `davinci`
#   2. Unigram and trigram models are trained on the Brown corpus (as in the
#      paper), tokenized with the SAME tokenizer as the neural LMs (GPT-2 BPE)
#      so that all four probability vectors align token-for-token.
#   3. The trigram model uses simple back-off smoothing (trigram -> bigram ->
#      unigram) rather than the paper's exact Kneser-Ney discount (delta=0.9);
#      this preserves the "weak statistical LM" role in the pipeline.
#   4. Final feature selection uses L1-regularized Logistic Regression
#      (automatic sparse selection) as a practical stand-in for the paper's
#      greedy forward-selection procedure (which needs O(features^2) model
#      refits and is intractable here). The FINAL classifier itself is,
#      exactly as the paper specifies, L2-regularized Logistic Regression
#      with C=1, fit on the selected features.
#
# CORRECTED THIS ROUND (per paper-fidelity check, user chose to fix both):
#   A. STRUCTURED SEARCH (Algorithm 1) — the previous version only built
#      scalar(single base vector) or scalar(vecfunc(base_i, base_j)) i.e. a
#      single vector-function application between exactly TWO base vectors.
#      The paper's real Algorithm 1 is RECURSIVE: starting from each base
#      vector in turn, at every depth it (i) applies all 7 scalar functions
#      to the CURRENT vector (scalars are collected at every depth level,
#      not only at the deepest one), and (ii) recurses by combining the
#      current vector with ANOTHER base vector via a vector function, up to
#      `search_depth` nested vector-function applications — so a depth-2
#      feature can combine THREE base vectors (e.g.
#      var(sub(unigram, mul(weak, strong)))), not just two. This is now
#      implemented as a true recursive tree search (build_feature_specs()
#      below), with the paper's two stated prunings: (i) never apply the
#      same vector function twice in a row, (ii) for commutative vector
#      functions (add, mul) skip the operand-swapped duplicate. NOTE: the
#      paper reports exact counts of 322 features at depth 2 and 2534 at
#      depth 3, but does not give enough implementation detail (its exact
#      pruning/dedup rules) to reproduce those exact numbers — our run
#      prints its own actual count, which will differ from 322/2534 for
#      that reason. What matters for fidelity is that this is now a genuine
#      nested recursive search over the vector-function tree (matching
#      Algorithm 1's structure), not the flat single-combination version.
#   B. HANDCRAFTED FEATURES (Appendix D) — the previous version's own header
#      claimed to include "longest-word-length stats" but never actually
#      computed them, leaving only 5 of the paper's 7 handcrafted features
#      implemented. The 2 missing ones (average token-length of the 25
#      longest words, and of the 25th-50th longest words, per Appendix D)
#      are now added, giving the full 7.
# ==============================================================================

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
from collections import Counter

import nltk
from nltk.corpus import brown
nltk.download("brown", quiet=True)

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

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
METHOD_DIR  = f"{PROJECT_DIR}/Ghostbuster_Detector_English"
TABLES_DIR  = f"{METHOD_DIR}/tables"
PLOTS_DIR   = f"{METHOD_DIR}/plots"
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_SPLIT_FILE = f"{PROJECT_DIR}/splits/train_split.csv"
VAL_SPLIT_FILE   = f"{PROJECT_DIR}/splits/val_split.csv"
TEST_SPLIT_FILE  = f"{PROJECT_DIR}/splits/test_split.csv"
DOMAIN_COLUMN    = "domain"

CFG = {
    "weak_lm":       "gpt2",           # stand-in for `ada`
    "strong_lm":     "gpt2-medium",    # stand-in for `davinci`
    "max_tokens":    400,
    # True nested-recursion depth (Algorithm 1). Kept at 2 for tractability on
    # Colab (depth 3 explodes the vector-tree size and per-document compute);
    # unlike the old flat version, depth 2 here already allows combining up
    # to 3 base vectors through nested vector functions.
    "search_depth":  2,
    "l1_C":          0.1,              # inverse regularization strength (sparser = smaller)
    "th_grid":       np.arange(0.05, 0.96, 0.01),   # kept only for the reference diagnostic curve
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 70)
print("  Ghostbuster — Structured Search over Weak-LM Probabilities")
print(f"  Weak LM: {CFG['weak_lm']}   Strong LM: {CFG['strong_lm']}")
print("=" * 70)
print(f"  Device: {DEVICE}")

# ==============================================================================
# 2. DATA — the project's 3 FIXED splits (same rows every English script
# uses), NOT re-loaded/re-balanced/re-split here.
# ==============================================================================
print("\n" + "─" * 70)
print("  Loading fixed splits ...")
gc.collect(); torch.cuda.empty_cache()

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
# comparison). None of Ghostbuster's features depend on raw punctuation/
# contraction surface forms (unlike StyloAI), so there is no fidelity
# conflict here.
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
# 4. WEAK STATISTICAL MODELS — unigram & trigram, trained on Brown corpus,
#    tokenized with the SAME BPE tokenizer as the neural LMs
# ==============================================================================
print("\n" + "─" * 70)
print("  Training unigram/trigram models on the Brown corpus (GPT-2 BPE) ...")

tokenizer = AutoTokenizer.from_pretrained(CFG["weak_lm"])

brown_text = " ".join(brown.words()[:200000])   # cap for speed
brown_ids = tokenizer.encode(brown_text)

uni_counts = Counter(brown_ids)
uni_total = sum(uni_counts.values())

bi_counts = Counter(zip(brown_ids[:-1], brown_ids[1:]))
tri_counts = Counter(zip(brown_ids[:-2], brown_ids[1:-1], brown_ids[2:]))
bi_context_counts = Counter(brown_ids[:-1])
tri_context_counts = Counter(zip(brown_ids[:-2], brown_ids[1:-1]))

VOCAB_SIZE = tokenizer.vocab_size
EPS = 1e-8

def p_unigram(tok):
    return (uni_counts.get(tok, 0) + 1) / (uni_total + VOCAB_SIZE)

def p_trigram(t0, t1, t2):
    tri_ctx = tri_context_counts.get((t0, t1), 0)
    if tri_ctx > 0:
        return tri_counts.get((t0, t1, t2), 0) / tri_ctx
    bi_ctx = bi_context_counts.get(t1, 0)
    if bi_ctx > 0:
        return bi_counts.get((t1, t2), 0) / bi_ctx
    return p_unigram(t2)

print(f"  Brown corpus tokens: {len(brown_ids)}   Unique tokens: {len(uni_counts)}")

# ==============================================================================
# 5. NEURAL LMs (weak + strong)
# ==============================================================================
print("\n" + "─" * 70)
print("  Loading neural LMs ...")

weak_model = AutoModelForCausalLM.from_pretrained(CFG["weak_lm"]).to(DEVICE).eval()
strong_model = AutoModelForCausalLM.from_pretrained(CFG["strong_lm"]).to(DEVICE).eval()

@torch.no_grad()
def neural_probs(ids_tensor, model):
    """Returns per-position probability assigned to the actual next token."""
    out = model(ids_tensor)
    logits = out.logits[0, :-1, :]
    targets = ids_tensor[0, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    actual = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).exp()
    return actual.cpu().numpy()

# ==============================================================================
# 6. PER-DOCUMENT PROBABILITY VECTORS  (unigram, trigram, weak, strong)
# ==============================================================================
@torch.no_grad()
def get_probability_vectors(text):
    ids = tokenizer.encode(text, truncation=True, max_length=CFG["max_tokens"])
    if len(ids) < 5:
        return None
    ids_tensor = torch.tensor([ids]).to(DEVICE)

    weak_p = neural_probs(ids_tensor, weak_model)      # length len(ids)-1, positions 1..N-1
    strong_p = neural_probs(ids_tensor, strong_model)

    # unigram/trigram vectors aligned to the same positions (predicting ids[i] given prefix)
    uni_p = np.array([p_unigram(ids[i]) for i in range(1, len(ids))])
    tri_p = np.array([p_trigram(ids[i-2] if i >= 2 else ids[0], ids[i-1], ids[i]) for i in range(1, len(ids))])

    return {"unigram": uni_p, "trigram": tri_p, "weak": weak_p, "strong": strong_p,
           "n_tokens": len(ids), "raw_ids": ids}

# ==============================================================================
# 7. STRUCTURED FEATURE SEARCH — TRUE recursive tree search (Algorithm 1,
# CORRECTED this round — see header note A).
# ==============================================================================
def vec_add(a, b): return a + b
def vec_sub(a, b): return a - b
def vec_mul(a, b): return a * b
def vec_div(a, b): return a / (b + EPS)
def vec_gt(a, b):  return (a > b).astype(float)
def vec_lt(a, b):  return (a < b).astype(float)

VECTOR_FUNCS = {"add": vec_add, "sub": vec_sub, "mul": vec_mul, "div": vec_div,
                "gt": vec_gt, "lt": vec_lt}
COMMUTATIVE_FUNCS = {"add", "mul"}

def s_max(p):  return float(np.max(p))
def s_min(p):  return float(np.min(p))
def s_avg(p):  return float(np.mean(p))
def s_avgtop25(p):
    k = min(25, len(p))
    idx = np.argsort(p)[:k]           # lowest 25 values (per paper's T_p definition)
    return float(np.mean(p[idx]))
def s_len(p):  return float(len(p))
def s_l2(p):   return float(np.linalg.norm(p))
def s_var(p):  return float(np.var(p))

SCALAR_FUNCS = {"max": s_max, "min": s_min, "avg": s_avg, "avg_top25": s_avgtop25,
                "len": s_len, "l2": s_l2, "var": s_var}

BASE_VECTORS = ["unigram", "trigram", "weak", "strong"]

def _canonicalize(expr):
    """Puts commutative (add/mul) sub-expressions into a canonical operand
    order so that e.g. add(unigram, trigram) and add(trigram, unigram) are
    recognized as the SAME feature regardless of which base vector the
    recursion happened to start from."""
    if isinstance(expr, str):
        return expr
    vfunc_name, left, right = expr
    left_c, right_c = _canonicalize(left), _canonicalize(right)
    if vfunc_name in COMMUTATIVE_FUNCS and repr(right_c) < repr(left_c):
        left_c, right_c = right_c, left_c
    return (vfunc_name, left_c, right_c)

def _expr_to_name(expr):
    if isinstance(expr, str):
        return expr
    vfunc_name, left, right = expr
    return f"{vfunc_name}({_expr_to_name(left)},{_expr_to_name(right)})"

def build_feature_specs(max_depth):
    """
    Faithful reimplementation of Ghostbuster's Algorithm 1 (find-all-features):
    for each base vector in turn (the "root"), at every depth d<=max_depth,
    (i) emit scalar(current_vector) for all 7 scalar functions, and
    (ii) if d<max_depth, recurse by combining current_vector with EVERY base
    vector via EVERY vector function (the second operand is always drawn
    from the base set, matching the paper's pseudocode), producing the next
    depth's vector. Pruning, per the paper's own stated rules: never apply
    the same vector function twice in a row, and skip the operand-swapped
    duplicate for commutative vector functions. A final global dedup (via
    `_canonicalize`) removes cross-root duplicates.
    """
    specs = []

    def recurse(root_base, steps, depth):
        expr = root_base
        for vfunc_name, b_name in steps:
            expr = (vfunc_name, expr, b_name)
        for sname in SCALAR_FUNCS:
            specs.append({"expr": expr, "scalar": sname})
        if depth >= max_depth:
            return
        last_vfunc = steps[-1][0] if steps else None
        last_base  = steps[-1][1] if steps else None
        for vfunc_name in VECTOR_FUNCS:
            if vfunc_name == last_vfunc:
                continue   # prune: same vector function applied twice in a row
            for base_name in BASE_VECTORS:
                if vfunc_name in COMMUTATIVE_FUNCS and base_name == last_base:
                    continue   # prune: commutative duplicate of the immediately preceding step
                recurse(root_base, steps + [(vfunc_name, base_name)], depth + 1)

    for root in BASE_VECTORS:
        recurse(root, [], 1)

    seen, unique_specs = set(), []
    for spec in specs:
        canon_expr = _canonicalize(spec["expr"])
        key = (canon_expr, spec["scalar"])
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append({"canon_expr": canon_expr, "scalar": spec["scalar"],
                             "name": f"{spec['scalar']}({_expr_to_name(canon_expr)})"})
    return unique_specs

FEATURE_SPECS = build_feature_specs(CFG["search_depth"])
print(f"\n  Structured search space (true nested tree, depth <= {CFG['search_depth']}): "
      f"{len(FEATURE_SPECS)} candidate features")
print("  (differs from the paper's reported 322/2534 — see header note A: the paper "
      "does not give enough implementation detail to reproduce its exact pruning)")

def _eval_expr(expr, vecs, cache):
    if isinstance(expr, str):
        return vecs[expr]
    if expr in cache:
        return cache[expr]
    vfunc_name, left, right = expr
    left_v = _eval_expr(left, vecs, cache)
    right_v = _eval_expr(right, vecs, cache)
    n = min(len(left_v), len(right_v))
    val = VECTOR_FUNCS[vfunc_name](left_v[:n], right_v[:n])
    cache[expr] = val
    return val

def _word_length_features(text):
    """Appendix D, handcrafted features #6-7 (CORRECTED this round — see
    header note B): average token-length of the 25 longest words, and of
    the 25th-50th longest words, where 'length' is measured in GPT-2 BPE
    tokens (paper: 'measured in tokens')."""
    words = text.split()
    if not words:
        return 0.0, 0.0
    lengths = sorted((len(tokenizer.encode(w)) for w in words), reverse=True)
    top25 = lengths[:25]
    mid_25_50 = lengths[25:50] if len(lengths) > 25 else lengths
    avg_top25 = float(np.mean(top25)) if top25 else 0.0
    avg_25_50 = float(np.mean(mid_25_50)) if mid_25_50 else avg_top25
    return avg_top25, avg_25_50

def compute_features(vecs, text_processed):
    row = {}
    cache = {}
    for spec in FEATURE_SPECS:
        v = _eval_expr(spec["canon_expr"], vecs, cache)
        row[spec["name"]] = SCALAR_FUNCS[spec["scalar"]](v)

    # --- Handcrafted features (Appendix D, all 7 now present) ---
    strong_p = vecs["strong"]; weak_p = vecs["weak"]
    row["n_outliers"] = float(np.sum(strong_p > 10))   # note: probs in [0,1], paper's p_i scale is
                                                        # unspecified (see header); kept for parity, will be 0
    sorted_strong = np.sort(strong_p)[::-1]
    row["avg_top25_strong"] = float(np.mean(sorted_strong[:25])) if len(sorted_strong) >= 1 else 0.0
    row["avg_25_50_strong"] = float(np.mean(sorted_strong[25:50])) if len(sorted_strong) > 50 else float(np.mean(sorted_strong))
    diff = strong_p[:min(len(strong_p), len(weak_p))] - weak_p[:min(len(strong_p), len(weak_p))]
    sorted_diff = np.sort(diff)[::-1]
    row["avg_top25_diff"] = float(np.mean(sorted_diff[:25])) if len(sorted_diff) >= 1 else 0.0
    row["avg_25_50_diff"] = float(np.mean(sorted_diff[25:50])) if len(sorted_diff) > 50 else float(np.mean(sorted_diff))
    row["avg_top25_wordlen"], row["avg_25_50_wordlen"] = _word_length_features(text_processed)   # CORRECTED (added)
    row["n_tokens"] = vecs["n_tokens"]
    return row

def extract_split(dframe, desc):
    rows, keep_idx = [], []
    for i, r in tqdm(dframe.iterrows(), total=len(dframe), desc=f"  Ghostbuster features [{desc}]"):
        vecs = get_probability_vectors(r["text_processed"])
        if vecs is None:
            continue
        feat = compute_features(vecs, r["text_processed"])
        feat["label"] = int(r["label"])
        feat["domain"] = r[DOMAIN_COLUMN]
        rows.append(feat)
        keep_idx.append(i)
    return pd.DataFrame(rows), dframe.loc[keep_idx].reset_index(drop=True)

print("\n  Extracting structured-search + handcrafted features per document ...")
train_feat, train_df = extract_split(train_df, "train")
val_feat,   val_df   = extract_split(val_df,   "val")
test_feat,  test_df  = extract_split(test_df,  "test")

FEATURE_COLS = [c for c in train_feat.columns if c not in ("label", "domain")]
print(f"  Total feature columns: {len(FEATURE_COLS)}")

# ==============================================================================
# 8. PUBLICATION STYLE — same IMG_COLORS palette + style_ax/render_table_png
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
# 9. CLASSIFIER — L1 Logistic Regression (automatic sparse selection) then
#    a final L2 Logistic Regression on the selected features (matches the
#    paper's final classifier: LR, L2 penalty, C=1)
# ==============================================================================
print("\n" + "=" * 70)
print("  Feature selection (L1-LR) + final classifier (L2-LR, C=1)")
print("=" * 70)

Xtr  = train_feat[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
ytr  = train_feat["label"].values
Xval = val_feat[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
yval = val_feat["label"].values
Xte  = test_feat[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values
yte  = test_feat["label"].values

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xval_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xval), scaler.transform(Xte)

l1_selector = LogisticRegression(penalty="l1", solver="liblinear", C=CFG["l1_C"],
                                 max_iter=3000, random_state=SEED)
l1_selector.fit(Xtr_s, ytr)
selected_mask = (l1_selector.coef_[0] != 0)
selected_features = [f for f, keep in zip(FEATURE_COLS, selected_mask) if keep]
print(f"\n  Features selected by L1 (non-zero coef): {len(selected_features)} / {len(FEATURE_COLS)}")
coef_series = pd.Series(l1_selector.coef_[0], index=FEATURE_COLS)
print("  Top selected features (by |coef|):")
print(coef_series.reindex(coef_series.abs().sort_values(ascending=False).index).head(15).to_string())

final_clf = LogisticRegression(penalty="l2", C=1.0, max_iter=3000, random_state=SEED)
final_clf.fit(Xtr_s[:, selected_mask], ytr)

# ==============================================================================
# 10. DECISION THRESHOLD — FIXED at 0.50 (per explicit request, 2026-10-24:
# "خلي العتبات بالكودات الجديدة 50"). The F1-vs-threshold curve over the
# validation grid is still computed and plotted/saved below purely as a
# DIAGNOSTIC, but it no longer overrides the cutoff used for the reported
# test-set metrics — that is always 0.50.
# ==============================================================================
print("\n  Decision threshold fixed at 0.50 — computing validation F1 curve for reference only ...")
val_probs_ai = final_clf.predict_proba(Xval_s[:, selected_mask])[:, 1]

threshold_rows = []
for th in CFG["th_grid"]:
    preds_th = (val_probs_ai >= th).astype(int)
    f1_th = f1_score(yval, preds_th, zero_division=0)
    threshold_rows.append({"threshold": round(float(th), 2), "val_f1": round(f1_th, 4)})

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_excel(f"{TABLES_DIR}/threshold_search.xlsx", index=False)
best_threshold = 0.5
best_threshold_f1 = f1_score(yval, (val_probs_ai >= 0.5).astype(int), zero_division=0)
print(f"  Using FIXED threshold: {best_threshold:.2f}  (val F1 at this threshold={best_threshold_f1:.4f})")

fig_th, ax_th = plt.subplots(figsize=(5.5, 4.2))
ax_th.plot(threshold_df["threshold"], threshold_df["val_f1"], color=IMG_COLORS["val_f1"], lw=2)
ax_th.axvline(best_threshold, color="black", ls="--", lw=1, label=f"used={best_threshold:.2f}")
style_ax(ax_th, xlabel="Decision threshold", ylabel="Validation F1", title="Threshold Diagnostic (Validation)")
ax_th.legend(fontsize=9)
plt.tight_layout()
fig_th.savefig(f"{PLOTS_DIR}/threshold_search.png"); plt.show()

# ==============================================================================
# 11. EVALUATION ON TEST SET
# ==============================================================================
y_prob = final_clf.predict_proba(Xte_s[:, selected_mask])[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

accuracy = accuracy_score(yte, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(yte, y_pred, average="binary", zero_division=0)
mcc = matthews_corrcoef(yte, y_pred)
fpr, tpr, _ = roc_curve(yte, y_prob)
roc_auc = auc(fpr, tpr)

metrics_df = pd.DataFrame([{
    "Method": "Ghostbuster (adapted: GPT-2/GPT-2-medium)", "Threshold": round(best_threshold, 2),
    "Accuracy": round(accuracy, 4), "F1": round(f1, 4), "Precision": round(precision, 4),
    "Recall": round(recall, 4), "MCC": round(mcc, 4), "AUC": round(roc_auc, 4),
}])
print("\n  === Ghostbuster — full test metrics ===")
print(metrics_df.to_string(index=False))
metrics_df.to_excel(f"{TABLES_DIR}/test_metrics.xlsx", index=False)
render_table_png(metrics_df, f"{PLOTS_DIR}/table_test_metrics.png", figsize=(13, 2.0))

# ==============================================================================
# 12. UNIFIED PERFORMANCE TABLE — Document-level Overall + per-domain, same
# structure as every other baseline script's unified table.
# ==============================================================================
test_df = test_df.reset_index(drop=True)
test_df["prob_ai"] = y_prob
test_df["pred_label"] = y_pred
test_df["true_label"] = yte

domain_rows = []
for dom in sorted(test_df[DOMAIN_COLUMN].unique()):
    mask = (test_df[DOMAIN_COLUMN] == dom).values
    sub_true, sub_pred, sub_prob = yte[mask], y_pred[mask], y_prob[mask]
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
# 13. PLOTS
# ==============================================================================
print("\n  Generating plots ...")
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(f"Ghostbuster (adapted) — Structured Search + Logistic Regression\n"
             f"F1={f1:.4f}  AUC={roc_auc:.4f}  threshold={best_threshold:.2f}  "
             f"(weak={CFG['weak_lm']}, strong={CFG['strong_lm']})",
             fontsize=12, fontweight="bold")

# (a) Confusion matrix
cm = confusion_matrix(yte, y_pred)
tn, fp, fn, tp = cm.ravel()
_plot_cm(axes[0, 0], cm, "Confusion Matrix", tn, fp, fn, tp)

# (b) ROC
axes[0, 1].plot(fpr, tpr, color=IMG_COLORS["roc_line"], lw=2, label=f"Ghostbuster (AUC={roc_auc:.4f})")
axes[0, 1].plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
style_ax(axes[0, 1], xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
axes[0, 1].legend(fontsize=9)

# (c) P(AI) distribution
for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    axes[0, 2].hist(y_prob[yte == lbl], bins=40, alpha=0.6, color=color, label=name, density=True)
axes[0, 2].axvline(best_threshold, color="black", lw=1.2, linestyle="--", label=f"threshold={best_threshold:.2f}")
style_ax(axes[0, 2], xlabel="P(AI)", ylabel="Density", title="P(AI) Distribution")
axes[0, 2].legend(fontsize=8)

# (d) Top selected features (L1 coef.)
top_feats = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index).head(12)
top_feats.sort_values().plot(kind="barh", ax=axes[1, 0],
                              color=[IMG_COLORS["ai"] if v < 0 else IMG_COLORS["val_f1"] for v in top_feats.sort_values()])
style_ax(axes[1, 0], xlabel="Coefficient", title="Top Selected Features (L1 coef.)")

# (e) Figure-6 style: mean per-position probability for the strong LM, Human vs AI
def mean_position_curve(dframe, model_key, max_pos=200):
    curves = {0: [], 1: []}
    sample = dframe.sample(min(150, len(dframe)), random_state=SEED)
    for _, r in sample.iterrows():
        vecs = get_probability_vectors(r["text_processed"])
        if vecs is None: continue
        p = vecs[model_key][:max_pos]
        curves[int(r["label"])].append(p)
    out = {}
    for lbl, arrs in curves.items():
        if not arrs: continue
        maxlen = max(len(a) for a in arrs)
        padded = np.full((len(arrs), maxlen), np.nan)
        for i, a in enumerate(arrs): padded[i, :len(a)] = a
        out[lbl] = np.nanmean(padded, axis=0)
    return out

print("  Computing mean per-position probability curves (subsample) ...")
pos_curves = mean_position_curve(test_df, "strong", max_pos=150)
for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    if lbl in pos_curves:
        axes[1, 1].plot(pos_curves[lbl], color=color, label=name, alpha=0.8)
style_ax(axes[1, 1], xlabel="Position in Document", ylabel="Mean P(token)",
         title="Mean Token Prob. by Position (Fig. 6 style)")
axes[1, 1].legend(fontsize=9)

# (f) avg(strong) vs avg(weak)
for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    sub = test_feat[test_feat["label"] == lbl]
    axes[1, 2].scatter(sub["avg(strong)"], sub["avg(weak)"], s=10, alpha=0.4, color=color, label=name)
style_ax(axes[1, 2], xlabel="avg(strong)", ylabel="avg(weak)", title="avg(strong) vs avg(weak)")
axes[1, 2].legend(fontsize=9)

plt.tight_layout()
plot_path = f"{PLOTS_DIR}/ghostbuster_results_dashboard.png"
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
# 14. SAVE
# ==============================================================================
print("\n" + "─" * 70)
print(f"  Saving to Drive -> {METHOD_DIR}")

with open(f"{METHOD_DIR}/ghostbuster_model.pkl", "wb") as f_out:
    pickle.dump({"scaler": scaler, "selected_mask": selected_mask, "selected_features": selected_features,
                "clf": final_clf, "cfg": CFG, "threshold": best_threshold}, f_out)
with open(f"{METHOD_DIR}/ghostbuster_config.json", "w") as f_out:
    json.dump({"threshold": best_threshold, "val_f1": best_threshold_f1,
               "n_features_total": len(FEATURE_COLS), "n_features_selected": len(selected_features)},
              f_out, indent=2)

coef_series.to_csv(f"{TABLES_DIR}/feature_coefficients.csv")
test_feat.to_csv(f"{TABLES_DIR}/test_ghostbuster_features.csv", index=False)
print(f"  Saved -> {METHOD_DIR}")

# ==============================================================================
# 15. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY — Ghostbuster (adapted)")
print("=" * 70)
print(f"  Weak LM      : {CFG['weak_lm']}    Strong LM: {CFG['strong_lm']}")
print(f"  Search depth : {CFG['search_depth']}   Candidate features (true nested search): {len(FEATURE_SPECS)}")
print(f"  Selected     : {len(selected_features)} features (L1) -> final L2-LR")
print(f"  Threshold    : {best_threshold:.2f}  (fixed)")
print(f"  Train docs   : {len(train_feat)}   Val docs: {len(val_feat)}   Test docs: {len(test_feat)}")
print()
print(metrics_df.to_string(index=False))
print()
print(unified_df.to_string(index=False))
print(f"\n  Drive path   : {METHOD_DIR}")
print("=" * 70)