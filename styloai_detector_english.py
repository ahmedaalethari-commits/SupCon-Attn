# ==============================================================================
# StyloAI — Distinguishing AI-Generated Content with Stylometric Analysis
# (Opara, 2024, arXiv:2405.10129)
# Reimplementation, ADAPTED to this project's data/normalization/workflow
# conventions (see chat, 2026-10-24) — the 31-feature set (6 categories) and
# the 6-classifier comparison (Random Forest primary + SVM/LR/DT/KNN/
# Gradient Boosting, default Scikit-Learn settings) were VERIFIED against
# the paper and are an EXACT match. Changes made here:
#   - Loads the project's 3 FIXED splits (train/val/test, with `domain`),
#     exactly like every other English baseline here, instead of re-loading
#     a raw file and re-balancing/re-splitting it from scratch.
#   - Applies the SAME 8-step English normalization pipeline used by every
#     other baseline in this project (normalize_english_text) before
#     stylometric feature extraction.
#     >>> IMPORTANT, FLAGGED AND RESOLVED PER USER'S EXPLICIT CHOICE <<<
#     The original paper extracts all 31 features from the RAW, unnormalized
#     text (no preprocessing is described at all in Opara 2024). This
#     project's normalize_english_text expands contractions (don't -> do not)
#     and strips most non-sentence-final punctuation, which directly zeroes
#     out or badly degrades 4 of the paper's features: ContractionCount,
#     PunctuationCount, QuestionCount, ExclamationCount. This was reported to
#     the user, who explicitly chose to apply the project's unified
#     normalization anyway, for input consistency with every other baseline
#     here, accepting that these 4 features will be weak/near-degenerate
#     signals in this run (NOT a bug — an intentional, acknowledged
#     trade-off; see chat, 2026-10-24).
#   - Adds a VALIDATION-set threshold search (same np.arange(0.05, 0.96,
#     0.01) grid used across the project) for the main StyloAI (Random
#     Forest) model, instead of a fixed 0.5 cutoff — applied to TEST only,
#     never tuned on test. The 5 comparison classifiers keep the
#     conventional fixed 0.5 cutoff, consistent with how every other method
#     in this project treats its secondary ablation/comparison models.
#   - Saves to this project's Drive layout
#     (PROJECT_DIR/StyloAI_Detector_English, with tables/ and plots/
#     subfolders) and reuses the project's IMG_COLORS palette +
#     style_ax/render_table_png helpers, so this method's outputs look and
#     read like every other method's in the project.
#   - Adds a unified Document-level Overall + per-domain performance table,
#     matching every other baseline script's table.
#
# The 31 features themselves, their 6 categories, and the feature-extraction
# ALGORITHM (spaCy POS/NER + TextBlob + VADER + textstat + n-gram
# uniqueness) are UNCHANGED from the user's original script.
# ==============================================================================

import gc, os, re, random, unicodedata, pickle, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

os.system("pip install -q spacy textblob vaderSentiment wordfreq textstat")
os.system("python -m spacy download en_core_web_sm -q")

import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordfreq import top_n_list
import textstat

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix, roc_curve, auc,
    matthews_corrcoef,
)
from tqdm.auto import tqdm
from collections import Counter

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 20)

# ==============================================================================
# 0. SEED
# ==============================================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ==============================================================================
# 1. CONFIG — same project layout/conventions as every other English baseline
# ==============================================================================
PROJECT_DIR = "/content/drive/MyDrive/english_ai_detector_project"
METHOD_DIR  = f"{PROJECT_DIR}/StyloAI_Detector_English"
TABLES_DIR  = f"{METHOD_DIR}/tables"
PLOTS_DIR   = f"{METHOD_DIR}/plots"
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_SPLIT_FILE = f"{PROJECT_DIR}/splits/train_split.csv"
VAL_SPLIT_FILE   = f"{PROJECT_DIR}/splits/val_split.csv"
TEST_SPLIT_FILE  = f"{PROJECT_DIR}/splits/test_split.csv"
DOMAIN_COLUMN    = "domain"

CFG = {
    "th_grid": np.arange(0.05, 0.96, 0.01),   # same threshold-search grid used project-wide
}

print("=" * 70)
print("  StyloAI — Stylometric Analysis for AI-Generated Text Detection")
print("  31 features -> Random Forest (+ 5 comparison classifiers)")
print("=" * 70)

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
# comparison). See the header note above: this is a deliberate, user-chosen
# trade-off against the paper's own (no-preprocessing) protocol.
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
# 4. RESOURCES FOR FEATURE EXTRACTION
# ==============================================================================
print("\n" + "─" * 70)
print("  Loading NLP resources (spaCy, VADER, common-word frequency list) ...")

nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
vader = SentimentIntensityAnalyzer()
COMMON_WORDS = set(top_n_list("en", 5000))   # for ComplexVerbCount

EMOTION_WORDS = set("""
happy joy joyful glad delighted excited cheerful pleased content satisfied
sad sadness unhappy depressed miserable sorrow grief heartbroken gloomy
fear afraid scared terrified anxious worried nervous frightened panic dread
anger angry furious mad irritated annoyed rage outraged frustrated resentment
disgust disgusted revolted repulsed nauseated
surprise surprised amazed astonished shocked stunned
trust confident secure reliable faithful hopeful
love loving affection adore cherish passion warmth
hate hatred despise loathe resent bitterness
proud pride ashamed shame guilt regret embarrassed
lonely loneliness isolated abandoned rejected
relieved relief calm peaceful serene tranquil
""".split())

ABSTRACT_SUFFIXES = ("ness", "ity", "tion", "sion", "ment", "ism", "ship", "acy", "ance", "ence", "dom")
SOPHISTICATED_ADJ_SUFFIXES = ("ive", "ous", "ic", "ary", "ful", "less", "al")
FIRST_PERSON = {"i", "we", "my", "our", "me", "us", "mine", "ours", "myself", "ourselves"}
DIRECT_ADDRESS = {"you", "your", "yours", "yourself", "yourselves"}

# ==============================================================================
# 5. FEATURE EXTRACTION — 31 stylometric features (UNCHANGED algorithm,
# verified against Opara 2024, Table 1: 6 Lexical Diversity + 12 Syntactic
# Complexity + 4 Sentiment & Subjectivity + 2 Readability + 4 Named Entity +
# 3 Uniqueness & Variety = 31). Operates on the normalized text (see note
# above re: ContractionCount/PunctuationCount/QuestionCount/ExclamationCount).
# ==============================================================================
def extract_stylometric_features(text, doc):
    tokens_all = list(doc)
    words = [t for t in tokens_all if t.is_alpha]
    words_lower = [t.text.lower() for t in words]
    sentences = list(doc.sents)
    n_words = len(words) if words else 1
    n_sents = len(sentences) if sentences else 1

    # --- Lexical Diversity (6) ---
    word_count = len(words)
    unique_words = set(words_lower)
    unique_word_count = len(unique_words)
    char_count = sum(len(t.text) for t in words)
    avg_word_length = char_count / n_words
    ttr = unique_word_count / n_words
    wc = Counter(words_lower)
    hapax_rate = sum(1 for w, c in wc.items() if c == 1) / n_words

    # --- Syntactic Complexity (12) ---
    sentence_count = len(sentences)
    avg_sentence_length = word_count / n_sents
    punctuation_count = sum(1 for t in tokens_all if t.is_punct)
    stopword_count = sum(1 for t in words if t.is_stop)
    abstract_noun_count = sum(1 for t in words if t.pos_ == "NOUN" and t.text.lower().endswith(ABSTRACT_SUFFIXES))
    complex_verb_count = sum(1 for t in words if t.pos_ == "VERB" and t.text.lower() not in COMMON_WORDS)
    sophisticated_adj_count = sum(1 for t in words if t.pos_ == "ADJ" and t.text.lower().endswith(SOPHISTICATED_ADJ_SUFFIXES))
    adverb_count = sum(1 for t in words if t.pos_ == "ADV")
    complex_sentence_count = sum(1 for s in sentences
                                 if any(t.pos_ == "SCONJ" for t in s) or
                                 sum(1 for t in s if t.pos_ in ("VERB", "AUX")) > 1)
    question_count = text.count("?")
    exclamation_count = text.count("!")
    contraction_count = len(re.findall(r"\b\w+'\w+\b", text))

    # --- Sentiment & Subjectivity (4) ---
    emotion_word_count = sum(1 for w in words_lower if w in EMOTION_WORDS)
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    vader_compound = vader.polarity_scores(text)["compound"]

    # --- Readability (2) ---
    try:
        flesch = textstat.flesch_reading_ease(text)
    except Exception:
        flesch = 0.0
    try:
        gunning_fog = textstat.gunning_fog(text)
    except Exception:
        gunning_fog = 0.0

    # --- Named Entities (4) ---
    first_person_count = sum(1 for w in words_lower if w in FIRST_PERSON)
    direct_address_count = sum(1 for w in words_lower if w in DIRECT_ADDRESS)
    person_entities = sum(1 for e in doc.ents if e.label_ == "PERSON")
    date_entities = sum(1 for e in doc.ents if e.label_ == "DATE")

    # --- Uniqueness & Variety (3) ---
    def ngram_uniqueness(seq, n):
        grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
        if not grams:
            return 0.0
        return len(set(grams)) / len(grams)
    bigram_uniqueness = ngram_uniqueness(words_lower, 2)
    trigram_uniqueness = ngram_uniqueness(words_lower, 3)
    syntax_variety = len(set(t.pos_ for t in tokens_all))

    return {
        "WordCount": word_count, "UniqueWordCount": unique_word_count, "CharCount": char_count,
        "AvgWordLength": avg_word_length, "TTR": ttr, "HapaxLegomenonRate": hapax_rate,
        "SentenceCount": sentence_count, "AvgSentenceLength": avg_sentence_length,
        "PunctuationCount": punctuation_count, "StopWordCount": stopword_count,
        "AbstractNounCount": abstract_noun_count, "ComplexVerbCount": complex_verb_count,
        "SophisticatedAdjectiveCount": sophisticated_adj_count, "AdverbCount": adverb_count,
        "ComplexSentenceCount": complex_sentence_count, "QuestionCount": question_count,
        "ExclamationCount": exclamation_count, "ContractionCount": contraction_count,
        "EmotionWordCount": emotion_word_count, "Polarity": polarity, "Subjectivity": subjectivity,
        "VaderCompound": vader_compound, "FleschReadingEase": flesch, "GunningFogIndex": gunning_fog,
        "FirstPersonCount": first_person_count, "DirectAddressCount": direct_address_count,
        "PersonEntities": person_entities, "DateEntities": date_entities,
        "BigramUniqueness": bigram_uniqueness, "TrigramUniqueness": trigram_uniqueness,
        "SyntaxVariety": syntax_variety,
    }

FEATURE_COLS = [
    "WordCount", "UniqueWordCount", "CharCount", "AvgWordLength", "TTR", "HapaxLegomenonRate",
    "SentenceCount", "AvgSentenceLength", "PunctuationCount", "StopWordCount", "AbstractNounCount",
    "ComplexVerbCount", "SophisticatedAdjectiveCount", "AdverbCount", "ComplexSentenceCount",
    "QuestionCount", "ExclamationCount", "ContractionCount", "EmotionWordCount", "Polarity",
    "Subjectivity", "VaderCompound", "FleschReadingEase", "GunningFogIndex", "FirstPersonCount",
    "DirectAddressCount", "PersonEntities", "DateEntities", "BigramUniqueness", "TrigramUniqueness",
    "SyntaxVariety",
]
assert len(FEATURE_COLS) == 31, f"Expected 31 features, got {len(FEATURE_COLS)}"

def extract_split(dframe, desc):
    texts = dframe["text_processed"].tolist()
    docs = nlp.pipe(texts, batch_size=64)
    rows = []
    for text, doc in tqdm(zip(texts, docs), total=len(texts), desc=f"  Stylometric features [{desc}]"):
        rows.append(extract_stylometric_features(text, doc))
    feat_df = pd.DataFrame(rows)
    feat_df["label"] = dframe["label"].values
    feat_df["domain"] = dframe[DOMAIN_COLUMN].values
    return feat_df

print("\n  Extracting 31 stylometric features per document ...")
train_feat = extract_split(train_df, "train")
val_feat   = extract_split(val_df,   "val")
test_feat  = extract_split(test_df,  "test")

# ==============================================================================
# 6. PUBLICATION STYLE — same IMG_COLORS palette + style_ax/render_table_png
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
CLASSIFIER_COLORS = {   # Table-2 style classifier comparison (not Human/AI)
    "Random Forest":       "#4CAF50",
    "SVM":                 "#2196F3",
    "Logistic Regression": "#9C27B0",
    "Decision Tree":       "#FF9800",
    "KNN":                 "#F44336",
    "Gradient Boosting":   "#795548",
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
# 7. CLASSIFIERS — Random Forest (primary / "StyloAI") + 5 comparison models
# (Table 2 of the paper), default Scikit-Learn settings, exactly as the paper.
# ==============================================================================
print("\n" + "=" * 70)
print("  Training classifiers (default Scikit-Learn settings, as in the paper)")
print("=" * 70)

Xtr = train_feat[FEATURE_COLS].values; ytr = train_feat["label"].values
Xval = val_feat[FEATURE_COLS].values;  yval = val_feat["label"].values
Xte = test_feat[FEATURE_COLS].values;  yte = test_feat["label"].values

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xval_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xval), scaler.transform(Xte)

classifiers = {
    "Random Forest":       RandomForestClassifier(random_state=SEED),
    "SVM":                 SVC(probability=True, random_state=SEED),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "Decision Tree":       DecisionTreeClassifier(random_state=SEED),
    "KNN":                 KNeighborsClassifier(),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=SEED),
}

fitted = {}
for name, clf in classifiers.items():
    clf.fit(Xtr_s, ytr)
    fitted[name] = clf

# ==============================================================================
# 8. DECISION THRESHOLD — FIXED at 0.50 for the main StyloAI (Random Forest)
# model (per explicit request, 2026-10-24: "خلي العتبات بالكودات الجديدة
# 50" — use 0.50 in the new scripts instead of a validation-searched
# threshold). The F1-vs-threshold curve over the validation grid is still
# computed and plotted/saved below purely as a DIAGNOSTIC (so you can see
# how sensitive F1 is to the cutoff), but it no longer overrides the cutoff
# actually used for the reported test-set metrics — that is always 0.50.
# ==============================================================================
print("\n  Decision threshold fixed at 0.50 (Random Forest) — computing validation F1 curve for reference only ...")
rf = fitted["Random Forest"]
val_probs_ai = rf.predict_proba(Xval_s)[:, 1]

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
ax_th.axvline(best_threshold, color="black", ls="--", lw=1, label=f"best={best_threshold:.2f}")
style_ax(ax_th, xlabel="Decision threshold", ylabel="Validation F1 (Random Forest)",
         title="Threshold Search (Validation)")
ax_th.legend(fontsize=9)
plt.tight_layout()
fig_th.savefig(f"{PLOTS_DIR}/threshold_search.png"); plt.show()

# ==============================================================================
# 9. EVALUATION ON TEST SET — all 6 classifiers (Table-2 style comparison)
# ==============================================================================
print("\n" + "─" * 70)
print("  Evaluating on held-out test set ...")

results_table = []
roc_curves = {}
for name, clf in fitted.items():
    y_prob = clf.predict_proba(Xte_s)[:, 1]
    th_used = best_threshold if name == "Random Forest" else 0.5
    y_pred = (y_prob >= th_used).astype(int)

    acc = accuracy_score(yte, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(yte, y_pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(yte, y_pred)
    fpr, tpr, _ = roc_curve(yte, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_curves[name] = (fpr, tpr, roc_auc)

    results_table.append({"Model": name, "Threshold": round(th_used, 2),
                          "Precision": round(p, 4), "Recall": round(r, 4),
                          "F1-Score": round(f1, 4), "Accuracy": round(acc, 4),
                          "MCC": round(mcc, 4), "AUC": round(roc_auc, 4)})

    if name == "Random Forest":
        rf_y_pred, rf_y_prob = y_pred, y_prob

results_df = pd.DataFrame(results_table).sort_values("Accuracy", ascending=False)
print("\n  === Classifier comparison (Table-2 style) ===")
print(results_df.to_string(index=False))
results_df.to_excel(f"{TABLES_DIR}/classifier_comparison_metrics.xlsx", index=False)
render_table_png(results_df, f"{PLOTS_DIR}/table_classifier_comparison.png",
                 figsize=(13, 0.55 * len(results_df) + 1.2))

main_row = results_df[results_df["Model"] == "Random Forest"].iloc[0]
accuracy, f1, precision, recall, mcc, roc_auc = (
    main_row["Accuracy"], main_row["F1-Score"], main_row["Precision"],
    main_row["Recall"], main_row["MCC"], main_row["AUC"])

print("\n" + "=" * 55)
print("FINAL RESULTS ON TEST SET — StyloAI (Random Forest, main model)")
print("=" * 55)
print(f"  Threshold : {best_threshold:.2f}  (fixed, val F1 at this threshold={best_threshold_f1:.4f})")
print(f"  Accuracy  : {accuracy:.4f}   F1: {f1:.4f}   Precision: {precision:.4f}")
print(f"  Recall    : {recall:.4f}   MCC: {mcc:.4f}   AUC: {roc_auc:.4f}")
print("=" * 55)

# ==============================================================================
# 10. FEATURE IMPORTANCE (Figure 2 style)
# ==============================================================================
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
top10 = importances.head(10)
print("\n  Top 10 most important features (Random Forest):")
print(top10.to_string())
importances.to_csv(f"{TABLES_DIR}/feature_importances.csv")

# ==============================================================================
# 11. UNIFIED PERFORMANCE TABLE — Document-level Overall + per-domain, same
# structure as every other baseline script's unified table (Random Forest at
# its validation-searched threshold).
# ==============================================================================
test_df = test_df.reset_index(drop=True)
test_df["prob_ai"] = rf_y_prob
test_df["pred_label"] = rf_y_pred
test_df["true_label"] = yte

domain_rows = []
for dom in sorted(test_df[DOMAIN_COLUMN].unique()):
    mask = (test_df[DOMAIN_COLUMN] == dom).values
    sub_true, sub_pred, sub_prob = yte[mask], rf_y_pred[mask], rf_y_prob[mask]
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
# 12. PLOTS
# ==============================================================================
print("\n  Generating plots ...")
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(f"StyloAI — Stylometric Detection (Random Forest, 31 features)\n"
             f"F1={f1:.4f}  AUC={roc_auc:.4f}  threshold={best_threshold:.2f}",
             fontsize=12, fontweight="bold")

# (a) Confusion matrix
cm = confusion_matrix(yte, rf_y_pred)
tn, fp, fn, tp = cm.ravel()
_plot_cm(axes[0, 0], cm, "Confusion Matrix (Random Forest)", tn, fp, fn, tp)

# (b) ROC — all 6 classifiers
for name, (fpr, tpr, roc_auc_m) in roc_curves.items():
    axes[0, 1].plot(fpr, tpr, lw=2, color=CLASSIFIER_COLORS.get(name, "gray"),
                    label=f"{name} (AUC={roc_auc_m:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
style_ax(axes[0, 1], xlabel="False Positive Rate", ylabel="True Positive Rate",
         title="ROC — All Classifiers")
axes[0, 1].legend(fontsize=7)

# (c) P(AI) distribution — project's Human/AI colors
for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    axes[0, 2].hist(rf_y_prob[yte == lbl], bins=40, alpha=0.6, color=color, label=name, density=True)
axes[0, 2].axvline(best_threshold, color="black", lw=1.2, linestyle="--", label=f"threshold={best_threshold:.2f}")
style_ax(axes[0, 2], xlabel="P(AI)", ylabel="Density", title="P(AI) Distribution")
axes[0, 2].legend(fontsize=8)

# (d) Top-10 feature importances (Figure 2 style)
top10.sort_values().plot(kind="barh", ax=axes[1, 0], color="#7E57C2")
style_ax(axes[1, 0], xlabel="Importance", title="Top 10 Feature Importances (Fig. 2 style)")

# (e) Accuracy by classifier (Table-2 style)
acc_bar = results_df.set_index("Model")["Accuracy"].sort_values()
acc_bar.plot(kind="barh", ax=axes[1, 1],
             color=[CLASSIFIER_COLORS.get(m, "gray") for m in acc_bar.index])
style_ax(axes[1, 1], xlabel="Accuracy", title="Accuracy by Classifier (Table-2 style)")
axes[1, 1].set_xlim(0, 1.05)

# (f) TTR vs UniqueWordCount (top features), Human/AI colors
for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    sub = test_feat[test_feat["label"] == lbl]
    axes[1, 2].scatter(sub["TTR"], sub["UniqueWordCount"], s=10, alpha=0.4, color=color, label=name)
style_ax(axes[1, 2], xlabel="TTR", ylabel="UniqueWordCount", title="TTR vs. UniqueWordCount (top features)")
axes[1, 2].legend(fontsize=9)

plt.tight_layout()
plot_path = f"{PLOTS_DIR}/styloai_results_dashboard.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"  Dashboard saved -> {plot_path}")

# Standalone ROC curve (project convention: also saved on its own)
fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.8))
for name, (fpr, tpr, roc_auc_m) in roc_curves.items():
    ax_roc.plot(fpr, tpr, lw=2, color=CLASSIFIER_COLORS.get(name, "gray"),
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
# 13. SAVE
# ==============================================================================
print("\n" + "─" * 70)
print(f"  Saving to Drive -> {METHOD_DIR}")

with open(f"{METHOD_DIR}/styloai_random_forest.pkl", "wb") as f_out:
    pickle.dump({"scaler": scaler, "clf": fitted["Random Forest"],
                "feature_cols": FEATURE_COLS, "cfg": CFG,
                "threshold": best_threshold, "val_f1_at_threshold": best_threshold_f1}, f_out)
with open(f"{METHOD_DIR}/styloai_config.json", "w") as f_out:
    json.dump({"threshold": best_threshold, "val_f1": best_threshold_f1}, f_out, indent=2)

test_feat.to_csv(f"{TABLES_DIR}/test_stylometric_features.csv", index=False)
print(f"  Saved -> {METHOD_DIR}")

# ==============================================================================
# 14. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY — StyloAI (Stylometric Features + Random Forest)")
print("=" * 70)
print(f"  Features     : 31 stylometric features (6 categories)")
print(f"  Train docs   : {len(train_feat)}   Val docs: {len(val_feat)}   Test docs: {len(test_feat)}")
print(f"  Threshold    : {best_threshold:.2f}  (fixed at 0.50)")
print()
print(results_df.to_string(index=False))
print()
print(unified_df.to_string(index=False))
print(f"\n  Drive path   : {METHOD_DIR}")
print("=" * 70)