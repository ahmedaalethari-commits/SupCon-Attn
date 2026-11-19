# ==============================================================================
# WaveDetect — Machine-Generated Text Detection via Wavelet Transform
# (Liu, Qin, He & Xu, ACL Findings 2026, arXiv:2606.23336)
# New reimplementation, built directly for this project's data/normalization/
# workflow conventions (see chat, 2026-10-24).
#
# THIS VERSION MATCHES THE PAPER'S FULL "WaveDetect-ALL" CONFIGURATION, i.e.
# the surrogate LM is Qwen2.5-0.5B-Base and it is FINE-TUNED JOINTLY with
# the CNN via the paper's own two-stage procedure (Section 4):
#   Stage 1 - Warm-up : CNN only trainable, LM frozen,  lr=1e-3
#   Stage 2 - Joint    : ALL parameters trainable (LM + CNN), lr=1e-5
# The per-token probability signal -> Morlet CWT -> CNN pipeline is fully
# differentiable end-to-end, so gradients from the classification loss flow
# back into the LM during Stage 2, exactly as the paper describes.
#
# PRACTICAL ADAPTATIONS (compute/data-scale only — the METHOD itself, model
# choice, and training procedure now match the paper exactly):
#   1. RAID (paper's training set) is ~6M samples on 4xRTX 6000 Ada for
#      ~5.5 hours. This project's labeled dataset is a few thousand
#      documents, so epoch counts are driven by EarlyStopping on validation
#      loss (per stage) rather than a fixed schedule tuned for 6M samples —
#      the same practical substitution already used for every other
#      trained-from-scratch method in this project (e.g. Bi-LSTM+Attention).
#   2. max_len=256 (vs. a longer effective context in the paper) and a
#      smaller batch_size for Stage 2 specifically: joint fine-tuning
#      backprops through the full 0.5B-parameter LM for every token in the
#      sequence, which is far more memory-hungry than the frozen-LM (BASE)
#      version. Mixed precision (bf16/fp16 autocast) and LM gradient
#      checkpointing are enabled to make this feasible on a single Colab
#      GPU; both are standard tools for exactly this situation and don't
#      change the computed quantities, only memory/speed.
#   3. The paper's 9:1 HWT:MGT class-weight ratio for its loss is specific
#      to RAID's own label imbalance. We instead compute the positive-class
#      weight from THIS project's own train split (so it correctly reduces
#      to plain BCE if our data is balanced, and reproduces the paper's
#      "weight the loss by the real class imbalance" intent otherwise,
#      rather than blindly importing a ratio from a different dataset).
#   4. Morlet central frequency (omega_0) and the number/range of wavelet
#      scales are not given numerically in the paper; standard values are
#      used (omega_0=6, log-spaced scales), documented in CFG below.
#   5. The CWT itself (Section 3, Eq. 2-4) is implemented EXACTLY as
#      described: a fixed-weight 1-D convolution with a bank of Morlet
#      kernels at different scales. "Differentiable" in the paper refers to
#      gradients flowing THROUGH this fixed transform back into the LM —
#      the wavelet kernels themselves are not learned (here or in the
#      paper); right-side zero-padding of the probability signal before the
#      transform matches the paper's Eq. 4.
# ==============================================================================

import gc, os, re, random, unicodedata, pickle, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM
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
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
set_seed()

# ==============================================================================
# 1. CONFIG — same project layout/conventions as every other English baseline
# ==============================================================================
PROJECT_DIR = "/content/drive/MyDrive/english_ai_detector_project"
METHOD_DIR  = f"{PROJECT_DIR}/WaveDetect_Detector_English"
TABLES_DIR  = f"{METHOD_DIR}/tables"
PLOTS_DIR   = f"{METHOD_DIR}/plots"
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_SPLIT_FILE = f"{PROJECT_DIR}/splits/train_split.csv"
VAL_SPLIT_FILE   = f"{PROJECT_DIR}/splits/val_split.csv"
TEST_SPLIT_FILE  = f"{PROJECT_DIR}/splits/test_split.csv"
DOMAIN_COLUMN    = "domain"

CFG = {
    "surrogate_lm": "Qwen/Qwen2.5-0.5B",   # paper's exact surrogate LM (base, not -Instruct)
    "max_len":      256,                    # see adaptation note 2 (joint fine-tune memory)
    "n_scales":     32,
    "scale_min":    1.0,
    "scale_max":    32.0,
    "morlet_w0":    6.0,

    "warmup_batch_size": 32,
    "warmup_lr":         1e-3,
    "warmup_max_epochs": 6,
    "warmup_patience":   4,

    "joint_batch_size":  8,     # smaller than paper's 64 — full LM backprop, see note 2
    "joint_lr":          1e-5,
    "joint_max_epochs":  4,
    "joint_patience":    4,
    "grad_accum_steps":  4,     # effective joint batch size 8*4=32

    "dropout": 0.3,
    "use_amp": True,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
print("=" * 70)
print("  WaveDetect-ALL — Wavelet-Spectral Detection (joint LM+CNN fine-tune)")
print(f"  Surrogate LM: {CFG['surrogate_lm']}   Scales: {CFG['n_scales']}")
print("=" * 70)
print(f"  Device: {DEVICE}   AMP dtype: {AMP_DTYPE}")

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

# ==============================================================================
# 4. SURROGATE LM — Qwen2.5-0.5B-Base (trainable: frozen in Stage 1, jointly
# fine-tuned in Stage 2). Tokenized once up-front (input_ids/attention_mask
# are fixed; what changes across stages is which parameters receive
# gradients during the forward pass that uses them).
# ==============================================================================
print("\n" + "─" * 70)
print("  Loading Qwen2.5-0.5B-Base (surrogate LM) ...")

tokenizer = AutoTokenizer.from_pretrained(CFG["surrogate_lm"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lm_model = AutoModelForCausalLM.from_pretrained(
    CFG["surrogate_lm"], torch_dtype=torch.float32
).to(DEVICE)
lm_model.gradient_checkpointing_enable()   # memory: see adaptation note 2

def tokenize_split(dframe, desc):
    enc = tokenizer(
        dframe["text_processed"].tolist(),
        truncation=True, padding="max_length",
        max_length=CFG["max_len"] + 1, return_tensors="pt",
    )
    print(f"  Tokenized [{desc}]: input_ids {enc['input_ids'].shape}")
    return enc["input_ids"], enc["attention_mask"]

print("\n  Tokenizing all splits ...")
train_ids, train_mask = tokenize_split(train_df, "train")
val_ids,   val_mask   = tokenize_split(val_df,   "val")
test_ids,  test_mask  = tokenize_split(test_df,  "test")

y_train = train_df["label"].values.astype(np.float32)
y_val   = val_df["label"].values.astype(np.float32)
y_test  = test_df["label"].values.astype(np.float32)

# ==============================================================================
# 5. CONTINUOUS WAVELET TRANSFORM — fixed-weight Morlet filter bank
# (Section 3, Eq. 2-4). Implemented with plain torch ops (conv1d), so
# gradients flow through it into the LM during Stage 2 without any extra
# plumbing — the filters themselves are buffers, not learned parameters.
# ==============================================================================
print("\n" + "─" * 70)
print("  Building the fixed Morlet wavelet filter bank ...")

def build_morlet_filters(n_scales, scale_min, scale_max, w0, device):
    scales = np.geomspace(scale_min, scale_max, n_scales)
    kernel_len = int(10 * scale_max)
    if kernel_len % 2 == 0:
        kernel_len += 1
    half = kernel_len // 2
    t = np.arange(-half, half + 1)
    real_f = np.zeros((n_scales, kernel_len), dtype=np.float32)
    imag_f = np.zeros((n_scales, kernel_len), dtype=np.float32)
    for i, s in enumerate(scales):
        norm = (np.pi ** -0.25) / np.sqrt(s)
        arg = t / s
        envelope = np.exp(-0.5 * arg ** 2)
        real_f[i] = norm * np.cos(w0 * arg) * envelope
        imag_f[i] = norm * np.sin(w0 * arg) * envelope
    real_t = torch.tensor(real_f, device=device).unsqueeze(1)  # (K, 1, L)
    imag_t = torch.tensor(imag_f, device=device).unsqueeze(1)
    return real_t, imag_t, kernel_len

MORLET_REAL, MORLET_IMAG, MORLET_KLEN = build_morlet_filters(
    CFG["n_scales"], CFG["scale_min"], CFG["scale_max"], CFG["morlet_w0"], DEVICE)
print(f"  Filter bank: {CFG['n_scales']} scales, kernel length {MORLET_KLEN}")

def cwt_transform(signal_batch):
    """signal_batch: (B, N) float tensor -> |W(s,tau)|: (B, K, N). Fully
    differentiable (plain conv1d with fixed, non-learned filters)."""
    x = signal_batch.unsqueeze(1)                              # (B,1,N)
    pad = MORLET_KLEN // 2
    real_resp = F.conv1d(x, MORLET_REAL.to(x.dtype), padding=pad)
    imag_resp = F.conv1d(x, MORLET_IMAG.to(x.dtype), padding=pad)
    mag = torch.sqrt(real_resp ** 2 + imag_resp ** 2 + 1e-12)
    mag = mag[:, :, :signal_batch.shape[-1]]                    # (B,K,N)
    return mag

# ==============================================================================
# 6. CNN CLASSIFIER — compact CNN over the (K scales x N tokens) spectrogram.
# ==============================================================================
class SpectralCNN(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)   # raw logit

cnn_model = SpectralCNN(CFG["dropout"]).to(DEVICE)
n_cnn_params = sum(p.numel() for p in cnn_model.parameters())
n_lm_params = sum(p.numel() for p in lm_model.parameters())
print(f"\n  CNN parameters: {n_cnn_params/1e3:.1f}K   LM parameters: {n_lm_params/1e6:.1f}M")

# ==============================================================================
# 7. END-TO-END DIFFERENTIABLE FORWARD PASS
# probability signal (Eq. 1, via the LM) -> CWT (Eq. 2-4) -> CNN -> logit
# ==============================================================================
def forward_pipeline(input_ids, attention_mask, lm_grad_enabled):
    """input_ids/attention_mask: (B, max_len+1). Returns (logit, spec)."""
    ids_in     = input_ids[:, :-1]
    mask_in    = attention_mask[:, :-1]
    ids_target = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:].to(torch.float32)   # 1=real token, 0=padding

    ctx = torch.enable_grad() if lm_grad_enabled else torch.no_grad()
    with ctx:
        out = lm_model(ids_in, attention_mask=mask_in)
        logits = out.logits.float()
    probs_full = F.softmax(logits, dim=-1)
    token_probs = probs_full.gather(-1, ids_target.unsqueeze(-1)).squeeze(-1)  # (B, max_len)
    token_probs = token_probs * target_mask   # right-zero-pad real padding positions (Eq. 4)

    spec = cwt_transform(token_probs)                       # (B, K, N)
    flat_min = spec.amin(dim=(1, 2), keepdim=True)
    flat_max = spec.amax(dim=(1, 2), keepdim=True)
    spec_norm = (spec - flat_min) / (flat_max - flat_min + 1e-8)
    logit = cnn_model(spec_norm.unsqueeze(1))
    return logit, spec_norm

# ==============================================================================
# 8. CLASS-WEIGHTED LOSS — weight derived from THIS project's own train
# split (see adaptation note 3), not RAID's 9:1 ratio.
# ==============================================================================
n_pos = float((y_train == 1).sum())
n_neg = float((y_train == 0).sum())
pos_weight_value = (n_neg / n_pos) if n_pos > 0 else 1.0
pos_weight = torch.tensor(pos_weight_value, device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"\n  Class balance (train): {int(n_neg)} human / {int(n_pos)} AI  ->  pos_weight={pos_weight_value:.3f}")

# ==============================================================================
# 9. TRAINING LOOP HELPERS
# ==============================================================================
def iterate_batches(n, batch_size, shuffle):
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]

def run_eval(ids, mask, labels, batch_size):
    cnn_model.eval(); lm_model.eval()
    all_logits = []
    with torch.no_grad():
        for b in iterate_batches(len(labels), batch_size, shuffle=False):
            bi = ids[b].to(DEVICE); bm = mask[b].to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(CFG["use_amp"] and DEVICE == "cuda")):
                logit, _ = forward_pipeline(bi, bm, lm_grad_enabled=False)
            all_logits.append(logit.float().cpu())
    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    loss = F.binary_cross_entropy(torch.tensor(probs), torch.tensor(labels), reduction="mean").item()
    acc = accuracy_score(labels, (probs >= 0.5).astype(int))
    return loss, acc, probs

def train_one_stage(stage_name, ids, mask, labels, optimizer, batch_size, max_epochs, patience,
                    lm_trainable, grad_accum_steps=1):
    print(f"\n{'─'*70}\n  STAGE: {stage_name}  (LM trainable: {lm_trainable})\n{'─'*70}")
    lm_model.requires_grad_(lm_trainable)
    cnn_model.requires_grad_(True)

    scaler = torch.cuda.amp.GradScaler(enabled=(CFG["use_amp"] and DEVICE == "cuda" and AMP_DTYPE == torch.float16))
    best_val_loss, best_cnn_state, best_lm_state, no_improve = float("inf"), None, None, 0
    stage_history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    for epoch in range(1, max_epochs + 1):
        cnn_model.train(); lm_model.train(lm_trainable)
        running_loss, all_logits_tr, all_labels_tr = 0.0, [], []
        optimizer.zero_grad()

        batches = list(iterate_batches(len(labels), batch_size, shuffle=True))
        for step, b in enumerate(tqdm(batches, desc=f"  [{stage_name}] epoch {epoch}")):
            bi = ids[b].to(DEVICE); bm = mask[b].to(DEVICE)
            by = torch.tensor(labels[b], dtype=torch.float32, device=DEVICE)

            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(CFG["use_amp"] and DEVICE == "cuda")):
                logit, _ = forward_pipeline(bi, bm, lm_grad_enabled=lm_trainable)
                loss = criterion(logit, by) / grad_accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(batches):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum_steps * len(b)
            all_logits_tr.append(logit.detach().float().cpu())
            all_labels_tr.append(by.detach().cpu())

        tr_loss = running_loss / len(labels)
        tr_probs = torch.sigmoid(torch.cat(all_logits_tr)).numpy()
        tr_labels_np = torch.cat(all_labels_tr).numpy()
        tr_acc = accuracy_score(tr_labels_np, (tr_probs >= 0.5).astype(int))

        val_loss, val_acc, _ = run_eval(val_ids, val_mask, y_val, CFG["warmup_batch_size"])
        stage_history["loss"].append(tr_loss); stage_history["accuracy"].append(tr_acc)
        stage_history["val_loss"].append(val_loss); stage_history["val_accuracy"].append(val_acc)
        print(f"  [{stage_name}] epoch {epoch}/{max_epochs}  loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_cnn_state = {k: v.clone() for k, v in cnn_model.state_dict().items()}
            if lm_trainable:
                best_lm_state = {k: v.clone() for k, v in lm_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping [{stage_name}] at epoch {epoch}")
                break

    if best_cnn_state is not None:
        cnn_model.load_state_dict(best_cnn_state)
    if lm_trainable and best_lm_state is not None:
        lm_model.load_state_dict(best_lm_state)
    print(f"  [{stage_name}] restored best weights (val_loss={best_val_loss:.4f})")
    return stage_history

# ==============================================================================
# 10. STAGE 1 — WARM-UP (CNN only, LM frozen, lr=1e-3)
# ==============================================================================
lm_model.requires_grad_(False)
warmup_optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=CFG["warmup_lr"])
warmup_history = train_one_stage(
    "Warm-up", train_ids, train_mask, y_train, warmup_optimizer,
    CFG["warmup_batch_size"], CFG["warmup_max_epochs"], CFG["warmup_patience"],
    lm_trainable=False,
)

# ==============================================================================
# 11. STAGE 2 — JOINT FINE-TUNE (LM + CNN, lr=1e-5)
# ==============================================================================
joint_optimizer = torch.optim.AdamW(
    list(lm_model.parameters()) + list(cnn_model.parameters()), lr=CFG["joint_lr"])
joint_history = train_one_stage(
    "Joint fine-tune", train_ids, train_mask, y_train, joint_optimizer,
    CFG["joint_batch_size"], CFG["joint_max_epochs"], CFG["joint_patience"],
    lm_trainable=True, grad_accum_steps=CFG["grad_accum_steps"],
)

# combined history for plotting (warm-up epochs followed by joint epochs)
history = {k: warmup_history[k] + joint_history[k] for k in warmup_history}
stage_boundary = len(warmup_history["loss"])
total_params = n_cnn_params + n_lm_params

# ==============================================================================
# 12. PUBLICATION STYLE — same IMG_COLORS palette + style_ax/render_table_png
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
# 13. DECISION THRESHOLD — FIXED at 0.50 (per "خلي العتبات بالكودات الجديدة
# 50"). The F1-vs-threshold curve over a validation grid is still computed
# purely as a DIAGNOSTIC, but the reported test-set metrics always use 0.50.
# ==============================================================================
_, _, val_probs = run_eval(val_ids, val_mask, y_val, CFG["warmup_batch_size"])
th_grid = np.arange(0.05, 0.96, 0.01)
threshold_rows = [{"threshold": round(float(th), 2),
                   "val_f1": round(f1_score(y_val, (val_probs >= th).astype(int), zero_division=0), 4)}
                  for th in th_grid]
threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_excel(f"{TABLES_DIR}/threshold_search.xlsx", index=False)
best_threshold = 0.5
best_threshold_f1 = f1_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0)
print(f"\n  Using FIXED threshold: {best_threshold:.2f}  (val F1 at this threshold={best_threshold_f1:.4f})")

fig_th, ax_th = plt.subplots(figsize=(5.5, 4.2))
ax_th.plot(threshold_df["threshold"], threshold_df["val_f1"], color=IMG_COLORS["val_f1"], lw=2)
ax_th.axvline(best_threshold, color="black", ls="--", lw=1, label=f"used={best_threshold:.2f}")
style_ax(ax_th, xlabel="Decision threshold", ylabel="Validation F1", title="Threshold Diagnostic (Validation)")
ax_th.legend(fontsize=9)
plt.tight_layout()
fig_th.savefig(f"{PLOTS_DIR}/threshold_search.png"); plt.show()

# ==============================================================================
# 14. EVALUATION ON TEST SET
# ==============================================================================
print("\n" + "─" * 70)
print("  Evaluating on held-out test set ...")

_, _, y_prob = run_eval(test_ids, test_mask, y_test, CFG["warmup_batch_size"])
y_true = y_test
y_pred = (y_prob >= best_threshold).astype(int)

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

metrics_df = pd.DataFrame([{
    "Method": "WaveDetect-ALL (Qwen2.5-0.5B, joint fine-tune)", "Threshold": round(best_threshold, 2),
    "Accuracy": round(accuracy, 4), "F1": round(f1, 4), "Precision": round(precision, 4),
    "Recall": round(recall, 4), "MCC": round(mcc, 4), "AUC": round(roc_auc, 4),
}])
print("\n  === Test metrics ===")
print(metrics_df.to_string(index=False))
metrics_df.to_excel(f"{TABLES_DIR}/test_metrics.xlsx", index=False)
render_table_png(metrics_df, f"{PLOTS_DIR}/table_test_metrics.png", figsize=(13, 2.0))

# ==============================================================================
# 15. UNIFIED PERFORMANCE TABLE — Document-level Overall + per-domain, same
# structure as every other baseline script's unified table.
# ==============================================================================
test_df = test_df.reset_index(drop=True)
test_df["prob_ai"] = y_prob
test_df["pred_label"] = y_pred
test_df["true_label"] = y_true

domain_rows = []
for dom in sorted(test_df[DOMAIN_COLUMN].unique()):
    mask_dom = (test_df[DOMAIN_COLUMN] == dom).values
    sub_true, sub_pred, sub_prob = y_true[mask_dom], y_pred[mask_dom], y_prob[mask_dom]
    dom_fpr, dom_tpr, _ = roc_curve(sub_true, sub_prob)
    domain_rows.append({
        "Level": f"Document — {dom.capitalize()}", "N": int(mask_dom.sum()),
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
# 16. PLOTS
# ==============================================================================
print("\n  Generating plots ...")
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(f"WaveDetect-ALL — Qwen2.5-0.5B + Wavelet CNN (joint fine-tune)\n"
             f"F1={f1:.4f}  AUC={roc_auc:.4f}  threshold={best_threshold:.2f}  Params={total_params/1e6:.1f}M",
             fontsize=12, fontweight="bold")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
_plot_cm(axes[0, 0], cm, "Confusion Matrix", tn, fp, fn, tp)

axes[0, 1].plot(fpr, tpr, color=IMG_COLORS["roc_line"], lw=2, label=f"WaveDetect (AUC={roc_auc:.4f})")
axes[0, 1].plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
style_ax(axes[0, 1], xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
axes[0, 1].legend(fontsize=9)

for lbl, color, name in [(0, IMG_COLORS["human"], "Human"), (1, IMG_COLORS["ai"], "AI")]:
    axes[0, 2].hist(y_prob[y_true == lbl], bins=40, alpha=0.6, color=color, label=name, density=True)
axes[0, 2].axvline(best_threshold, color="black", lw=1.2, linestyle="--", label=f"threshold={best_threshold:.2f}")
style_ax(axes[0, 2], xlabel="P(AI)", ylabel="Density", title="P(AI) Distribution")
axes[0, 2].legend(fontsize=8)

ep = range(1, len(history["loss"]) + 1)
axes[1, 0].plot(ep, history["loss"], color=IMG_COLORS["human"], lw=2, label="Train Loss")
axes[1, 0].plot(ep, history["val_loss"], color=IMG_COLORS["ai"], lw=2, marker="o", label="Val Loss")
axes[1, 0].axvline(stage_boundary + 0.5, color="black", ls=":", lw=1, label="Warm-up -> Joint")
style_ax(axes[1, 0], xlabel="Epoch", title="Training Loss (both stages)")
axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(ep, history["accuracy"], color=IMG_COLORS["val_f1"], lw=2, label="Train Acc")
axes[1, 1].plot(ep, history["val_accuracy"], color=IMG_COLORS["val_acc"], lw=2, marker="s", label="Val Acc")
axes[1, 1].axvline(stage_boundary + 0.5, color="black", ls=":", lw=1)
style_ax(axes[1, 1], xlabel="Epoch", title="Accuracy (both stages)")
axes[1, 1].set_ylim(0, 1.05); axes[1, 1].legend(fontsize=8)

mean_human = test_df.loc[test_df["true_label"] == 0]
mean_ai = test_df.loc[test_df["true_label"] == 1]
# recompute a few example spectrograms for the qualitative panel
with torch.no_grad():
    bi = test_ids[:min(64, len(test_df))].to(DEVICE)
    bm = test_mask[:min(64, len(test_df))].to(DEVICE)
    _, spec_sample = forward_pipeline(bi, bm, lm_grad_enabled=False)
    spec_sample = spec_sample.cpu().numpy()
labels_sample = y_true[:min(64, len(test_df))]
if (labels_sample == 0).any() and (labels_sample == 1).any():
    diff = spec_sample[labels_sample == 1].mean(axis=0) - spec_sample[labels_sample == 0].mean(axis=0)
    im = axes[1, 2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                           vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    style_ax(axes[1, 2], xlabel="Time (token position)", ylabel="Wavelet scale index",
             title="Mean Spectrogram: AI − Human (sample)")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
else:
    axes[1, 2].axis("off")

plt.tight_layout()
plot_path = f"{PLOTS_DIR}/wavedetect_results_dashboard.png"
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

# ==============================================================================
# 17. SAVE
# ==============================================================================
print("\n" + "─" * 70)
print(f"  Saving to Drive -> {METHOD_DIR}")

torch.save(cnn_model.state_dict(), f"{METHOD_DIR}/wavedetect_cnn.pt")
lm_model.save_pretrained(f"{METHOD_DIR}/wavedetect_lm_finetuned")
tokenizer.save_pretrained(f"{METHOD_DIR}/wavedetect_lm_finetuned")

with open(f"{METHOD_DIR}/wavedetect_config.json", "w") as f_out:
    json.dump({"threshold": best_threshold, "val_f1": best_threshold_f1,
               "cfg": {k: v for k, v in CFG.items()},
               "n_cnn_params": int(n_cnn_params), "n_lm_params": int(n_lm_params)},
              f_out, indent=2)

pd.DataFrame(history).to_csv(f"{TABLES_DIR}/training_history.csv", index=False)
print(f"  Saved -> {METHOD_DIR}")

# ==============================================================================
# 18. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY — WaveDetect-ALL (Qwen2.5-0.5B, joint fine-tune)")
print("=" * 70)
print(f"  Surrogate LM : {CFG['surrogate_lm']} (jointly fine-tuned, Stage 2)")
print(f"  CNN params   : {n_cnn_params/1e3:.1f}K   LM params: {n_lm_params/1e6:.1f}M")
print(f"  Threshold    : {best_threshold:.2f}  (fixed)")
print(f"  Train docs   : {len(train_df)}   Val docs: {len(val_df)}   Test docs: {len(test_df)}")
print()
print(metrics_df.to_string(index=False))
print()
print(unified_df.to_string(index=False))
print(f"\n  Drive path   : {METHOD_DIR}")
print("=" * 70)