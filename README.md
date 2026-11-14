SupCon-Attn

Code and data for SupCon-Attn: Supervised Contrastive Sentence Encoding with Attention-Pooled Document Classification for AI-Generated Text Detection and Sentence-Level Interpretation.

The detector encodes each sentence with DeBERTa-v3-base trained with a supervised contrastive (SupCon) objective, pools the sentence vectors with a learned attention layer to classify the whole document, and explains the decision with signed per-sentence Integrated Gradients contributions.

Repository contents
Part	What it contains
Training	Script of the proposed method (three stages: SupCon sentence encoder, auxiliary sentence head, attention document head)
Baselines	Twelve baseline scripts: BERT, RoBERTa, ELECTRA, XLNet, DistilBERT, DeBERTa-v3 (document-level fine-tuning), GLTR, Fast-DetectGPT, Ghostbuster, Stylometric, Bi-LSTM + Attention, WaveDetect
Data	Train/validation/test splits of the KUFA dataset with the source_id column that groups all texts derived from the same source
Evaluation	Unseen-generator and attacked-text evaluation, per-seed results, and the sentence-level interpretation demo
Results	Per-seed results, selected thresholds and epochs
Data

KUFA is a balanced English corpus of 83,020 documents from seven domains. The source-grouped split contains 66,462 training, 8,275 validation, and 8,283 test documents. All documents that share a source_id (a human text and every AI text derived from it) are in the same partition.

Columns: text, label (0 = human, 1 = AI), domain, text type, generator, source_id.

Composition of the full corpus

Domain	Human	AI: title-based	AI: paraphrased	AI: humanized	AI: subtotal	Total	Avg. length (words)
Academic	26,996	9,333	8,722	8,823	26,878	53,874	307
News	2,421	807	806	807	2,420	4,841	381
Opinion	1,937	644	642	644	1,930	3,867	333
Answers	2,587	862	862	863	2,587	5,174	354
Reviews	2,624	874	874	876	2,624	5,248	219
Story	2,509	836	832	837	2,505	5,014	551
Wiki	2,501	833	833	835	2,501	5,002	147
Total	41,575	14,189	13,571	13,685	41,445	83,020	315

Test split per domain (documents): Academic 5,398 · Reviews 522 · Answers 511 · Wiki 496 · Story 485 · News 481 · Opinion 390.

Licensing: the Academic human texts come from PLOS research articles (open access). Texts from other public corpora are redistributed only where their licenses allow; otherwise identifiers and labels are provided with instructions to rebuild the text.

Installation
pip install -r requirements.txt

A GPU (NVIDIA A100 was used) is recommended. Text generation scripts need OPENAI_API_KEY (and the corresponding key for the other provider) as an environment variable; no key is stored in the repository.

How to reproduce
Place the split files in the folder expected by the scripts (see the paths at the top of each script).
Train the proposed method with the five seeds used in the paper: 42, 123, 2026, 3407, 7890. Each seed selects its epochs and thresholds on the validation split only; the test split is never used for selection.
Run the baseline scripts. Their per-seed outputs are aggregated into mean ± standard deviation tables.
Run the evaluation scripts on the unseen-generator and attacked sets, and the demo for the Integrated Gradients example.

Thresholds: selected by maximizing F1 on the validation split; document threshold grid 0.40–0.60 and sentence threshold grid 0.05–0.95 (step 0.01) for the proposed method.

Baseline configurations
Detector	Model / checkpoint	Configuration	Threshold (validation)
BERT	bert-base-cased	Document-level fine-tuning; classification head; max. length 512 tokens (longer documents truncated); batch 32; 5 epochs; AdamW, lr 2e-5, weight decay 0.01, 500 warm-up steps; bf16; class-weighted cross-entropy; final epoch kept	F1, grid 0.40–0.60
RoBERTa	roberta-base	Same as BERT	F1, grid 0.40–0.60
ELECTRA	google/electra-base-discriminator	Same as BERT	F1, grid 0.40–0.60
XLNet	xlnet-base-cased	Same as BERT	F1, grid 0.40–0.60
DistilBERT	distilbert-base-cased	Same as BERT	F1, grid 0.40–0.60
DeBERTa-v3	microsoft/deberta-v3-base	Same as BERT	F1, grid 0.40–0.60
GLTR	GPT-2 (124M)	Token-rank features per document; logistic regression	F1, grid 0.05–0.95
Fast-DetectGPT	GPT-2-XL (1.5B)	Conditional-probability curvature score; logistic regression	F1, grid 0.05–0.95
Ghostbuster	GPT-2 (weak), GPT-2-medium (strong)	Token-probability features; L1 logistic-regression feature selection; final L2 logistic regression (C = 1)	F1, grid 0.05–0.95
Stylometric	31 hand-crafted features	Standardized features; six classifiers with default scikit-learn settings; SVM reported	SVM: fixed 0.50 (random forest: grid 0.05–0.95)
Bi-LSTM + Attention	Part-of-speech tag sequences	Embedding 32; Conv1D (128 filters, width 3) with max pooling; Bi-LSTM (64 units); additive attention; dense head, dropout 0.3	F1, grid 0.05–0.95
WaveDetect	Qwen2.5-0.5B (surrogate LM)	Surrogate LM jointly fine-tuned with a wavelet convolutional classifier	F1, grid 0.05–0.95
Proposed method	microsoft/deberta-v3-base	Section 4.7 and Table 4	F1, grid 0.40–0.60
Evaluation sets

Unseen generators: 17,832 samples (3,000 academic human texts and 14,832 AI texts from Gemma 3 27B, Ministral 14B, Claude Sonnet 4.6, DeepSeek V3, and Llama 4 Scout 17B), each generated with the same three strategies (title-based generation, paraphrasing, humanization) from a separate split of the human-source pool.

Attacked evaluation set

Six attacks are applied to 1,000 AI texts each.

Category	Attack	N	Mechanism
Programmatic	Synonym Replacement	1,000	LLM-based (DeepSeek-V4.1-Flash), ≈40% of words replaced.
Programmatic	Typo Injection	1,000	Deterministic code, exactly 5% of eligible words typo'd; real-misspelling dictionary + generic fallback; concentrated in half of each document's sentences.
Programmatic	Human-AI Sentence Mix	1,000	50% of AI sentences replaced, position-matched, by the title-matched human document's sentences.
Programmatic	Back-Translation	1,000	Round-trip machine translation from English to Arabic then back to English.
External tool	Grammarly AI Humanizer	1,000	Manually processed through Grammarly's web-based humanization tool.
External tool	QuillBot AI Humanizer	1,000	Manually processed through QuillBot's web-based humanization tool.
Citation

If you use this repository, please cite the paper (reference to be added after acceptance).

License

See the LICENSE file.
