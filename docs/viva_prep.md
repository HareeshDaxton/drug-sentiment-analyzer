# Viva preparation — Drug Sentiment Analysis

Everything the project decided, why, and the answer to give when asked. Numbers marked *(report)* come from
`artifacts/reports/`, so check them against the latest run before the viva.

---

## 1. The project in 60 seconds

> Each row is a patient comment plus one drug name, and the label is the sentiment **toward that drug** — 0
> positive, 1 negative, 2 neutral, scored with weighted F1. A comment can discuss several drugs with different
> opinions, so I do not classify the text: I cut it down to the sentences that mention the drug the row asks about,
> mask the drug names (`targetdrug` / `otherdrug`), and add features about how the comment talks about that drug.
> I screened ten classical models on identical folds, tuned the best three with Optuna and shipped a tuned
> **LinearSVC** on TF-IDF + handcrafted + VADER + MiniLM features. Everything runs on a laptop CPU in minutes, and
> `main.py all` rebuilds the submission from the raw CSVs.

---

## 2. The data, in facts I can quote

| Fact | Value | Why it mattered |
|---|---|---|
| Records | train **5,279**, test **2,924** | the brief's 5,619 / 3,107 are *physical line counts* — comments contain line breaks inside quoted CSV fields |
| Classes | neutral 72.5%, negative 15.9%, positive 11.7% | imbalance; the majority-class baseline is weighted F1 **0.609** |
| Missing / duplicate / conflicting | none | no row cleaning was needed |
| Drugs | 102 in train, 95 in test, 9 unseen | fuzzy alias matching so misspellings still resolve |
| Drug named in the text | 99.8% of rows; 34% mention it more than once | mention-based windows are possible at all |
| Comment length | median 807 chars, p95 8.3k, max 128k | long documents dilute the signal |
| First mention | median word 41, but after word 380 in 6.9% of rows | head truncation would lose the drug entirely |
| Shared comments | 88 comments across 186 rows | group the CV folds by comment text |

---

## 3. Every design decision, and its defence

### 3.1 Drug-centred window instead of the whole comment
**What:** find the mentions of the target drug and keep exactly those sentences (`ctx_k0`), capped at 250 words.
**Why:** the label belongs to the (comment, drug) pair. The rest of a long comment often discusses other drugs,
dosing tables or forum chatter, which is noise for this row and, worse, is *shared* with rows that carry a different
label.
**Evidence:** `artifacts/reports/ablation_scores.csv` re-runs the whole cross-validation with the drug-blind full
comment, the masked full comment, and windows k=0/1/2; `artifacts/reports/window_c_search.csv` then re-tunes `C` for
each window so none is judged on another window's regularisation. The tightest window wins both:

| Text the model reads | Weighted F1 | Macro F1 |
|---|---|---|
| whole comment, drug-blind | 0.724 | 0.567 |
| whole comment, drugs masked | 0.723 | 0.564 |
| window k=1 (±1 sentence) | 0.726 | 0.569 |
| window k=2 (±2 sentences) | 0.725 | 0.568 |
| **window k=0 (mention sentences only)** | **0.735** | **0.586** |

**The honest story to tell:** the experiments notebook screened and tuned all ten models on the k=1 window, and
LinearSVC won there. The evaluation stage then measured the window itself, the tighter one was better, and the
project changed to it — with `C` re-tuned so the comparison was fair. That is a decision made by measurement after
the fact, not a lucky guess, and it is worth saying so.
**Why the tighter window wins:** neighbouring sentences in these comments usually talk about something else — a
different drug, a dose schedule, a date — so they add noise faster than signal. It is also the best evidence that
the drug conditioning is doing the work: the more the text is narrowed to the target drug, the better the score.

### 3.2 Masking the drug names
**What:** the row's drug becomes `targetdrug`, any other known drug becomes `otherdrug`.
**Why:** two reasons. (1) It tells the model *which* drug the question is about — the window may mention three. (2)
It stops the model memorising drug names as sentiment shortcuts, which would break on the 9 drugs that appear only
in test. The drug identity is still available, as a one-hot feature, so genuine drug priors can still be learned.

### 3.3 The drug lexicon (synonyms + fuzzy matching)
**What:** the 102 training drug names, plus a hand-written `config/drug_synonyms.yaml` of brand/generic pairs, plus
rapidfuzz matching (ratio ≥ 80, same first letter, length difference ≤ 2).
**Why:** patients write `stellara` for Stelara and `osmertinib` for Osimertinib. Exact matching would leave those
rows with no window at all.
**Care taken:** biosimilars are *not* merged with their reference product, because they are different products with
their own reception.

### 3.4 Features
| Block | Why it is there |
|---|---|
| TF-IDF word 1–2-grams | the sentiment-bearing words and short phrases ("not worth", "side effects") |
| TF-IDF character 2–5-grams (`char_wb`) | robust to the misspellings in patient writing; also captures drug-name fragments |
| Handcrafted counts | mention count and position, number of other drugs, lengths, `!`/`?`, negations, first-person words — how the comment talks about *this* drug |
| VADER | a lexicon sentiment score that needs no training: on the window, and as the mean sentence score of the whole comment |
| MiniLM (384-d, frozen) | meaning that bag-of-words misses ("didn't help" vs "was useless") |

**What each block is worth** *(report: `ablation_scores.csv`)*: TF-IDF alone 0.710; + handcrafted and VADER 0.732;
+ drug one-hot 0.725; all of it 0.735. MiniLM is worth about +0.011. The drug one-hot is the one block that does not
pay for itself — dropping it scores 0.737, which is inside the fold spread — and I kept it only because it is the
only place a per-drug prior can be learned and the folds are steadier with it. Say that plainly if asked; claiming
every feature helped would be easy to disprove from my own table.

**Stop words:** sklearn's English list **minus the negations** (`not`, `no`, `never`, `nor`, `n't`). Removing
negations would flip the meaning of exactly the sentences that matter.
**No lemmatisation:** with character n-grams already handling morphology, it added nothing and cost time.

### 3.5 Handling the 72% neutral class
`class_weight="balanced"` on the estimator (and balanced sample weights for the boosters, which have no such
option). It reweights the loss so the small classes are not simply given away. Without it the weighted F1 falls from
0.735 to 0.702 and the macro F1 collapses from 0.586 to 0.472 — the model simply answers neutral more often. It is
the single most valuable decision in the ablation table.
**Why not oversampling (SMOTE)?** On text, SMOTE interpolates between sparse TF-IDF vectors and invents documents
that no patient wrote. Class weights achieve the same aim without inventing data.

### 3.6 Cross-validation: StratifiedGroupKFold(5), grouped by comment text
**Stratified** keeps the class mix in every fold. **Grouped** keeps a comment that appears with several drugs inside
one fold. Without the grouping, a model can memorise a comment in training and be scored on it in validation — the
scores rise and mean nothing. The fold id is stored in the preprocessed pickle, so every model in the project is
scored on exactly the same splits.

### 3.7 Leakage discipline
- The preprocessed pickles hold **row-wise** values only: cleaning, windows, masks, counts, VADER, and frozen
  embeddings. Nothing there is fitted on the data, so they can be reused by any fold.
- Everything **fitted** — TF-IDF vocabulary, SVD, scalers, one-hot encoder — lives inside the sklearn `Pipeline`, so
  `fit` sees the training fold only.
- `test.csv` is read exactly once, by the `predict` stage.

### 3.8 Model choice
Screened 10 models with base settings on **half** the training data (same folds), took the top 3 by validation
weighted F1, tuned them with Optuna on the **full** data, and picked the winner. Screening on half the data is a
deliberate time/accuracy trade: it is enough to rank families, and it kept the whole screening under ten minutes.

Screening result on half the data, k=1 window *(report: `model_screening_half.csv`)*: LinearSVC 0.721,
XGBoost 0.709, LogisticRegression 0.702, GradientBoosting 0.702, SVC 0.700, LightGBM 0.700, ComplementNB 0.679,
KNN 0.659, RandomForest 0.643, DecisionTree 0.574. Tuned on the full data: LinearSVC 0.727, XGBoost 0.715,
LogisticRegression 0.709.

**Why LinearSVC wins:** ~5k short documents with tens of thousands of sparse features is the regime linear models
own — a margin-based linear classifier generalises where trees overfit. The boosters reached ~1.00 F1 on their
training folds and gained nothing on unseen rows.
**Why not the RBF SVC:** it needs the dense SVD representation, which throws away the exact-word evidence, and it is
an order of magnitude slower.
**Why not ComplementNB, even though it is built for imbalanced text:** its independence assumption is too strong
here; it was 4 points behind.

### 3.9 Hyperparameter tuning
Optuna's **TPE** sampler (seeded) with a **median pruner**: a trial that is below the median of earlier trials after
a fold is stopped instead of finishing all five. Budget: 30 trials or 10 minutes per model. The base settings are
scored on the same folds first, and kept if no trial beats them. For LinearSVC the only real knob is `C` (the
regularisation strength: smaller = simpler model, larger = fits training data harder).

---

## 4. Concepts I should be able to explain on the spot

**TF-IDF.** Term frequency × inverse document frequency: a word counts for more in a document when it is frequent
there and rare across the corpus. `sublinear_tf=True` uses 1+log(tf), so a word repeated 20 times does not count 20
times.

**Character n-grams (`char_wb`).** N-grams taken inside word boundaries. `humira` and `humaira` share most of their
5-grams, so a misspelling still matches — which matters for patient writing.

**Weighted vs macro F1.** Weighted F1 averages the per-class F1 weighted by class support: the 72% neutral class
dominates it. Macro F1 gives each class equal weight, so it exposes a model that ignores the small classes. The
capstone is scored on weighted F1; I watch macro F1 so I do not win it by predicting neutral everywhere.

**Why not accuracy.** Predicting neutral for everything gives 72.5% accuracy and a useless model.

**LinearSVC.** A linear support vector machine trained with the squared hinge loss and L2 regularisation, one-vs-rest
for three classes. It finds the hyperplane that separates the classes with the largest margin; `C` trades margin
width against training errors.

**Why LinearSVC has no `predict_proba`.** An SVM produces signed distances to the hyperplane, not probabilities.
Where scores are needed (ROC AUC), I convert the three decision values with a softmax, which preserves their
ranking. For genuine probabilities I would wrap it in `CalibratedClassifierCV` (Platt scaling) — not needed here,
because the metric only uses the predicted label.

**ROC AUC one-vs-rest.** Each class against the rest, averaged with class weights. It measures ranking quality,
independent of the decision threshold.

**StratifiedGroupKFold.** Folds that preserve the class distribution *and* keep every group (here: a comment) whole
inside one fold.

**Data leakage.** Any information from validation or test reaching training. Fitting a TF-IDF vocabulary on all rows
before splitting is the classic case; so is a comment appearing in both sides of the split.

**MiniLM / sentence embeddings.** A 6-layer transformer distilled from larger models, trained so that sentences with
similar meaning get similar vectors. I use it frozen — no fine-tuning — so each row's embedding depends on that row
alone, which is why it can live in the cached pickle without leaking.

**Why not fine-tune BERT?** No GPU: 7.3 GB RAM, integrated graphics. Fine-tuning a transformer on 5k long documents
on this CPU would take hours per epoch. That is the honest reason, and it is also the first item on the next-steps
list.

**VADER.** A rule and lexicon based sentiment scorer built for social text; it understands negation, intensifiers and
punctuation. It gives a useful prior without any training, and it is a feature, not the model.

**TruncatedSVD.** PCA for sparse matrices: it compresses 38k TF-IDF columns to 300 dense components so trees, KNN and
the RBF SVC can work with them at all.

**Optuna TPE.** Tree-structured Parzen Estimator: it models which hyperparameter regions produced good and bad
scores and samples where the good/bad ratio is highest — smarter than grid or random search at the same budget.

---

## 5. Questions I expect, with short answers

**Q: Why is this not just sentiment analysis?**
Because the target is a pair. 186 training rows share their comment with another drug, and the same text can be
positive about one drug and negative about another. A text classifier cannot express that; a drug-conditioned one
can.

**Q: How do you know the window helps?**
The ablation table: the same model and folds, with the whole comment (drug-blind) instead of the window, scores
lower. `artifacts/reports/ablation_scores.csv` has the exact deltas and `eval_04_ablations.png` shows them.

**Q: What if the drug is not mentioned in the text?**
0.2% of rows. The extractor falls back to the whole comment, `match_type` records `none`, and the model can use
that flag. The per-slice report shows how those rows score.

**Q: How do you handle drugs in test that were never in train?**
The name is masked to `targetdrug` anyway, so the text features are unaffected; the one-hot encoder maps unseen
drugs to an "infrequent" column instead of failing; and fuzzy matching resolves most of them to a known name.

**Q: Why 5 folds, not 10?**
5 folds keep ~1,050 validation rows per fold — enough for a stable weighted F1 — at half the compute of 10. With
the grouping constraint, more folds also make the group balance harder.

**Q: Is the tuned score optimistic?**
Yes, mildly: the top 3 were tuned on the same folds used to pick the winner, so the selection can chase fold noise.
It is fine for *choosing between models*, and it is why I report the fold standard deviation next to every score. A
nested CV would remove the bias at roughly five times the cost.

**Q: What would you do with more compute?**
Fine-tune a biomedical transformer (BioBERT/PubMedBERT) on the masked windows; that is where the remaining headroom
is. Then blend it with the linear model, whose errors are different.

**Q: What are the model's weaknesses?**
Long comments, rows where the drug is mentioned once in passing, and sarcasm or mixed reports ("it worked but the
rash was awful"). The slice report quantifies the first two.

**Q: How reproducible is it?**
Seed 42 in config, one seeded fold assignment stored with the data, a model card recording parameters, feature
count, scores and training time, and `uv.lock` pinning every dependency. `uv run python main.py all` rebuilds
everything from the raw CSVs.

**Q: Where could leakage still hide?**
The drug lexicon is built from the training drug column — a name list, not labels. The synonym file is hand-written
from domain knowledge. MiniLM and VADER are pretrained and frozen. Nothing else touches labels outside a training
fold.

---

## 6. Numbers to have ready

- Baseline (always neutral): **0.609** weighted F1, 0.280 macro F1.
- Shipped model: **LinearSVC**, `ctx_k0` window, `C = 0.05`, `class_weight="balanced"` — out-of-fold weighted F1
  **0.735**, macro F1 **0.586**, accuracy 0.739, fold standard deviation 0.004.
- Final out-of-fold weighted F1, macro F1, per-class recall: `artifacts/reports/oof_classification_report.csv` and
  `artifacts/model/model_card.json`.
- Worst slices: `artifacts/reports/oof_slice_scores.csv`.
- What each decision is worth: `artifacts/reports/ablation_scores.csv`.
- Predicted label mix on test: `artifacts/submission/predictions.csv`.

## 7. What I would say about how the work was done

The design decisions, the analysis and the defence above are mine; I used an AI assistant the way a developer uses
one — for implementation, refactoring and review — and I can walk through any file in `src/` and explain what it
does and why it is written that way. If the programme's policy asks for a statement on AI assistance, make it
explicitly before the viva.
