# Drug Sentiment Analysis

Analytics Vidhya Blackbelt capstone: predict the sentiment of a patient comment **toward a specific drug**
(0 = positive, 1 = negative, 2 = neutral). Metric: weighted F1.

## Setup

```bash
uv sync
```

Place the competition files in `Data/raw/`:
`train_F3WbcTw_1icmK82.csv`, `test_tOlRoBf_VeRQtHl.csv`, `sample_submission_Wy5x1QK.csv`.

## Pipeline stages

| Stage | Command | Output |
|---|---|---|
| Ingest & validate | `uv run python main.py ingest` | log of record counts and validation checks |
| Preprocess | `uv run python main.py preprocess` | `artifacts/preprocessed/` train and test pkl files, drug lexicon |

The preprocessed pkl files hold only row-wise outputs: cleaned text, drug-centred windows with the target
drug masked as `targetdrug` and other drugs as `otherdrug`, drug-mention and handcrafted features, and (train
only) the label and a CV `fold` column. Anything fitted on data (TF-IDF, scaling, encoders) happens later
inside each model pipeline, so the files are safe to reuse for every model and fold.

Run tests with `uv run pytest`.

## Exploratory analysis

[`NoteBook/01_EDA.ipynb`](NoteBook/01_EDA.ipynb) explores the data and records the modelling decision each finding
leads to. Its 9 figures are saved to `artifacts/reports/figures/` for the presentation.

## Project layout

```
config/config.yaml        paths and pipeline settings
src/drug_sentiment/       package: data, preprocessing, features, models, ...
main.py                   stage runner
tests/                    pytest suite
artifacts/                preprocessed pkl files, models, reports, submission
NoteBook/                 exploration notebooks
```

## Data note

The brief quotes 5,619 train / 3,107 test rows. Those are physical line counts: some comments contain line
breaks inside a quoted CSV field. The files hold **5,279** and **2,924** records, and `sample_submission`
has 2,924 ids in test order, so the submission has 2,924 rows.
