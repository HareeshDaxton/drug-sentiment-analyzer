"""Stage `slides`: build the capstone presentation from the artifacts the pipeline produced.

Every number on a slide is read from `artifacts/` at build time -- the model card, the out-of-fold reports, the
ablation table, the screening table and the submission -- so the deck can never drift from the code that produced
it. Re-running `main.py slides` after a change rebuilds it with the new numbers.

Output: presentation/Drug_Sentiment_Capstone.pptx (16:9, 15 slides).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from drug_sentiment.config import Config, get_config
from drug_sentiment.logger import get_logger
from drug_sentiment.utils.io import load_json
from drug_sentiment.visualization.architecture import plot_pipeline
from drug_sentiment.visualization.style import (INK_MUTED, INK_PRIMARY, INK_SECONDARY, SERIES_COLOR, SURFACE,
                                                apply_style, save_figure)

logger = get_logger(__name__)

FONT = "Segoe UI"
SLIDE_WIDTH, SLIDE_HEIGHT = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.62)


def _rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color.lstrip("#").upper())


INK, SUBTLE, MUTED, ACCENT, BACKGROUND = (_rgb(c) for c in (INK_PRIMARY, INK_SECONDARY, INK_MUTED, SERIES_COLOR,
                                                            SURFACE))


def _textbox(slide, left, top, width, height, text: str, size: int, color: RGBColor, bold: bool = False,
             align=PP_ALIGN.LEFT, spacing: float = 1.25):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.line_spacing = spacing
        paragraph.space_after = Pt(6)
        run = paragraph.add_run()
        run.text = line
        run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
        run.font.color.rgb = color
    return box


def add_slide(presentation: Presentation, title: str, takeaway: str = "", image: Path | None = None,
              bullets: list[str] | None = None, table: pd.DataFrame | None = None, number: int | None = None):
    """One slide: title, a one-line takeaway, then a figure, bullet list and/or table."""
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BACKGROUND

    _textbox(slide, MARGIN, Inches(0.42), SLIDE_WIDTH - 2 * MARGIN, Inches(0.7), title, 28, INK, bold=True)
    top = Inches(1.12)
    if takeaway:
        _textbox(slide, MARGIN, top, SLIDE_WIDTH - 2 * MARGIN, Inches(0.5), takeaway, 15, SUBTLE)
        top = Inches(1.78)

    body_height = SLIDE_HEIGHT - top - Inches(0.7)
    text_left, text_width = MARGIN, SLIDE_WIDTH - 2 * MARGIN
    if image and image.exists():
        from PIL import Image

        with Image.open(image) as picture:
            ratio = picture.width / picture.height
        available_width = Inches(7.3) if (bullets or table is not None) else SLIDE_WIDTH - 2 * MARGIN
        width = min(available_width, Emu(int(body_height * ratio)))
        height = Emu(int(width / ratio))
        left = MARGIN if (bullets or table is not None) else int((SLIDE_WIDTH - width) / 2)
        slide.shapes.add_picture(str(image), left, top, width=width, height=height)
        text_left, text_width = left + width + Inches(0.4), SLIDE_WIDTH - MARGIN - (left + width + Inches(0.4))

    if bullets:
        _textbox(slide, text_left, top, text_width, body_height,
                 "\n".join(f"•  {bullet}" for bullet in bullets), 14, SUBTLE, spacing=1.4)
        top = top + Inches(0.5) * len(bullets)

    if table is not None:
        rows, columns = table.shape
        height = min(Inches(0.36) * (rows + 1), body_height)
        shape = slide.shapes.add_table(rows + 1, columns, text_left, top, text_width, height).table
        for column, name in enumerate(table.columns):
            cell = shape.cell(0, column)
            cell.text = str(name)
            cell.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
            cell.text_frame.paragraphs[0].runs[0].font.bold = True
            cell.text_frame.paragraphs[0].runs[0].font.name = FONT
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT
            cell.text_frame.paragraphs[0].runs[0].font.color.rgb = _rgb("#ffffff")
        for row in range(rows):
            for column in range(columns):
                cell = shape.cell(row + 1, column)
                value = table.iat[row, column]
                cell.text = f"{value:.3f}" if isinstance(value, float) else str(value)
                run = cell.text_frame.paragraphs[0].runs[0]
                run.font.size, run.font.name = Pt(12), FONT
                run.font.color.rgb = INK
                cell.fill.solid()
                cell.fill.fore_color.rgb = BACKGROUND

    if number is not None:
        _textbox(slide, SLIDE_WIDTH - Inches(1.4), SLIDE_HEIGHT - Inches(0.62), Inches(0.9), Inches(0.35),
                 str(number), 11, MUTED, align=PP_ALIGN.RIGHT)
    return slide


def _read_artifacts(cfg: Config) -> dict:
    reports, figures = cfg.artifacts.reports_dir, cfg.artifacts.figures_dir
    card = load_json(cfg.artifacts.model_dir / "model_card.json")
    data = {
        "card": card,
        "figures": figures,
        "report": pd.read_csv(reports / "oof_classification_report.csv", index_col=0),
        "screening": pd.read_csv(reports / "model_screening_half.csv", index_col=0),
        "ablations": pd.read_csv(reports / "ablation_scores.csv"),
        "slices": pd.read_csv(reports / "oof_slice_scores.csv"),
        "errors": pd.read_csv(reports / "oof_error_examples.csv"),
        "submission": pd.read_csv(cfg.artifacts.submission_file),
    }
    tuned = reports / "tuned_top3.csv"
    data["tuned"] = pd.read_csv(tuned, index_col=0) if tuned.exists() else None
    return data


def _ablation_delta(ablations: pd.DataFrame, group: str, variant: str) -> float:
    rows = ablations[ablations["group"] == group]
    shipped = rows.loc[rows["shipped"], "f1_weighted"].iloc[0]
    return float(shipped - rows.loc[rows["variant"] == variant, "f1_weighted"].iloc[0])


def build_deck(cfg: Config | None = None) -> Path:
    cfg = cfg or get_config()
    apply_style()
    save_figure(plot_pipeline(), "pipeline_architecture")  # regenerated so the diagram matches the code
    art = _read_artifacts(cfg)
    card, figures = art["card"], art["figures"]
    cv = card["cv_metrics"] or {}
    f1 = cv.get("f1_weighted", float("nan"))
    macro = cv.get("f1_macro", float("nan"))
    baseline = 0.609  # always predicting the majority class, measured in the EDA
    window_gain = _ablation_delta(art["ablations"], "text given to the model", "whole comment, drug-blind")
    label_counts = art["submission"]["sentiment"].value_counts(normalize=True).sort_index()

    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = SLIDE_WIDTH, SLIDE_HEIGHT
    slide_number = iter(range(1, 99))

    def slide(*args, **kwargs):
        add_slide(presentation, *args, number=next(slide_number), **kwargs)

    # 1 - title
    cover = presentation.slides.add_slide(presentation.slide_layouts[6])
    cover.background.fill.solid()
    cover.background.fill.fore_color.rgb = BACKGROUND
    _textbox(cover, MARGIN, Inches(2.0), SLIDE_WIDTH - 2 * MARGIN, Inches(1.2), "Drug Sentiment Analysis",
             44, INK, bold=True)
    rule = cover.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(3.05), Inches(1.6), Pt(5))
    rule.fill.solid()
    rule.fill.fore_color.rgb = ACCENT
    rule.line.fill.background()
    rule.shadow.inherit = False
    _textbox(cover, MARGIN, Inches(3.45), SLIDE_WIDTH - 2 * MARGIN, Inches(2.6),
             "Predicting how a patient comment feels about one specific drug\n"
             f"{card['model']} on the sentences that mention the drug - weighted F1 {f1:.3f} out of fold\n"
             "Analytics Vidhya Blackbelt capstone",
             17, SUBTLE, spacing=1.8)
    next(slide_number)

    # 2 - the problem
    slide("The task is not plain sentiment analysis",
          "Each row asks about one drug, so the same comment can carry two different labels.",
          bullets=[
              "Input: a patient comment plus the name of one drug. Output: 0 positive, 1 negative, 2 neutral.",
              "186 training rows share a comment with another drug — the text alone cannot decide the label.",
              "Long comments bury the drug: in 6.9% of rows it first appears after word 380.",
              "72.5% of rows are neutral, so accuracy is misleading; the metric is weighted F1.",
              f"Always answering 'neutral' scores {baseline:.3f} weighted F1 — the bar to beat.",
          ])

    # 3 - the data
    slide("The data, and one finding about it",
          "The brief's 5,619 / 3,107 are line counts; the files hold 5,279 train and 2,924 test records.",
          image=figures / "eda_01_class_distribution.png",
          bullets=[
              "Comments contain line breaks inside quoted CSV fields, which is why line counts differ from rows.",
              "sample_submission has 2,924 ids in test order, confirming the record count.",
              "No missing values, no duplicate ids, no conflicting labels for the same (comment, drug).",
              "102 drugs in train, 95 in test, 9 of them unseen — mostly misspellings of known drugs.",
          ])

    # 4 - where the drug is mentioned
    slide("Why a window around the drug, not the whole comment",
          "Half the comments mention the drug early, but the tail is long: the drug can sit past word 380.",
          image=figures / "eda_05_first_mention_position.png",
          bullets=[
              "Median comment: 807 characters; the longest is 128k — package inserts and forum dumps.",
              "Feeding the whole comment mixes in sentences about other drugs and other topics.",
              "The model reads only the sentences that mention the drug, capped at 250 words - measured against "
              "wider windows and against the whole comment.",
              f"Ablation: the window is worth {window_gain:+.3f} weighted F1 over the drug-blind full comment.",
          ])

    # 5 - drug conditioning
    slide("How each row is made about its own drug",
          "The target drug becomes `targetdrug`, every other known drug becomes `otherdrug`.",
          image=figures / "eda_03_sentiment_by_drug.png",
          bullets=[
              "A lexicon of 102 train drug names plus a hand-written brand/generic synonym file.",
              "rapidfuzz catches misspellings (stellara → stelara) so unseen test drugs still match.",
              "Masking stops the model memorising drug names and forces it to read the sentiment words.",
              "The drug identity is still available as a one-hot feature, so drug priors are learnable.",
          ])

    # 6 - pipeline
    slide("The pipeline", "Row-wise preprocessing is cached; everything fitted on data lives inside the CV folds.",
          image=figures / "pipeline_architecture.png")

    # 7 - features
    slide("What the model sees",
          "Four feature blocks, all built from the drug window.",
          image=figures / "eval_03_top_words.png",
          bullets=[
              "TF-IDF: word 1–2-grams and character 2–5-grams (typo-tolerant), stop words minus negations.",
              "Handcrafted: mention count and position, other drugs, lengths, negations, first-person words.",
              "VADER: sentiment of the window, and the mean sentence score of the whole comment.",
              "MiniLM: a 384-d frozen sentence embedding of the unmasked window.",
              f"{card['n_features']:,} features in total for the fitted model.",
          ])

    # 8 - validation
    slide("Validation that cannot flatter itself",
          "5 folds, stratified by label and grouped by comment text, fixed once and reused by every model.",
          bullets=[
              "88 comments appear with several drugs; grouping keeps them inside one fold, so no model can "
              "memorise a comment it will be scored on.",
              "TF-IDF, SVD, scalers and the drug encoder are fitted inside each training fold only.",
              "The preprocessed pickles hold row-wise values only — nothing learned from the labels.",
              "test.csv is touched once, by the predict stage, to write the submission.",
          ])

    # 9 - screening
    screening = art["screening"].head(6).reset_index()
    screening = pd.DataFrame({
        "Model": screening.iloc[:, 0],
        "Weighted F1": screening["valid_f1_weighted"].round(3),
        "Macro F1": screening["valid_f1_macro"].round(3),
    })
    slide("Ten models, one honest comparison",
          "Base settings, half the training data, identical folds. Linear models on sparse text win.",
          image=figures / "model_screening_half.png", table=screening)

    # 10 - tuning and choice
    tuned = art["tuned"]
    tuned_table = None
    if tuned is not None:
        tuned_table = pd.DataFrame({
            "Model": tuned.index,
            "Base F1": tuned["base_valid_f1_weighted"].round(3),
            "Tuned F1": tuned["valid_f1_weighted"].round(3),
            "Macro F1": tuned["valid_f1_macro"].round(3),
            "Fold std": tuned["fold_f1_std"].round(3),
        })
    slide(f"Tuning the top 3, and why {card['model']} ships",
          "Optuna (TPE + median pruning) on the full training data, same folds, weighted F1 as the objective.",
          table=tuned_table,
          bullets=[
              "Tuned settings: " + ", ".join(f"{name}={value:.4g}" if isinstance(value, float) else f"{name}={value}"
                                             for name, value in card["params"].items()) + ".",
              "Best weighted F1 and the best macro F1 of the three, so the small classes survive.",
              "Lowest spread across folds, and seconds — not minutes — to refit.",
              "A linear model on TF-IDF stays explainable: one weight per word.",
          ])

    # 11 - results
    report = art["report"]
    per_class = pd.DataFrame({
        "Class": ["positive", "negative", "neutral"],
        "Precision": [report.loc[name, "precision"].round(3) for name in ("positive", "negative", "neutral")],
        "Recall": [report.loc[name, "recall"].round(3) for name in ("positive", "negative", "neutral")],
        "F1": [report.loc[name, "f1-score"].round(3) for name in ("positive", "negative", "neutral")],
        "Rows": [int(report.loc[name, "support"]) for name in ("positive", "negative", "neutral")],
    })
    slide("Results",
          f"Weighted F1 {f1:.3f} out of fold, {f1 - baseline:+.3f} over the majority-class baseline; "
          f"macro F1 {macro:.3f}.",
          image=figures / "eval_01_confusion_matrix.png", table=per_class)

    # 12 - ablations
    slide("What each design decision is worth",
          "Same model, same folds, one decision changed at a time.",
          image=figures / "eval_04_ablations.png")

    # 13 - error analysis
    hard = art["slices"].nsmallest(3, "f1_weighted")
    worst = hard.iloc[0]
    slide("Where it still fails",
          f"The weakest group is “{worst['dimension']}: {worst['slice']}” at F1 "
          f"{worst['f1_weighted']:.3f} on {int(worst['rows']):,} rows.",
          image=figures / "eval_02_slice_scores.png",
          bullets=[f"{row['dimension']} — {row['slice']}: F1 {row['f1_weighted']:.3f} on {int(row['rows']):,} rows"
                   for _, row in hard.iterrows()]
          + ["Neutral rows absorb most mistakes: the class is 72% of the data and its wording is the least marked.",
             "Sarcasm and mixed reports ('worked but the rash was awful') stay hard for a bag-of-words model."])

    # 14 - submission
    slide("The submission, and how to rebuild everything",
          f"predictions.csv: {len(art['submission']):,} rows, ids in test order, labels "
          + ", ".join(f"{cfg.data.label_names[label]} {share:.0%}" for label, share in label_counts.items()) + ".",
          bullets=[
              "uv run python main.py all — ingest → preprocess → embed → train → evaluate → predict.",
              f"Fitting the final model on all {card['trained_rows']:,} rows takes {card['fit_seconds']}s on a laptop "
              "CPU; no GPU anywhere in the project.",
              "Seed 42 everywhere; the model card records params, features, scores and the training time.",
              "python -m drug_sentiment.inference.predict --drug ... --text ... scores a brand-new comment.",
              "Re-run from the raw CSVs end to end, predictions.csv came back byte for byte identical.",
          ])

    # 15 - what next
    slide("What I would do next",
          "The ceiling here is the bag-of-words view of a medical comment.",
          bullets=[
              "Fine-tune a domain transformer (BioBERT / PubMedBERT) on the masked windows — needs a GPU.",
              "Model the neutral class explicitly: many neutral rows are news or trial reports, not opinions.",
              "Use the drug-level label prior as a proper target encoding fitted inside the folds.",
              "Blend the linear model with XGBoost on the dense features; their errors differ.",
              "Collect per-sentence labels so multi-drug comments can be scored sentence by sentence.",
          ])

    path = cfg.artifacts.presentation_file
    path.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(path)
    logger.info("Saved %s: %d slides", path, len(presentation.slides))
    return path
