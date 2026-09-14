import pandas as pd

from drug_sentiment.analysis.eda_stats import drug_sentences, first_mention_word, label_share


def test_first_mention_word_counts_words_before_the_drug():
    assert first_mention_word("I started Gilenya last year", "gilenya") == 2
    assert first_mention_word("No drug is named here", "gilenya") == -1


def test_drug_sentences_keep_only_sentences_naming_the_drug():
    text = "I took Correctol and felt awful. Then I tried Meftal Spas. It worked wonders."
    assert drug_sentences(text, "meftal spas") == "Then I tried Meftal Spas."
    assert drug_sentences(text, "humira") == text  # falls back to the whole text


def test_label_share_rows_sum_to_one_with_group_sizes():
    df = pd.DataFrame({"drug": ["a", "a", "b"], "label": ["negative", "neutral", "neutral"]})
    share = label_share(df, "drug")
    assert share.drop(columns="n").sum(axis=1).tolist() == [1.0, 1.0]
    assert share["n"].tolist() == [2, 1]
