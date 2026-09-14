from drug_sentiment.preprocessing.cleaning import STOP_WORDS, clean_classical, clean_minimal


def test_clean_minimal_removes_markup_but_keeps_wording():
    text = "<p>Took Ocrevus&nbsp;today!</p>\n\nSee https://example.com for   more."
    assert clean_minimal(text) == "Took Ocrevus today!\nSee url for more."


def test_links_keep_the_readable_words_of_their_path():
    text = "Read http://www.joslin.org/news/Lucentis-proves-effective.html now"
    assert clean_minimal(text) == "Read url news Lucentis proves effective now"


def test_clean_classical_keeps_negation_and_emphasis_as_tokens():
    text = "I DIDN'T like it, can't sleep & won't stop!!"
    assert clean_classical(text) == "i did not like it can not sleep will not stop ! !"


def test_stop_words_keep_negations():
    assert {"not", "no", "never"}.isdisjoint(STOP_WORDS)
    assert "the" in STOP_WORDS
