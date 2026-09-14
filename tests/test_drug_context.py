from drug_sentiment.preprocessing.drug_context import (
    SYNONYMS_PATH,
    DrugContextExtractor,
    DrugLexicon,
    split_sentences,
)

# The example from the capstone brief: one comment, two drugs, opposite sentiment.
BRIEF_EXAMPLE = (
    "I looked up stomach pain after taking Correctol...and here I am, wishing I had read these comments "
    "before I took it. I'm sitting here with the worst stomach pain I've had since labor. I will NEVER take "
    "this medication again! Post that I took Meftal Spas and it worked wonders for me."
)


def make_extractor(names, synonym_groups=(), max_window_words=250):
    lexicon = DrugLexicon(names, synonym_groups, fuzzy_threshold=80)
    return DrugContextExtractor(lexicon, "targetdrug", "otherdrug", window_sizes=(0, 1, 2),
                                natural_window_size=1, max_window_words=max_window_words)


def test_same_comment_gives_each_drug_its_own_window():
    extractor = make_extractor(["correctol", "meftal spas"])
    meftal = extractor.extract(BRIEF_EXAMPLE, "Meftal Spas")
    correctol = extractor.extract(BRIEF_EXAMPLE, "correctol")

    assert meftal.windows_masked[0] == "Post that I took targetdrug and it worked wonders for me."
    assert correctol.windows_masked[0].startswith("I looked up stomach pain after taking targetdrug")
    assert "otherdrug" not in correctol.windows_masked[0]
    assert "targetdrug" in correctol.full_masked and "otherdrug" in correctol.full_masked
    assert meftal.n_other_drugs == correctol.n_other_drugs == 1


def test_brand_and_generic_names_count_as_the_target():
    extractor = make_extractor(["gilenya", "fingolimod", "ocrevus"], synonym_groups=[["gilenya", "fingolimod"]])
    context = extractor.extract("I switched from Ocrevus to Fingolimod last year.", "gilenya")

    assert context.match_type == "exact"
    assert context.mention_count == 1
    assert context.windows_masked[0] == "I switched from otherdrug to targetdrug last year."
    assert context.n_other_drugs == 1


def test_misspelt_drug_name_is_found_by_fuzzy_matching():
    context = make_extractor(["stelara"]).extract("Started Stelara in May and feel better.", "stellara")

    assert context.match_type == "fuzzy"
    assert context.windows_masked[0] == "Started targetdrug in May and feel better."
    assert context.n_other_drugs == 0


def test_missing_drug_falls_back_to_the_whole_text():
    context = make_extractor(["humira"]).extract("Any advice?", "humira")

    assert (context.match_type, context.mention_count, context.first_mention_pos) == ("none", 0, -1.0)
    assert context.windows_masked[1] == "Any advice?"


def test_long_window_is_cropped_around_the_mention():
    filler = " ".join(["word"] * 600)
    context = make_extractor(["humira"], max_window_words=50).extract(f"{filler} Humira helped me {filler}", "humira")

    words = context.windows_masked[2].split()
    assert "targetdrug" in words
    assert len(words) <= 50
    assert "Humira" in context.window_natural.split()


def test_synonym_file_groups_brand_and_generic_names():
    lexicon = DrugLexicon.from_train(["gilenya", "tarceva"], fuzzy_threshold=80, synonyms_path=SYNONYMS_PATH)

    assert lexicon.same_drug("Gilenya") == {"gilenya", "fingolimod"}
    assert "erlotinib" in lexicon.same_drug("tarceva")


def test_split_sentences_handles_missing_space_after_period():
    assert split_sentences("Great drug.It works!\nNew line") == ["Great drug.", "It works!", "New line"]
