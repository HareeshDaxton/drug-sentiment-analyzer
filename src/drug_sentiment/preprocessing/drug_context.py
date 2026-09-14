"""Locate the target drug in a comment and build drug-centred, masked context windows.

A comment can discuss several drugs with different sentiment, and each row's label belongs to one of
them. Cropping the text to the sentences around the target drug and masking drug names
(target -> `targetdrug`, any other known drug -> `otherdrug`) makes the model input explicitly about
the row's drug. It also keeps the relevant part inside a pretrained model's length limit: in ~7% of
training rows the first mention appears after word 380.

Everything here is row-wise. The only reference data is the list of drug names from train.csv's
`drug` column plus a hand-written brand/generic synonym file; no labels or test data are used.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

from drug_sentiment.config import PROJECT_ROOT

SYNONYMS_PATH = PROJECT_ROOT / "config" / "drug_synonyms.yaml"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[a-z][.!?])(?=[A-Z])|\n+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def normalise_drug(name: str) -> str:
    return re.sub(r"[\s\-]+", " ", name.strip().lower())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def _name_pattern(name: str) -> str:
    """Pattern for one drug name: tolerant to space/hyphen variants, never matching inside a longer word."""
    parts = [re.escape(part) for part in name.split()]
    return r"(?<![a-z0-9])" + r"[\s\-]*".join(parts) + r"(?![a-z0-9])"


@lru_cache(maxsize=4096)
def _alternation(names: frozenset[str]) -> re.Pattern[str]:
    ordered = sorted(names, key=len, reverse=True)  # longest first so "meftal spas" wins over "meftal"
    return re.compile("|".join(_name_pattern(n) for n in ordered), re.IGNORECASE)


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _crop_words(text: str, is_anchor: Callable[[str], bool], max_words: int) -> str:
    """Limit text to max_words, keeping the words around anchor (drug mention) words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    anchors = [i for i, word in enumerate(words) if is_anchor(word)]
    if not anchors:
        return " ".join(words[:max_words])

    radius = max(max_words // (2 * len(anchors)), 15)
    kept, budget = [], max_words
    for start, end in _merge_ranges((max(0, i - radius), min(len(words) - 1, i + radius)) for i in anchors):
        span = words[start : end + 1][:budget]
        kept.append(" ".join(span))
        budget -= len(span)
        if budget <= 0:
            break
    return " ... ".join(kept)


class DrugLexicon:
    """Known drug names plus brand/generic groups (e.g. Gilenya = fingolimod)."""

    def __init__(self, names: Iterable[str], synonym_groups: Iterable[Iterable[str]], fuzzy_threshold: int) -> None:
        self._synonyms: dict[str, frozenset[str]] = {}
        for group in synonym_groups:
            members = frozenset(normalise_drug(m) for m in group)
            for member in members:
                self._synonyms[member] = self._synonyms.get(member, frozenset()) | members
        self.all_names = frozenset(normalise_drug(n) for n in names) | frozenset(self._synonyms)
        self.fuzzy_threshold = fuzzy_threshold

    @classmethod
    def from_train(cls, drugs: Iterable[str], fuzzy_threshold: int, synonyms_path: Path = SYNONYMS_PATH) -> DrugLexicon:
        with open(synonyms_path, encoding="utf-8") as f:
            groups = yaml.safe_load(f) or {}
        return cls(names=drugs, synonym_groups=groups.values(), fuzzy_threshold=fuzzy_threshold)

    def same_drug(self, name: str) -> frozenset[str]:
        name = normalise_drug(name)
        return self._synonyms.get(name, frozenset({name}))

    def find_fuzzy_alias(self, drug: str, text: str) -> str | None:
        """Closest spelling of `drug` in the text, for misspellings such as 'stellara' vs 'Stelara'."""
        n_parts = len(drug.split())
        words = _WORD_RE.findall(text.lower())
        candidates = {
            " ".join(words[i : i + n_parts])
            for i in range(len(words) - n_parts + 1)
            if words[i][0] == drug[0]
        }
        candidates = {c for c in candidates if abs(len(c) - len(drug)) <= 2}
        match = process.extractOne(drug, candidates, scorer=fuzz.ratio, score_cutoff=self.fuzzy_threshold)
        return match[0] if match else None


@dataclass(frozen=True)
class DrugContext:
    match_type: str  # "exact", "fuzzy" (misspelling) or "none"
    mention_count: int
    first_mention_pos: float  # relative character position of the first mention; -1 when not found
    n_mention_sentences: int
    n_other_drugs: int  # distinct other molecules mentioned in the comment
    full_masked: str
    windows_masked: dict[int, str]  # k -> mention sentences +/- k neighbours, drugs masked
    window_natural: str  # unmasked window for pretrained models


class DrugContextExtractor:
    """Row-wise: (text, drug) -> DrugContext. Learns nothing from the data, so it cannot leak."""

    def __init__(
        self,
        lexicon: DrugLexicon,
        target_token: str,
        other_token: str,
        window_sizes: tuple[int, ...],
        natural_window_size: int,
        max_window_words: int,
    ) -> None:
        self.lexicon = lexicon
        self.target_token = target_token
        self.other_token = other_token
        self.window_sizes = window_sizes
        self.natural_window_size = natural_window_size
        self.max_window_words = max_window_words

    def extract(self, text: str, drug: str) -> DrugContext:
        drug = normalise_drug(drug)
        target_names = self.lexicon.same_drug(drug)
        match_type = "exact" if _alternation(target_names).search(text) else "none"
        if match_type == "none":
            alias = self.lexicon.find_fuzzy_alias(drug, text)
            if alias:
                match_type = "fuzzy"
                target_names = target_names | {alias} | self.lexicon.same_drug(alias)

        target_re = _alternation(target_names)
        other_re = _alternation(self.lexicon.all_names - target_names)

        def mask(fragment: str) -> str:
            return other_re.sub(self.other_token, target_re.sub(self.target_token, fragment))

        mentions = list(target_re.finditer(text))
        other_molecules = {min(self.lexicon.same_drug(m.group(0))) for m in other_re.finditer(text)}
        sentences = split_sentences(text)
        anchor_ids = [i for i, sentence in enumerate(sentences) if target_re.search(sentence)]

        raw_windows = {
            k: self._sentence_window(sentences, anchor_ids, k) if anchor_ids else text
            for k in {*self.window_sizes, self.natural_window_size}
        }
        return DrugContext(
            match_type=match_type,
            mention_count=len(mentions),
            first_mention_pos=mentions[0].start() / len(text) if mentions else -1.0,
            n_mention_sentences=len(anchor_ids),
            n_other_drugs=len(other_molecules),
            full_masked=mask(text),
            windows_masked={
                k: _crop_words(mask(raw_windows[k]), lambda w: self.target_token in w, self.max_window_words)
                for k in self.window_sizes
            },
            window_natural=_crop_words(
                raw_windows[self.natural_window_size],
                lambda w: target_re.search(w) is not None,
                self.max_window_words,
            ),
        )

    @staticmethod
    def _sentence_window(sentences: list[str], anchor_ids: list[int], k: int) -> str:
        last = len(sentences) - 1
        ranges = _merge_ranges((max(0, i - k), min(last, i + k)) for i in anchor_ids)
        return " ... ".join(" ".join(sentences[start : end + 1]) for start, end in ranges)
