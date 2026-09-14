"""Text normalisation at two strengths.

clean_minimal   -> for pretrained transformers: keep case, punctuation and wording; only remove markup noise.
clean_classical -> for TF-IDF / bag-of-words: additionally lowercase, expand negated contractions
                   ("didn't" -> "did not", so negation survives tokenisation) and strip symbols.
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlsplit

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations flip sentiment ("not effective"), but sklearn's English stop-word list would remove them.
NEGATIONS = frozenset({"not", "no", "never", "nor", "cannot"})
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS

_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,200}>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_URL_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_URL_NOISE = frozenset({"www", "com", "org", "net", "html", "htm", "php", "asp", "aspx"})
_JUNK_CHARS_RE = re.compile("[�​­]")  # replacement char, zero-width space, soft hyphen
_SPACES_RE = re.compile(r"[^\S\n]+")
_NEWLINES_RE = re.compile(r"\s*\n\s*")
_WHITESPACE_RE = re.compile(r"\s+")

# Order matters: the irregular forms must be expanded before the generic "n't" rule.
_CONTRACTIONS = (
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"n't\b"), " not"),
)
_SYMBOLS_RE = re.compile(r"[^a-z0-9!?\s]")
_ELONGATED_RE = re.compile(r"([a-z])\1{2,}")  # "sooooo" -> "soo"
_EMPHASIS_RE = re.compile(r"([!?])")


def _url_to_words(match: re.Match[str]) -> str:
    """Replace a link with a `url` token plus the readable words of its path. Slugs such as
    /Lucentis-proves-effective often name the drug, and some comments name it only there."""
    link = match.group(0)
    try:
        parts = urlsplit(link if link.lower().startswith("http") else f"http://{link}")
    except ValueError:  # malformed link
        return " url "
    words = [word for word in _URL_WORD_RE.findall(f"{parts.path} {parts.query}") if word.lower() not in _URL_NOISE]
    return f" url {' '.join(words)} "


def clean_minimal(text: str) -> str:
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(_url_to_words, text)
    text = _JUNK_CHARS_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text)
    return _NEWLINES_RE.sub("\n", text).strip()  # keep line breaks: they mark sentence boundaries


def clean_classical(text: str) -> str:
    text = clean_minimal(text).lower().replace("’", "'")
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    text = _SYMBOLS_RE.sub(" ", text)
    text = _ELONGATED_RE.sub(r"\1\1", text)
    text = _EMPHASIS_RE.sub(r" \1 ", text)  # "great!!" -> "great ! !" so emphasis becomes a token
    return _WHITESPACE_RE.sub(" ", text).strip()
