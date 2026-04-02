from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


FRENCH_STOPWORDS = {
    "a",
    "alors",
    "au",
    "aucun",
    "aussi",
    "autre",
    "aux",
    "c",
    "ce",
    "ces",
    "cet",
    "cette",
    "d",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "il",
    "ils",
    "je",
    "l",
    "la",
    "le",
    "les",
    "leur",
    "ma",
    "mes",
    "mon",
    "ne",
    "par",
    "pas",
    "pour",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "ta",
    "te",
    "tes",
    "ton",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "y",
}


def normalize_instruction_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("'", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\bcapital\b", "capitale", text)
    text = re.sub(r"\bquelle jour\b", "quel jour", text)
    text = re.sub(r"\bquelle mois\b", "quel mois", text)
    text = re.sub(r"\bquelle animal\b", "quel animal", text)
    text = re.sub(r"\butilisee\b", "utilise", text)
    text = re.sub(r"\butilisees\b", "utilise", text)
    text = re.sub(r"\butilises\b", "utilise", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_eval_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def canonicalize_token(token: str) -> str:
    if token.endswith("ees") and len(token) > 5:
        return token[:-3]
    if token.endswith("ee") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("e") and len(token) > 4:
        return token[:-1]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in normalize_instruction_text(text).split():
        if token in FRENCH_STOPWORDS:
            continue
        token = canonicalize_token(token)
        if token and token not in FRENCH_STOPWORDS:
            tokens.append(token)
    return tokens


def content_similarity(a: str, b: str) -> tuple[float, float]:
    tokens_a = set(content_tokens(a))
    tokens_b = set(content_tokens(b))
    if not tokens_a or not tokens_b:
        return 0.0, 0.0
    overlap = len(tokens_a & tokens_b)
    precision = overlap / len(tokens_b)
    recall = overlap / len(tokens_a)
    return precision, recall


def instruction_similarity(a: str, b: str) -> float:
    tokens_a = set(normalize_instruction_text(a).split())
    tokens_b = set(normalize_instruction_text(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    ratio = SequenceMatcher(None, normalize_instruction_text(a), normalize_instruction_text(b)).ratio()
    precision, recall = content_similarity(a, b)
    return 0.35 * jaccard + 0.2 * ratio + 0.2 * precision + 0.25 * recall


def normalize_capital_subject_tokens(tokens: list[str], start: int = 0) -> str | None:
    while start < len(tokens) and tokens[start] in {"de", "du", "des", "d", "la", "le", "les", "l"}:
        start += 1
    subject_tokens = tokens[start:]
    if not subject_tokens:
        return None
    return " ".join(subject_tokens)


def extract_capital_subject(text: str) -> str | None:
    tokens = normalize_instruction_text(text).split()
    if "capitale" not in tokens:
        return None
    return normalize_capital_subject_tokens(tokens, tokens.index("capitale") + 1)


@dataclass
class ParsedCapitalResponse:
    country_phrase: str
    capital: str


def parse_capital_response(response: str) -> ParsedCapitalResponse | None:
    match = re.match(r"^\s*La capitale (.+) est ([^.]+)\.?\s*$", response)
    if not match:
        return None
    return ParsedCapitalResponse(country_phrase=match.group(1).strip(), capital=match.group(2).strip())


def is_degenerate_text(text: str) -> bool:
    compact = text.replace(" ", "")
    return bool(compact and len(compact) >= 12 and len(set(compact)) <= 2)
