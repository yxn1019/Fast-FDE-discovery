"""Lightweight EqGPT-style knowledge prior for candidate term ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from transporteq_discovery.compat import dataclass


SPECIAL_TOKENS = {"<pad>", "E", "+", "*", "/", "S"}


@dataclass(slots=True)
class RankedCandidateTerm:
    """Ranking details for one proposed candidate term."""

    term: str
    prior_score: float
    numeric_score: float
    complexity_score: float
    combined_score: float
    penalized_score: float


def load_eqgpt_vocabulary(path: str | Path | None) -> tuple[str, ...]:
    """Load EqGPT vocabulary tokens from ``dict_datas_0725.json`` if provided."""

    if path is None:
        return ()
    vocabulary_path = Path(path)
    if not vocabulary_path.exists():
        return ()
    with vocabulary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    word2id = payload.get("word2id", {})
    terms = [
        term
        for term, token_id in sorted(word2id.items(), key=lambda item: int(item[1]))
        if term not in SPECIAL_TOKENS and int(token_id) >= 6
    ]
    return tuple(terms)


def canonical_term_name(term: str) -> str:
    """Map local derivative names to EqGPT-style token spellings."""

    clean = term.strip()
    replacements = {
        "u_xxx": "uxxx",
        "u_xx": "uxx",
        "u_x": "ux",
        "u*u_xx": "u*uxx",
        "u*u_x": "u*ux",
    }
    for old, new in replacements.items():
        clean = clean.replace(old, new)
    clean = clean.replace("Caputo", "ut")
    clean = clean.replace(" ", "")
    return clean


def compute_term_complexity(term: str) -> float:
    """Small explainable complexity penalty matching the EqGPT demo spirit."""

    clean = canonical_term_name(term)
    complexity = 0.04 * len(clean)
    complexity += 0.35 * clean.count("(")
    complexity += 0.15 * clean.count(")")
    complexity += 0.35 * clean.count("^")
    complexity += 0.50 * clean.count("*")
    complexity += 0.45 * clean.count("/")
    complexity += 0.20 * clean.count("xx")
    complexity += 0.25 * clean.count("xxx")
    complexity += 0.45 * clean.count("log")
    return float(complexity)


def prior_score_for_term(
    candidate_term: str,
    active_terms: Sequence[str],
    eqgpt_vocabulary: Sequence[str] = (),
) -> float:
    """Estimate structural plausibility from EqGPT tokens and simple priors."""

    candidate = canonical_term_name(candidate_term)
    active = {canonical_term_name(term) for term in active_terms}
    vocab = set(eqgpt_vocabulary)

    score = 0.0
    if candidate in vocab:
        score += 0.35
    if any(token in active for token in ("ux", "uxx", "uxxx")) and "ux" in candidate:
        score += 0.25
    if "ut" in candidate and any(token in active for token in ("u", "ux", "uxx")):
        score += 0.20
    if "log" in candidate and any(token in active for token in ("ut", "ux", "uxx", "u")):
        score += 0.25
    if "uxx" in candidate and "ux" in active:
        score += 0.10
    return min(float(score), 1.0)


def rank_candidate_terms(
    active_terms: Sequence[str],
    candidate_terms: Sequence[str],
    numeric_scores: Mapping[str, float] | None = None,
    *,
    eqgpt_dictionary_path: str | Path | None = None,
    gamma: float = 0.35,
    complexity_weight: float = 0.10,
) -> tuple[RankedCandidateTerm, ...]:
    """Rank candidates by numeric evidence plus EqGPT-style structural prior."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    numeric_scores = numeric_scores or {}
    vocabulary = load_eqgpt_vocabulary(eqgpt_dictionary_path)

    ranked: list[RankedCandidateTerm] = []
    for term in dict.fromkeys(candidate_terms):
        prior = prior_score_for_term(term, active_terms, vocabulary)
        numeric = float(numeric_scores.get(term, 0.0))
        complexity = compute_term_complexity(term)
        combined = (1.0 - gamma) * numeric + gamma * prior
        penalized = combined - complexity_weight * complexity
        ranked.append(
            RankedCandidateTerm(
                term=term,
                prior_score=prior,
                numeric_score=numeric,
                complexity_score=complexity,
                combined_score=float(combined),
                penalized_score=float(penalized),
            )
        )

    return tuple(sorted(ranked, key=lambda item: (-item.penalized_score, item.term)))
