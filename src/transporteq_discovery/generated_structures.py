"""Generated RHS structure normalization.

This module keeps the generated structure layer separate from the numerical
operator scan. The generated token ``D_x^beta H`` is deliberately abstract:
it is bound to a concrete beta0 only when the discoverer instantiates the
candidate library. No target-structure prior is used here; EqGPT probabilities
and data-driven rewards live in ``eqgpt_adapter`` and ``gj_hybrid_discoverer``.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable


FRACTIONAL_TOKEN = "D_x^beta H"

_BARE_SECOND_ORDER_ALIASES = {"Hxx", "c_xx", "uxx", "u_xx"}
_FRACTIONAL_ALIASES = {FRACTIONAL_TOKEN, "D_x^beta c", "D_x^beta u"}

_TERM_ALIASES = {
    "1": "1",
    "H": "H",
    "c": "H",
    "u": "H",
    "Hx": "Hx",
    "c_x": "Hx",
    "ux": "Hx",
    "u_x": "Hx",
    "Hxxx": "Hxxx",
    "c_xxx": "Hxxx",
    "uxxx": "Hxxx",
    "u_xxx": "Hxxx",
    "H^2": "H^2",
    "c^2": "H^2",
    "u^2": "H^2",
    "H*Hx": "H*Hx",
    "c*c_x": "H*Hx",
    "u*ux": "H*Hx",
    "u*u_x": "H*Hx",
    "H*Hxx": "H*Hxx",
    "c*c_xx": "H*Hxx",
    "u*uxx": "H*Hxx",
    "u*u_xx": "H*Hxx",
    "H*Hxxx": "H*Hxxx",
    "H^2*Hx": "H^2*Hx",
    "c^2*c_x": "H^2*Hx",
    "u^2*ux": "H^2*Hx",
    "u^2*u_x": "H^2*Hx",
    "H^2*Hxx": "H^2*Hxx",
    "c^2*c_xx": "H^2*Hxx",
    "u^2*uxx": "H^2*Hxx",
    "u^2*u_xx": "H^2*Hxx",
    "H^2*Hxxx": "H^2*Hxxx",
}

_DEFAULT_RAW_POOL = (
    "Hx",
    "Hxx",
    "Hxxx",
    "H",
    "H^2",
    "H*Hx",
    "H*Hxx",
    "H^2*Hx",
    "H^2*Hxx",
    FRACTIONAL_TOKEN,
)

_CANONICAL_ORDER = {
    term: index
    for index, term in enumerate(
        (
            "1",
            "H",
            "Hx",
            FRACTIONAL_TOKEN,
            "Hxxx",
            "H^2",
            "H*Hx",
            "H*Hxx",
            "H*Hxxx",
            "H^2*Hx",
            "H^2*Hxx",
            "H^2*Hxxx",
        )
    )
}

@dataclass(frozen=True)
class GeneratedStructure:
    """A normalized RHS structure with provenance and ranking metadata."""

    terms: tuple[str, ...]
    raw_terms: tuple[str, ...]
    normalized_from: tuple[str, ...]
    prior_score: float
    complexity_score: float
    rank_score: float
    token_ids: tuple[int, ...] = ()
    source_equation: str = ""
    source: str = "grammar"
    invalid_reason: str | None = None

    @property
    def has_fractional(self) -> bool:
        return FRACTIONAL_TOKEN in self.terms

    @property
    def text(self) -> str:
        return " + ".join(self.terms)

    @property
    def raw_text(self) -> str:
        return " + ".join(self.raw_terms)


def normalize_rhs_terms(raw_terms: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalize a generated RHS term list.

    Bare second-order diffusion columns are mapped to the fractional macro.
    Compound nonlinear diffusion terms such as ``H*Hxx`` are intentionally kept
    unchanged in v1.
    """

    normalized: list[str] = []
    provenance: list[str] = []
    for raw in raw_terms:
        token = str(raw).strip()
        if token in _BARE_SECOND_ORDER_ALIASES:
            canonical = FRACTIONAL_TOKEN
            provenance.append(f"{token} -> {FRACTIONAL_TOKEN}")
        elif token in _FRACTIONAL_ALIASES:
            canonical = FRACTIONAL_TOKEN
            if token != FRACTIONAL_TOKEN:
                provenance.append(f"{token} -> {FRACTIONAL_TOKEN}")
        else:
            canonical = _TERM_ALIASES.get(token, token)
            if canonical != token:
                provenance.append(f"{token} -> {canonical}")
        if canonical not in normalized:
            normalized.append(canonical)
    normalized.sort(key=lambda term: (_CANONICAL_ORDER.get(term, 10_000), term))
    return tuple(normalized), tuple(provenance)


def generated_structure_from_eqgpt_tokens(
    token_ids: Iterable[int],
    id2word: Iterable[str],
    *,
    max_terms: int = 5,
) -> GeneratedStructure | None:
    """Map an EqGPT sentence to the locally supported RHS structure.

    The paper fixes the left-hand side as the time-fractional target. Therefore
    standalone time-derivative tokens such as ``ut`` are treated as LHS markers
    and removed from the generated RHS. Composite terms containing time
    derivatives are rejected because they cannot be interpreted as RHS physical
    columns in the current fractional discovery workflow.
    """

    words = tuple(str(item) for item in id2word)
    raw_ids = tuple(int(item) for item in token_ids)
    visible = [words[item] for item in raw_ids if 0 <= item < len(words) and words[item] not in {"<pad>", "S", "E"}]
    if not visible:
        return None

    raw_terms: list[str] = []
    current: list[str] = []
    current_ops: list[str] = []
    invalid_reason: str | None = None

    def flush_current() -> None:
        nonlocal invalid_reason
        if invalid_reason is not None or not current:
            current.clear()
            current_ops.clear()
            return
        mapped = _map_eqgpt_factor_expression(current, current_ops)
        if mapped is None:
            invalid_reason = "unsupported EqGPT term expression: " + "".join(
                _interleave_terms_ops(current, current_ops)
            )
        elif mapped != "__LHS__":
            raw_terms.append(mapped)
        current.clear()
        current_ops.clear()

    expect_term = True
    for token in visible:
        if token == "+":
            if expect_term:
                invalid_reason = "operator '+' appeared where a term was expected"
                break
            flush_current()
            expect_term = True
        elif token in {"*", "/"}:
            if expect_term:
                invalid_reason = f"operator '{token}' appeared where a term was expected"
                break
            current_ops.append(token)
            expect_term = True
        else:
            if not expect_term:
                invalid_reason = "two terms appeared without an operator"
                break
            current.append(token)
            expect_term = False
    if invalid_reason is None:
        if expect_term and current_ops:
            invalid_reason = "sentence ended after an operator"
        else:
            flush_current()
    if invalid_reason is not None or not raw_terms:
        return None
    normalized, provenance = normalize_rhs_terms(raw_terms)
    if not normalized or len(normalized) > max_terms:
        return None
    complexity = _structure_complexity(normalized)
    return GeneratedStructure(
        terms=normalized,
        raw_terms=tuple(raw_terms),
        normalized_from=provenance,
        prior_score=0.0,
        complexity_score=complexity,
        rank_score=-complexity,
        token_ids=raw_ids,
        source_equation="".join(visible),
        source="eqgpt",
        invalid_reason=None,
    )


def generate_rhs_structures(
    *,
    max_count: int = 100,
    max_terms: int = 5,
    seed: int = 7,
    eqgpt_dictionary_path: str | None = None,
) -> tuple[GeneratedStructure, ...]:
    """Generate neutral grammar RHS structures.

    This fallback is intentionally neutral and does not inject a target
    transport-diffusion prior. The paper-facing generated route should use the
    EqGPT adapter whenever an EqGPT checkpoint is provided.
    """

    if max_count <= 0:
        return ()
    max_terms = max(1, int(max_terms))
    rng = random.Random(int(seed))
    seen: set[tuple[str, ...]] = set()
    structures: list[GeneratedStructure] = []

    def add(raw_terms: Iterable[str]) -> None:
        raw_tuple = tuple(str(item) for item in raw_terms)
        if not raw_tuple or len(raw_tuple) > max_terms:
            return
        normalized, provenance = normalize_rhs_terms(raw_tuple)
        if not normalized or normalized in seen or len(normalized) > max_terms:
            return
        seen.add(normalized)
        complexity = _structure_complexity(normalized)
        structures.append(
            GeneratedStructure(
                terms=normalized,
                raw_terms=raw_tuple,
                normalized_from=provenance,
                prior_score=0.0,
                complexity_score=complexity,
                rank_score=-complexity,
            )
        )

    for term_count in range(1, min(max_terms, len(_DEFAULT_RAW_POOL)) + 1):
        for combo in itertools.combinations(_DEFAULT_RAW_POOL, term_count):
            if len(structures) >= max_count * 3:
                break
            add(combo)
        if len(structures) >= max_count * 3:
            break

    while len(structures) < max_count * 3 and len(seen) < 2 ** len(_DEFAULT_RAW_POOL):
        term_count = rng.randint(1, min(max_terms, len(_DEFAULT_RAW_POOL)))
        add(rng.sample(list(_DEFAULT_RAW_POOL), term_count))

    structures.sort(key=lambda item: (item.rank_score, -item.complexity_score, item.raw_text), reverse=True)
    return tuple(structures[:max_count])


def _structure_complexity(terms: tuple[str, ...]) -> float:
    complexity = float(len(terms))
    for term in terms:
        if "*" in term:
            complexity += 1.0
        if "^2" in term:
            complexity += 0.5
        if "Hxxx" in term:
            complexity += 1.0
        if term == FRACTIONAL_TOKEN:
            complexity += 0.5
    return complexity


def _interleave_terms_ops(terms: list[str], ops: list[str]) -> list[str]:
    out: list[str] = []
    for idx, term in enumerate(terms):
        out.append(term)
        if idx < len(ops):
            out.append(ops[idx])
    return out


def _map_eqgpt_factor_expression(terms: list[str], ops: list[str]) -> str | None:
    if "/" in ops:
        return None
    if len(terms) == 1 and _is_lhs_time_token(terms[0]):
        return "__LHS__"
    if any(_is_lhs_time_token(term) for term in terms):
        return None
    if any(op != "*" for op in ops):
        return None
    counts: dict[str, int] = {}
    for term in terms:
        key = _map_eqgpt_factor(term)
        if key is None:
            return None
        counts[key] = counts.get(key, 0) + 1

    if counts == {"H": 1}:
        return "H"
    if counts == {"Hx": 1}:
        return "Hx"
    if counts == {"Hxx": 1}:
        return "Hxx"
    if counts == {"Hxxx": 1}:
        return "Hxxx"
    if counts == {"H": 2} or counts == {"H^2": 1}:
        return "H^2"
    if counts == {"H": 1, "Hx": 1}:
        return "H*Hx"
    if counts == {"H": 1, "Hxx": 1}:
        return "H*Hxx"
    if counts == {"H": 1, "Hxxx": 1}:
        return "H*Hxxx"
    if counts == {"H": 2, "Hx": 1} or counts == {"H^2": 1, "Hx": 1}:
        return "H^2*Hx"
    if counts == {"H": 2, "Hxx": 1} or counts == {"H^2": 1, "Hxx": 1}:
        return "H^2*Hxx"
    if counts == {"H": 2, "Hxxx": 1} or counts == {"H^2": 1, "Hxxx": 1}:
        return "H^2*Hxxx"
    return None


def _map_eqgpt_factor(term: str) -> str | None:
    mapping = {
        "u": "H",
        "ux": "Hx",
        "uxx": "Hxx",
        "uxxx": "Hxxx",
        "u^2": "H^2",
    }
    return mapping.get(term)


def _is_lhs_time_token(term: str) -> bool:
    return term in {"ut", "utt", "uxt", "uxxt", "uxxtt", "ut^2", "ut^3", "uyyt"}
