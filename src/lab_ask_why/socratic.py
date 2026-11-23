"""Socratic 'why' question generation grounded in retrieved rationale evidence."""
from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

from lab_ask_why.bm25 import tokenize

_GENERIC_RATIONALE_BOILERPLATE = frozenset(
    {
        "conclusion",
        "conclusions",
        "background",
        "objective",
        "objectives",
        "methods",
        "results",
        "discussion",
        "study",
        "studies",
        "found",
        "show",
        "showed",
        "shown",
        "shows",
        "suggest",
        "suggests",
        "suggested",
        "propose",
        "proposed",
        "however",
        "therefore",
        "thus",
        "may",
        "can",
        "could",
        "would",
        "also",
        "used",
        "using",
        "between",
        "among",
        "whether",
        "appear",
        "appears",
        "associated",
        "significant",
        "significantly",
        "substantially",
        "contribute",
        "contributes",
        "careful",
        "consideration",
        "interpretation",
        "alternative",
        "explanation",
        "possibility",
        "requires",
        "identified",
        "least",
        "more",
        "less",
        "often",
        "much",
        "one",
        "two",
        "other",
        "normal",
        "model",
        "predicted",
        "result",
        "common",
        "uncommon",
        "create",
        "creates",
        "created",
        "creating",
        "increase",
        "increased",
        "increases",
        "decrease",
        "decreased",
        "decreases",
        "individuals",
        "children",
        "subjects",
        "patients",
        "have",
        "has",
        "had",
        "easily",
        "convincingly",
        "replicated",
        "hundreds",
        "follow",
    }
)


def _format_embedded_claim(clean_claim: str) -> str:
    """Lowercase leading character only when the first word is a standard capitalized word or 'A'.

    Preserves biomedical acronyms, gene symbols, chemical formulas, and digits
    (e.g., 'GWAS', 'RNA', 'ATP', 'IL-6', 'COVID-19', 'E. coli', '1,000').
    """
    first_word = clean_claim.split()[0]
    if first_word in {"A", "An", "The"}:
        return clean_claim[0].lower() + clean_claim[1:]
    if (
        len(first_word) > 1
        and first_word[0].isascii()
        and first_word[0].isupper()
        and first_word[1:].isalpha()
        and first_word[1:].islower()
    ):
        return clean_claim[0].lower() + clean_claim[1:]
    return clean_claim


def extract_differential_terms(
    claim: str,
    top_sentences: Sequence[str],
    *,
    max_terms: int = 3,
) -> list[str]:
    """Extract salient mechanism/confounder tokens present in retrieved sentences but absent from claim."""
    claim_tokens = set(tokenize(claim))
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    order_idx = 0
    for sent in top_sentences:
        for tok in tokenize(sent):
            if (
                tok in claim_tokens
                or tok in _GENERIC_RATIONALE_BOILERPLATE
                or len(tok) < 4
                or tok[0].isdigit()
                or tok in {"http", "https", "www", "gov", "org"}
            ):
                continue
            counts[tok] += 1
            if tok not in first_seen:
                first_seen[tok] = order_idx
                order_idx += 1

    # Prefer recurring or specific biomedical mechanism terms from top-ranked sentences
    ranked = sorted(
        counts.keys(),
        key=lambda t: (-counts[t], -min(len(t), 12), first_seen[t]),
    )
    deduped: list[str] = []
    seen_stems: set[str] = set()
    for term in ranked:
        stem = term.rstrip("s") if len(term) > 4 else term
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        deduped.append(term)
        if len(deduped) >= max(1, int(max_terms)):
            break
    return deduped


def why_questions(
    claim: str,
    top_sentences: Sequence[str] | None = None,
    *,
    stance: str | None = None,
    llm_fn: Callable[[str, Sequence[str], str | None], list[str]] | None = None,
) -> list[str]:
    """Generate three Socratic epistemic probes for `claim`, optionally grounded in `top_sentences`.

    Raises `ValueError` if `claim` is empty or contains only whitespace/punctuation.
    """
    if not isinstance(claim, str):
        raise ValueError("Claim must be a string")
    c = claim.strip().rstrip(".?!").strip()
    if not c or not any(ch.isalnum() for ch in c):
        raise ValueError("Claim must contain non-empty alphanumeric content")

    if llm_fn is not None:
        return list(llm_fn(c, list(top_sentences or ()), stance))

    embedded = _format_embedded_claim(c)
    if not top_sentences:
        return [
            f"What evidence makes you think that {embedded}?",
            f"What are you assuming must be true for '{c}' to hold?",
            f"What result would change your mind about '{c}'?",
        ]

    diff_terms = extract_differential_terms(c, top_sentences, max_terms=3)
    if not diff_terms:
        return [
            f"What evidence makes you think that {embedded}?",
            f"What are you assuming must be true for '{c}' to hold?",
            f"What result would change your mind about '{c}'?",
        ]

    focus = ", ".join(diff_terms)
    lead_term = diff_terms[0]
    polarity_hint = (
        f"contradicting observation around '{lead_term}'"
        if stance == "CONTRADICT"
        else f"mechanistic role of '{lead_term}'"
    )
    return [
        f"What direct evidence links {focus} to the claim that {embedded}?",
        f"What are you assuming about the {polarity_hint} ({focus}) for '{c}' to hold?",
        f"What controlled perturbation or measurement of {focus} would change your mind about '{c}'?",
    ]
