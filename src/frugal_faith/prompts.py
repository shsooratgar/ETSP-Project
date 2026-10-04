"""Prompt templates.

RQ1-RQ3 use ``direct``. The variants exist to quantify prompt sensitivity on a
subsample (see the proposal); they are not run over the full benchmark.
"""

from __future__ import annotations

from typing import Callable

#: Label words scored against each other. Index 0 is "inconsistent", 1 "consistent",
#: matching the SummaC convention where 1 means the summary is faithful.
LABEL_WORDS: tuple[str, str] = ("No", "Yes")

PromptFn = Callable[[str, str], list[dict[str, str]]]


def _direct(document: str, summary: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"Document:\n{document}\n\n"
                f"Summary:\n{summary}\n\n"
                "Is every statement in the summary supported by the document? "
                "Answer Yes or No."
            ),
        }
    ]


def _entailment(document: str, summary: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"Premise:\n{document}\n\nHypothesis:\n{summary}\n\n"
                "Does the premise entail the hypothesis? Answer Yes or No."
            ),
        }
    ]


def _fact_check(document: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful fact checker."},
        {
            "role": "user",
            "content": (
                f"Source:\n{document}\n\nClaim:\n{summary}\n\n"
                "Does the source fully support the claim, with no added or "
                "contradicted facts? Answer Yes or No."
            ),
        },
    ]


PROMPTS: dict[str, PromptFn] = {
    "direct": _direct,
    "entailment": _entailment,
    "fact_check": _fact_check,
}


def build(prompt_id: str, document: str, summary: str) -> list[dict[str, str]]:
    """Return chat messages for ``prompt_id``.

    Raises:
        KeyError: if ``prompt_id`` is not a known template.
    """
    if prompt_id not in PROMPTS:
        raise KeyError(f"Unknown prompt {prompt_id!r}; have {sorted(PROMPTS)}")
    return PROMPTS[prompt_id](document, summary)
