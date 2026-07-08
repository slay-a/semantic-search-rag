"""Advanced prompt engineering for grounded generation.

This module is where retrieval turns into a well-structured prompt. The design
choices here are the difference between a RAG system that hallucinates and one
that stays faithful to its sources:

1. **A strong, cache-stable system prompt** that fixes the model's role, its
   grounding contract, and its citation format. It is byte-identical across
   requests so it can be prompt-cached (see the ``cache_control`` breakpoint in
   ``generator.py``).
2. **Numbered, delimited context blocks** — each retrieved chunk is wrapped in
   an XML-style tag with an explicit ``[n]`` marker and its source path. Claude
   follows XML-delimited structure reliably, and the markers give us a citation
   handle to resolve afterwards.
3. **An explicit "insufficient context" escape hatch** so the model abstains
   instead of inventing an answer when retrieval comes back thin.
4. **Query transformation (HyDE)** — an optional step that rewrites a terse
   question into a hypothetical answer paragraph, which embeds closer to real
   passages than the bare question does.
"""

from __future__ import annotations

from .types import ScoredChunk

# Kept byte-stable on purpose: this is the cached prefix. Do not interpolate
# per-request values (dates, the question, source counts) into it.
SYSTEM_PROMPT = """\
You are a precise research assistant that answers strictly from a set of \
retrieved source passages supplied in the user message.

Follow these rules without exception:
- Ground every factual claim in the provided sources. Do not use outside \
knowledge, and do not speculate beyond what the passages support.
- Cite the sources you use inline with bracketed markers matching the passage \
numbers, e.g. "The service retries on 429 [2][5]." Place the marker \
immediately after the claim it supports.
- If the passages do not contain enough information to answer, say so plainly \
(begin with "I don't have enough information to answer that") and state what \
is missing. Never fabricate a citation.
- Prefer quoting exact figures, names, and identifiers from the sources over \
paraphrasing them.
- Be concise and direct. Lead with the answer; add only the supporting detail \
the sources justify.
- When sources conflict, surface the disagreement and cite each side rather \
than silently picking one."""


HYDE_SYSTEM_PROMPT = """\
You rewrite a user's question into a short, factual paragraph that a good \
source document would contain if it answered the question. Write 2-4 sentences \
in a neutral, encyclopedic tone as if excerpted from documentation. Do not \
answer conversationally, do not add caveats, and do not say you are unsure — \
just produce the hypothetical passage. Output only the paragraph."""


def format_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as numbered, XML-delimited source blocks."""
    blocks: list[str] = []
    for i, sc in enumerate(chunks, start=1):
        source = sc.chunk.source
        page = sc.chunk.metadata.get("page")
        loc = f"{source}" + (f" (page {page})" if page else "")
        blocks.append(
            f'<source index="{i}" path="{loc}" score="{sc.score:.3f}">\n'
            f"{sc.chunk.text.strip()}\n"
            f"</source>"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    """Assemble the per-request user turn: sources first, then the question.

    Sources come before the question so that, if you later add a cache
    breakpoint after the corpus, the volatile question stays outside the cached
    prefix. The closing instruction restates the grounding contract at the point
    of the ask, where it has the most steering effect.
    """
    if not chunks:
        context = "(no sources retrieved)"
    else:
        context = format_context(chunks)

    return (
        "Answer the question using only the sources below.\n\n"
        "<sources>\n"
        f"{context}\n"
        "</sources>\n\n"
        f"<question>{question.strip()}</question>\n\n"
        "Write the answer now. Cite sources inline as [n] using the source "
        "index numbers above. If the sources are insufficient, say so."
    )


def build_hyde_prompt(question: str) -> str:
    return (
        f"Question: {question.strip()}\n\n"
        "Write the hypothetical source paragraph that would answer it."
    )
