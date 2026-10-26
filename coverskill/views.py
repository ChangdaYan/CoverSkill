from __future__ import annotations
import re


def clean(value):
    return " ".join(str(value or "").strip().split())


_FACET_KEYWORDS = {
    "interface": (
        "input",
        "output",
        "parameter",
        "argument",
        "api",
        "endpoint",
        "schema",
        "config",
        "auth",
    ),
    "procedure": (
        "usage",
        "workflow",
        "step",
        "example",
        "how to",
        "process",
        "pipeline",
        "implement",
    ),
    "constraints": (
        "limit",
        "constraint",
        "security",
        "guardrail",
        "error",
        "warning",
        "require",
        "must",
        "avoid",
    ),
}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_+.#/-]+", text.casefold())
        if len(token) > 2
    }


def semantic_facets(skill: dict) -> dict[str, str]:
    """Build interpretable Markdown facets instead of arbitrary token prefixes."""
    name = clean(skill.get("name"))
    description = clean(skill.get("description"))
    body = str(skill.get("skill_md") or skill.get("body") or "")
    identity = clean(f"{name} | {description}")
    buckets: dict[str, list[str]] = {key: [] for key in _FACET_KEYWORDS}
    current = "procedure"
    for raw in body.splitlines():
        line = clean(raw)
        if not line:
            continue
        lowered = line.casefold()
        if raw.lstrip().startswith("#"):
            matches = [
                facet
                for facet, words in _FACET_KEYWORDS.items()
                if any(word in lowered for word in words)
            ]
            current = matches[0] if matches else "procedure"
        direct = [
            facet
            for facet, words in _FACET_KEYWORDS.items()
            if any(word in lowered for word in words)
        ]
        bucket = direct[0] if direct else current
        if len(" ".join(buckets[bucket]).split()) < 640:
            buckets[bucket].append(line)
    facets = {"identity": identity}
    for facet, lines in buckets.items():
        text = clean(f"{name} | {description} | {' '.join(lines)}")
        if text and text != identity:
            facets[facet] = text
    return facets


def routed_semantic_facet(skill: dict, routing_text: str) -> tuple[str, str]:
    return route_semantic_facets(semantic_facets(skill), routing_text)


def route_semantic_facets(facets: dict[str, str], routing_text: str) -> tuple[str, str]:
    candidates = [(name, text) for name, text in facets.items() if name != "identity"]
    if not candidates:
        return "identity", facets["identity"]
    route_terms = _terms(routing_text)
    ranked = []
    for name, text in candidates:
        facet_terms = _terms(text)
        overlap = len(route_terms & facet_terms) / max(1, len(route_terms))
        ranked.append((overlap, name, text))
    _, name, text = max(ranked, key=lambda row: (row[0], row[1]))
    return name, text


FACET_KEYS = ("identity", "procedure", "interface", "constraints")


def four_facets(skill):
    """Same lexical construction for train and deployment; no query or qrels."""
    views = semantic_facets(skill)
    # Empty buckets retain the identity plus trailing separator, as in the run.
    name = clean(skill.get("name"))
    description = clean(skill.get("description"))
    fallback = clean(f"{name} | {description} | ")
    return [views.get(key, fallback) for key in FACET_KEYS]
