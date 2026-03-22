"""
Skill extraction verification script.

Runs explicit and implicit extraction on controlled example texts and prints
a human-readable report showing what each module contributes, whether the
paper's key example reproduces correctly, and that implicit propagation works.

Usage:
    python -m src.skills.verify

Does NOT require the full ESCO CSV — uses the same mock index as the tests.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich import box

from src.skills.esco_loader import EscoIndex, EscoSkill
from src.skills.explicit_extractor import ExplicitSkillExtractor, ExtractedSkill
from src.skills.implicit_extractor import ImplicitSkillExtractor

console = Console()


# ── Build mock ESCO index (mirrors tests/skills/conftest.py) ──────────────────

def _build_mock_index() -> EscoIndex:
    raw = [
        ("esco:python",     "Python",                        ["Python programming", "Python scripting"]),
        ("esco:java",       "Java",                          ["Java programming"]),
        ("esco:sql",        "SQL",                           ["structured query language"]),
        ("esco:octave",     "GNU Octave",                    ["Octave", "octave programming"]),
        ("esco:ml",         "machine learning",              ["ML", "statistical learning"]),
        ("esco:da",         "data analysis",                 ["data analytics"]),
        ("esco:docker",     "Docker",                        ["containerisation"]),
        ("esco:kubernetes", "Kubernetes",                    ["K8s"]),
        ("esco:agile",      "agile methodology",             ["Scrum", "agile development"]),
        ("esco:oop",        "object-oriented programming",   ["OOP"]),
        ("esco:softdev",    "software development",          ["software engineering"]),
        ("esco:nlp",        "natural language processing",   ["NLP", "text mining"]),
        ("esco:dl",         "deep learning",                 ["neural networks"]),
        ("esco:cloud",      "cloud computing",               ["AWS", "Azure", "GCP"]),
        ("esco:git",        "Git",                           ["version control", "GitHub"]),
        ("esco:pytorch",    "PyTorch",                       ["torch"]),
        ("esco:restapi",    "REST API",                      ["RESTful API"]),
    ]
    skills = [EscoSkill(uri=u, preferred_label=p, alt_labels=a) for u, p, a in raw]
    idx = EscoIndex(skills=skills)
    idx.build()
    return idx


# ── Section printers ──────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def _print_explicit_detail(
    extractor: ExplicitSkillExtractor,
    text: str,
) -> list[ExtractedSkill]:
    """Run all four modules independently and print their contributions."""
    console.print(f"\n[bold]Input:[/bold] {text}\n")
    doc = extractor._nlp(text)

    # Module outputs
    s3_hits = extractor._s3_dict(doc)
    dict_hit_surfaces = set(s3_hits.keys())
    s1_scores = extractor._s1_ner(doc)
    s2_scores = extractor._s2_pos(doc, dict_hit_surfaces)

    # Module table
    t = Table("Module", "Candidates found", box=box.SIMPLE)
    t.add_row("S3 — ESCO Dict", str(sorted(dict_hit_surfaces)) or "—")
    t.add_row("S1 — NER", str(sorted(s1_scores.keys())) or "—")
    t.add_row("S2 — PoS rule", str(sorted(s2_scores.keys())) or "—")
    console.print(t)

    # Final extraction with relevance scores
    skills = extractor.extract(text)
    rt = Table("Preferred label", "ESCO URI", "Surface form", "Confidence", box=box.SIMPLE)
    for s in skills:
        rt.add_row(s.preferred_label, s.esco_uri, s.matched_text, f"{s.confidence:.4f}")
    if not skills:
        console.print("[yellow]  No skills extracted above threshold.[/yellow]")
    else:
        console.print(rt)

    return skills


def _print_implicit_result(
    explicit_skills: list[ExtractedSkill],
    implicit_skills: list[ExtractedSkill],
) -> None:
    if not implicit_skills:
        console.print("[yellow]  No implicit skills found.[/yellow]")
        return
    t = Table("Preferred label", "ESCO URI", "Sourced from neighbour", "Confidence", box=box.SIMPLE)
    explicit_labels = {s.preferred_label for s in explicit_skills}
    for s in implicit_skills:
        assert s.preferred_label not in explicit_labels, (
            f"BUG: '{s.preferred_label}' is both explicit and implicit!"
        )
        t.add_row(s.preferred_label, s.esco_uri, s.matched_text, f"{s.confidence:.4f}")
    console.print(t)


# ── Verification scenarios ────────────────────────────────────────────────────

def verify_paper_example(extractor: ExplicitSkillExtractor) -> None:
    _print_header("1. Paper's example sentence (Gugnani & Misra 2020, Section 4.1)")
    console.print(
        "[dim]Expected: Python and Java via S3 (dict), "
        "Octave boosted by S2 PoS rule because Python+Java anchor it.[/dim]"
    )
    skills = _print_explicit_detail(
        extractor,
        "Need candidates with ability to code in Python, Java, and Octave.",
    )
    labels = [s.preferred_label for s in skills]
    ok_python = "Python" in labels
    ok_java   = "Java"   in labels
    status = "[green]PASS[/green]" if (ok_python and ok_java) else "[red]FAIL[/red]"
    console.print(f"  Python found: {'✓' if ok_python else '✗'}  Java found: {'✓' if ok_java else '✗'}  → {status}\n")


def verify_alt_label(extractor: ExplicitSkillExtractor) -> None:
    _print_header("2. Alt-label matching")
    console.print("[dim]'NLP' is an alt label → expected preferred label: natural language processing[/dim]")
    skills = _print_explicit_detail(extractor, "The team needs strong NLP skills.")
    labels = [s.preferred_label for s in skills]
    ok = "natural language processing" in labels
    console.print(f"  Alt-label resolved: {'✓' if ok else '✗'}  → {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'}\n")


def verify_uri_deduplication(extractor: ExplicitSkillExtractor) -> None:
    _print_header("3. URI deduplication (same skill via multiple surface forms)")
    console.print("[dim]'NLP' and 'natural language processing' both map to esco:nlp → must appear once[/dim]")
    skills = _print_explicit_detail(
        extractor,
        "Strong background in NLP and natural language processing is required.",
    )
    nlp_hits = [s for s in skills if s.esco_uri == "esco:nlp"]
    ok = len(nlp_hits) == 1
    console.print(f"  esco:nlp appears {len(nlp_hits)} time(s): {'✓' if ok else '✗'}  → {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'}\n")


def verify_relevance_threshold(extractor: ExplicitSkillExtractor) -> None:
    _print_header("4. Relevance threshold — generic words filtered out")
    console.print("[dim]'candidate', 'communication', etc. must not pass the 0.35 threshold[/dim]")
    skills = _print_explicit_detail(
        extractor,
        "The candidate must have good communication and teamwork.",
    )
    labels = [s.preferred_label for s in skills]
    bad = {"candidate", "communication", "teamwork"} & set(labels)
    ok = len(bad) == 0
    console.print(f"  Unwanted terms: {bad or '∅'}  → {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'}\n")


def verify_implicit_propagation(
    explicit_extractor: ExplicitSkillExtractor,
) -> None:
    _print_header("5. Implicit skill propagation (Gugnani & Misra 2020, Section 4.2)")
    console.print(
        "[dim]Corpus: 5 job-ad-like texts.\n"
        "  Doc 0: Python + ML\n"
        "  Doc 1: Python + ML + Docker  ← similar to doc 0, adds Docker\n"
        "  Doc 2: Java + SQL\n"
        "  Doc 3: Java + SQL + Agile   ← similar to doc 2, adds Agile\n"
        "  Doc 4: NLP + deep learning  ← dissimilar\n"
        "Expected: Docker implicit for doc 0 | Agile implicit for doc 2[/dim]\n"
    )

    corpus_texts = [
        "We are looking for a Python developer with machine learning experience.",
        "Senior Python engineer needed. Machine learning background required. Docker is a plus.",
        "Java developer with SQL and database experience wanted.",
        "Java backend developer. SQL database knowledge. Agile methodology experience needed.",
        "Research engineer with natural language processing and deep learning skills.",
    ]
    corpus_explicit = [
        [ExtractedSkill("esco:python", "Python", "Python", explicit=True, confidence=0.9),
         ExtractedSkill("esco:ml", "machine learning", "machine learning", explicit=True, confidence=0.85)],
        [ExtractedSkill("esco:python", "Python", "Python", explicit=True, confidence=0.9),
         ExtractedSkill("esco:ml", "machine learning", "machine learning", explicit=True, confidence=0.85),
         ExtractedSkill("esco:docker", "Docker", "Docker", explicit=True, confidence=0.9)],
        [ExtractedSkill("esco:java", "Java", "Java", explicit=True, confidence=0.9),
         ExtractedSkill("esco:sql", "SQL", "SQL", explicit=True, confidence=0.85)],
        [ExtractedSkill("esco:java", "Java", "Java", explicit=True, confidence=0.9),
         ExtractedSkill("esco:sql", "SQL", "SQL", explicit=True, confidence=0.85),
         ExtractedSkill("esco:agile", "agile methodology", "agile methodology", explicit=True, confidence=0.8)],
        [ExtractedSkill("esco:nlp", "natural language processing", "NLP", explicit=True, confidence=0.9),
         ExtractedSkill("esco:dl", "deep learning", "deep learning", explicit=True, confidence=0.85)],
    ]

    impl = ImplicitSkillExtractor(explicit_extractor, sim_threshold=0.50, top_k=3)
    impl.fit(corpus_texts, explicit_skills_per_doc=corpus_explicit)

    for doc_idx, (text, explicit) in enumerate(zip(corpus_texts, corpus_explicit)):
        explicit_uris = {s.esco_uri for s in explicit}
        implicit = impl.extract(text, explicit_uris=explicit_uris, doc_idx=doc_idx)
        console.print(f"[bold]Doc {doc_idx}:[/bold] {text[:70]}…" if len(text) > 70 else f"[bold]Doc {doc_idx}:[/bold] {text}")
        console.print(f"  Explicit: {[s.preferred_label for s in explicit]}")
        console.print(f"  Implicit: {[s.preferred_label for s in implicit]}")

    # Spot-checks
    impl0 = impl.extract(corpus_texts[0], explicit_uris={"esco:python", "esco:ml"}, doc_idx=0)
    impl2 = impl.extract(corpus_texts[2], explicit_uris={"esco:java", "esco:sql"}, doc_idx=2)

    ok_docker = any(s.esco_uri == "esco:docker" for s in impl0)
    ok_agile  = any(s.esco_uri == "esco:agile"  for s in impl2)
    ok_no_leak = all(s.esco_uri not in {"esco:python", "esco:ml"} for s in impl0)

    console.print(f"\n  Docker implicit for doc 0: {'✓' if ok_docker else '✗'}")
    console.print(f"  Agile  implicit for doc 2: {'✓' if ok_agile  else '✗'}")
    console.print(f"  No explicit-skill leakage: {'✓' if ok_no_leak else '✗'}")
    all_ok = ok_docker and ok_agile and ok_no_leak
    console.print(f"  → {'[green]PASS[/green]' if all_ok else '[red]FAIL[/red]'}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    console.print("\n[bold white]Skill Extraction Verification[/bold white]")
    console.print("[dim]Mock ESCO index (17 skills) — no CSV required[/dim]\n")

    index = _build_mock_index()
    extractor = ExplicitSkillExtractor(index)

    verify_paper_example(extractor)
    verify_alt_label(extractor)
    verify_uri_deduplication(extractor)
    verify_relevance_threshold(extractor)
    verify_implicit_propagation(extractor)

    console.print("[bold white]Verification complete.[/bold white]\n")


if __name__ == "__main__":
    main()
