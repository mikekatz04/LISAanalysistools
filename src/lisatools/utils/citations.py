"""Lightweight citation registry for lisatools.

Lifted from FastEMRIWaveforms' citation pattern (`few.utils.citations`)
but **simpler**: no pydantic dep, no JSON schema validation. Just a
dataclass-based parser over ``CITATION.cff`` at the package root. Each
class that wants to declare references inherits from :class:`Citable`
and overrides :meth:`module_references`:

    from lisatools.utils.citations import Citable, REFERENCE

    class MyAnalysis(Citable):
        @classmethod
        def module_references(cls):
            return [REFERENCE.GLOBAL_FIT_KATZ_2024]

Then users can do ``MyAnalysis.citation()`` to get the rendered BibTeX
for those references, or run ``lisatools_citations lisatools.analysiscontainer``
from the shell.

If the citation pattern outgrows this minimal implementation (e.g. we
start needing per-author ORCID, multi-arxiv identifiers, software vs
article distinction), migrate to FEW's full pydantic-based version. The
:class:`Citable` mixin's public API is intentionally a subset of FEW's
so the migration is a no-op for callers.
"""
from __future__ import annotations

import dataclasses
import enum
import pathlib
from typing import Iterable, List, Optional, Union

try:
    import yaml
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    yaml = None  # type: ignore


@dataclasses.dataclass
class Author:
    given_names: str = ""
    family_names: str = ""
    orcid: str = ""
    affiliation: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Author":
        return cls(
            given_names=d.get("given-names", "") or d.get("given_names", ""),
            family_names=d.get("family-names", "") or d.get("family_names", ""),
            orcid=d.get("orcid", "") or "",
            affiliation=d.get("affiliation", "") or "",
        )

    def to_bibtex(self) -> str:
        if self.family_names and self.given_names:
            return f"{self.family_names}, {self.given_names}"
        return self.family_names or self.given_names


@dataclasses.dataclass
class Reference:
    """Minimal citation record. Maps to one entry under ``references:`` in CITATION.cff."""

    key: str
    title: str = ""
    type: str = "article"
    authors: List[Author] = dataclasses.field(default_factory=list)
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    arxiv: str = ""
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Reference":
        return cls(
            key=d["key"],
            title=d.get("title", ""),
            type=d.get("type", "article"),
            authors=[Author.from_dict(a) for a in d.get("authors", []) or []],
            year=d.get("year"),
            journal=d.get("journal", ""),
            doi=d.get("doi", ""),
            arxiv=d.get("arxiv", ""),
            note=d.get("note", ""),
        )

    def to_bibtex(self) -> str:
        """Render this reference as a BibTeX entry. Not RFC-perfect — humans-only."""
        kind = "@software" if self.type == "software" else "@article"
        lines = [f"{kind}{{{self.key},"]
        if self.title:
            lines.append(f"  title  = {{{self.title}}},")
        if self.authors:
            lines.append("  author = {" + " and ".join(a.to_bibtex() for a in self.authors) + "},")
        if self.year is not None:
            lines.append(f"  year   = {{{self.year}}},")
        if self.journal:
            lines.append(f"  journal = {{{self.journal}}},")
        if self.doi:
            lines.append(f"  doi    = {{{self.doi}}},")
        if self.arxiv:
            lines.append(f"  eprint = {{{self.arxiv}}},")
            lines.append("  archivePrefix = {arXiv},")
        if self.note:
            lines.append(f"  note   = {{{self.note}}},")
        lines.append("}")
        return "\n".join(lines)


class REFERENCE(str, enum.Enum):
    """Stable keys for the references in CITATION.cff.

    Add to this enum (matching the ``key`` in CITATION.cff) when introducing
    a new shared reference; downstream classes import the enum member
    instead of stringly-typed keys.
    """

    LISATOOLS_PACKAGE = "LISAToolsPackage"
    GLOBAL_FIT_KATZ_2024 = "GlobalFitKatz2024"
    WDM_HETERODYNE = "WDMheterodyne"


class CitationRegistry:
    """Container holding all references parsed from CITATION.cff."""

    def __init__(self, references: Iterable[Reference]):
        self._by_key = {r.key: r for r in references}

    def get(self, key: Union[str, REFERENCE]) -> Reference:
        key_str = key.value if isinstance(key, REFERENCE) else str(key)
        if key_str not in self._by_key:
            raise KeyError(
                f"Unknown citation key {key_str!r}; known keys: {sorted(self._by_key)}"
            )
        return self._by_key[key_str]

    def all(self) -> List[Reference]:
        return list(self._by_key.values())


_REGISTRY: Optional[CitationRegistry] = None


def _locate_citation_cff() -> pathlib.Path:
    """Find CITATION.cff. In editable installs it lives at the repo root; in
    a wheel install it ships under the package data dir.
    """
    import lisatools

    pkg_root = pathlib.Path(lisatools.__file__).parent
    # editable install: ../../../CITATION.cff (parent of `src/lisatools/`)
    for parent in [pkg_root.parent.parent, pkg_root.parent, pkg_root]:
        candidate = parent / "CITATION.cff"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"CITATION.cff not found near {pkg_root}; expected at repo root or "
        f"shipped with the wheel."
    )


def build_citation_registry() -> CitationRegistry:
    """Parse CITATION.cff and build the global registry."""
    if yaml is None:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to build the citation registry; "
            "pip install pyyaml"
        )
    citation_cff = _locate_citation_cff()
    with open(citation_cff, "rt") as fp:
        cff = yaml.safe_load(fp)

    refs: List[Reference] = []
    # Top-level "self" reference for the package
    refs.append(
        Reference(
            key=REFERENCE.LISATOOLS_PACKAGE.value,
            title=cff.get("title", "LISAanalysistools"),
            type="software",
            authors=[Author.from_dict(a) for a in cff.get("authors", []) or []],
            doi=", ".join(
                i.get("value", "") for i in (cff.get("identifiers") or []) if i.get("type") == "doi"
            ),
            note=cff.get("abstract", "").strip().split("\n")[0] if cff.get("abstract") else "",
        )
    )
    for entry in cff.get("references", []) or []:
        refs.append(Reference.from_dict(entry))
    return CitationRegistry(refs)


def get_citation_registry() -> CitationRegistry:
    """Return the lazily-built process-global citation registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_citation_registry()
    return _REGISTRY


class Citable:
    """Mixin for classes that declare module-specific references.

    Subclasses override :meth:`module_references` to return an iterable of
    :class:`REFERENCE` enum members (or their string-key equivalents). The
    base implementation returns just the package reference.
    """

    @classmethod
    def module_references(cls) -> Iterable[Union[REFERENCE, str]]:
        """Override to declare which references this class depends on."""
        return [REFERENCE.LISATOOLS_PACKAGE]

    @classmethod
    def citation(cls) -> str:
        """Return the module references as a printable BibTeX string."""
        registry = get_citation_registry()
        entries = [registry.get(key).to_bibtex() for key in cls.module_references()]
        return "\n\n".join(entries)

    @classmethod
    def all_citations(cls) -> str:
        """Return every reference in the registry as a BibTeX string."""
        registry = get_citation_registry()
        return "\n\n".join(r.to_bibtex() for r in registry.all())


__all__ = [
    "Author",
    "Reference",
    "REFERENCE",
    "CitationRegistry",
    "Citable",
    "build_citation_registry",
    "get_citation_registry",
]
