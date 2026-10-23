"""Lightweight citation registry for lisatools.

Lifted from FastEMRIWaveforms' citation pattern (``few.utils.citations``)
but **simpler**: no pydantic dep, no JSON schema validation. Just a
dataclass-based parser over ``CITATION.cff`` (read with PyYAML; searched at the
repo root for editable installs, then next to the package). Each
class that wants to declare references inherits from :class:`Citable`
and overrides :meth:`Citable.module_references`:

.. code-block:: python

    from lisatools.utils.citations import Citable, REFERENCE

    class MyAnalysis(Citable):
        @classmethod
        def module_references(cls):
            return [REFERENCE.GLOBAL_FIT_KATZ_2024]

Then users can do ``MyAnalysis.citation()`` to get the rendered BibTeX
for those references, or run ``lisatools_citations lisatools.analysiscontainer``
from the shell (:mod:`lisatools.cmd.citations`). No package class subclasses
:class:`Citable` yet, so the CLI currently prints the package reference for any module.

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
    """One author of a :class:`Reference` (a CITATION.cff ``authors`` entry).

    Args:
        given_names: Given name(s). Default ``""``.
        family_names: Family name(s). Default ``""``.
        orcid: ORCID URL or id. Default ``""``.
        affiliation: Affiliation. Default ``""``.
    """

    given_names: str = ""
    family_names: str = ""
    orcid: str = ""
    affiliation: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Author":
        """Build from a CITATION.cff author mapping.

        Reads ``given-names`` / ``family-names`` (or the underscore spellings), ``orcid`` and
        ``affiliation``; missing keys become ``""``.

        Args:
            d: One entry of a CITATION.cff ``authors`` list.

        Returns:
            Author: The parsed author.
        """
        return cls(
            given_names=d.get("given-names", "") or d.get("given_names", ""),
            family_names=d.get("family-names", "") or d.get("family_names", ""),
            orcid=d.get("orcid", "") or "",
            affiliation=d.get("affiliation", "") or "",
        )

    def to_bibtex(self) -> str:
        """Return the BibTeX author form: ``"Family, Given"``, or whichever part is set."""
        if self.family_names and self.given_names:
            return f"{self.family_names}, {self.given_names}"
        return self.family_names or self.given_names


@dataclasses.dataclass
class Reference:
    """Minimal citation record. Maps to one entry under ``references:`` in CITATION.cff.

    Args:
        key: Registry key (the CITATION.cff ``key``; see :class:`REFERENCE`).
        title: Title. Default ``""``.
        type: ``"software"`` renders as ``@software``, anything else as ``@article``. Default
            ``"article"``.
        authors: Author list. Default empty.
        year: Publication year, or ``None`` (omitted from the BibTeX).
        journal: Journal. Default ``""``.
        doi: DOI. Default ``""``.
        arxiv: arXiv identifier, rendered as ``eprint`` with ``archivePrefix = {arXiv}``.
            Default ``""``.
        note: Free-text note. Default ``""``.
    """

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
        """Build from one CITATION.cff ``references`` mapping.

        Args:
            d: The mapping; ``key`` is required, every other field is optional.

        Returns:
            Reference: The parsed reference.

        Raises:
            KeyError: When ``d`` has no ``key``.
        """
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
    """Container holding all references parsed from CITATION.cff, indexed by key.

    Args:
        references: The references; a later duplicate key replaces an earlier one.
    """

    def __init__(self, references: Iterable[Reference]):
        self._by_key = {r.key: r for r in references}

    def get(self, key: Union[str, REFERENCE]) -> Reference:
        """Return the reference stored under ``key``.

        Args:
            key: A :class:`REFERENCE` member or its string value.

        Returns:
            Reference: The matching reference.

        Raises:
            KeyError: When the key is unknown (the message lists the known keys).
        """
        key_str = key.value if isinstance(key, REFERENCE) else str(key)
        if key_str not in self._by_key:
            raise KeyError(
                f"Unknown citation key {key_str!r}; known keys: {sorted(self._by_key)}"
            )
        return self._by_key[key_str]

    def all(self) -> List[Reference]:
        """Return every reference, in insertion order (package reference first)."""
        return list(self._by_key.values())


_REGISTRY: Optional[CitationRegistry] = None


def _locate_citation_cff() -> pathlib.Path:
    """Find CITATION.cff. In editable installs it lives at the repo root; in
    a wheel install it ships under the package data dir.

    Checks, in order, two levels above the package directory (the repo root of a ``src/``
    layout), one level above, and the package directory itself.

    Raises:
        FileNotFoundError: When none of the three locations holds the file.
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
    """Parse CITATION.cff and build a citation registry.

    The first entry is the package itself (key ``LISAToolsPackage``, type ``software``,
    built from the top-level ``title`` / ``authors`` / DOI ``identifiers`` / first line of
    ``abstract``), followed by every ``references`` entry. Does not cache; see
    :func:`get_citation_registry`.

    Returns:
        CitationRegistry: The parsed registry.

    Raises:
        ImportError: When PyYAML is not installed.
        FileNotFoundError: When CITATION.cff cannot be located.
    """
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
    """Return the lazily-built process-global citation registry.

    Built by :func:`build_citation_registry` on the first call and cached in the module.

    Returns:
        CitationRegistry: The shared registry.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_citation_registry()
    return _REGISTRY


class Citable:
    """Mixin for classes that declare module-specific references.

    Subclasses override :meth:`module_references` to return an iterable of
    :class:`REFERENCE` enum members (or their string-key equivalents). The
    base implementation returns just the package reference.

    Example:
        .. code-block:: python

            from lisatools.utils.citations import Citable, REFERENCE

            class MyAnalysis(Citable):
                @classmethod
                def module_references(cls):
                    return [REFERENCE.LISATOOLS_PACKAGE, REFERENCE.GLOBAL_FIT_KATZ_2024]

            print(MyAnalysis.citation())
    """

    @classmethod
    def module_references(cls) -> Iterable[Union[REFERENCE, str]]:
        """Override to declare which references this class depends on.

        Returns:
            Iterable[REFERENCE | str]: Registry keys. The base returns
            ``[REFERENCE.LISATOOLS_PACKAGE]``.
        """
        return [REFERENCE.LISATOOLS_PACKAGE]

    @classmethod
    def citation(cls) -> str:
        """Return the module references as a printable BibTeX string.

        Returns:
            str: One BibTeX entry per key of :meth:`module_references`, blank-line separated.

        Raises:
            KeyError: When a key is absent from the registry.
        """
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
