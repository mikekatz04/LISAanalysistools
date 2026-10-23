"""``lisatools_citations`` CLI: print the BibTeX references of LISA Analysis Tools modules.

Wired as ``lisatools_citations = lisatools.cmd.citations:main`` in ``pyproject.toml``.

.. code-block:: bash

    lisatools_citations                       # print all references
    lisatools_citations lisatools.detector    # print references for one module
    lisatools_citations lisatools.detector lisatools.analysiscontainer
                                              # print references for several modules

For each argument the CLI imports the module, walks its attributes, finds any
:class:`lisatools.utils.citations.Citable` subclasses, collects their
``module_references()``, deduplicates the keys and renders them as BibTeX from the
:func:`~lisatools.utils.citations.get_citation_registry` registry. A module with no
``Citable`` subclass falls back to the package reference (a note goes to stderr).

Pattern lifted from ``few.cmd.citations``; simpler implementation (no argparse subcommands,
no fancy formatting).
"""
from __future__ import annotations

import importlib
import inspect
import sys
from typing import Iterable, List, Set

from lisatools.utils.citations import Citable, REFERENCE, get_citation_registry


def _citables_in_module(modname: str) -> Iterable[type]:
    """Import ``modname`` and yield each distinct ``Citable`` subclass among its attributes.

    ``Citable`` itself is skipped. Import errors propagate to the caller.
    """
    mod = importlib.import_module(modname)
    seen: Set[type] = set()
    for name in dir(mod):
        obj = getattr(mod, name, None)
        if inspect.isclass(obj) and issubclass(obj, Citable) and obj is not Citable:
            if obj in seen:
                continue
            seen.add(obj)
            yield obj


def _collect_refs(modules: List[str]) -> List[str]:
    """Return the deduplicated reference keys (first-seen order) for the given modules.

    A module that fails to import is reported on stderr and skipped; a module with no
    ``Citable`` subclass contributes the package reference (``Citable``'s default).
    """
    keys: List[str] = []
    seen: Set[str] = set()
    for modname in modules:
        try:
            citables = list(_citables_in_module(modname))
        except (ImportError, ModuleNotFoundError) as e:
            print(f"# Could not import {modname}: {e}", file=sys.stderr)
            continue
        if not citables:
            print(
                f"# {modname}: no Citable subclasses found; printing package reference only.",
                file=sys.stderr,
            )
            citables = [Citable]
        for cls in citables:
            for k in cls.module_references():
                key = k.value if isinstance(k, REFERENCE) else str(k)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    return keys


def main(argv: List[str] | None = None) -> int:
    """Run the ``lisatools_citations`` CLI and print BibTeX to stdout.

    Args:
        argv: Module names to collect references for. ``None`` (default) reads
            ``sys.argv[1:]``; an empty list prints every reference in the registry.

    Returns:
        int: Exit status, ``0`` on success, ``1`` when no reference key was found.

    Raises:
        KeyError: When a ``Citable`` subclass names a key absent from ``CITATION.cff``.
        FileNotFoundError: When ``CITATION.cff`` cannot be located.

    Example:
        .. code-block:: bash

            lisatools_citations lisatools.analysiscontainer
    """
    if argv is None:
        argv = sys.argv[1:]

    registry = get_citation_registry()
    if not argv:
        # print everything
        print("\n\n".join(r.to_bibtex() for r in registry.all()))
        return 0

    keys = _collect_refs(argv)
    if not keys:
        print("# No references found.", file=sys.stderr)
        return 1

    print("\n\n".join(registry.get(k).to_bibtex() for k in keys))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
