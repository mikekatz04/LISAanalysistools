"""Write ``lisaconstants_values.h``: the ``lisaconstants`` values for C/C++/CUDA.

C++ cannot import a Python package, so the native code of LAT, GBGPU, BBHx
and FEW reads the constants from a header generated FROM ``lisaconstants``
(Mike 2026-10-08: "update all the hard-coded places to use the lisaconstants
import"). Each repo commits its own copy next to the sources that include it,
so the include always resolves (several headers are copied between repos at
configure time) and no build step needs Python; a unit test in each repo
parses the committed copy and checks every value against the installed
``lisaconstants``, so a version bump or a hand edit cannot drift silently.

``lisaconstants`` ships its own header generator (``python -m lisaconstants
cpp``), but the 2.0.2 output does not compile (``LISA_EPOCH_TCB`` is emitted
as a bare date literal), hence this one: plain ``#define``s, usable from C,
C++ and CUDA device code alike, numeric constants only, each written with
``repr(float)`` so the double round-trips exactly.

Besides the package's own constants the header carries the few DERIVED values
the kernels use, computed here in Python with the same expressions as
:mod:`lisatools.utils.constants` (so C and Python agree to the last bit):
``LISACONSTANTS_MTSUN`` (GM_sun / c^3, s), ``LISACONSTANTS_MRSUN``
(GM_sun / c^2, m), ``LISACONSTANTS_AU_LIGHT_TIME`` (AU / c, s) and
``LISACONSTANTS_GPC_LIGHT_TIME`` (1e9 pc / c, s).

Usage::

    python -m lisatools.utils.lisaconstants_header PATH [PATH ...]
"""
from __future__ import annotations

import re
import sys

import lisaconstants as lc

PREFIX = "LISACONSTANTS_"

#: derived values: name -> (expression over lisaconstants, description)
DERIVED = {
    "MTSUN": (lambda: lc.SOLAR_MASS_PARAMETER / lc.SPEED_OF_LIGHT**3,
              "GM_sun / c^3 [s]"),
    "MRSUN": (lambda: lc.SOLAR_MASS_PARAMETER / lc.SPEED_OF_LIGHT**2,
              "GM_sun / c^2 [m]"),
    "AU_LIGHT_TIME": (lambda: lc.ASTRONOMICAL_UNIT / lc.SPEED_OF_LIGHT,
                      "AU / c [s]"),
    "GPC_LIGHT_TIME": (lambda: 1e9 * lc.PARSEC / lc.SPEED_OF_LIGHT,
                       "1 Gpc / c [s]"),
}


def numeric_constants() -> dict:
    """Return ``{NAME: float}`` for every numeric public constant of ``lisaconstants``.

    Public means an upper-case name not starting with ``_``; booleans and non-numeric values
    are skipped.
    """
    out = {}
    for name in dir(lc):
        if not name.isupper() or name.startswith("_"):
            continue
        val = getattr(lc, name)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        out[name] = float(val)
    return out


def expected_values() -> dict:
    """Return ``{MACRO: float}`` -- what a correct header defines.

    The ``LISACONSTANTS_``-prefixed package constants plus the :data:`DERIVED` values; the
    per-repo unit tests compare a committed header's :func:`parse` output against this.
    """
    vals = {PREFIX + k: v for k, v in numeric_constants().items()}
    vals.update({PREFIX + k: float(f()) for k, (f, _) in DERIVED.items()})
    return vals


def render() -> str:
    """Return the header text.

    An include-guarded header with ``LISACONSTANTS_VERSION`` (a string), one ``#define`` per
    numeric constant (sorted by name, written with ``repr(float)``) and the derived values.
    """
    lines = [
        "/* lisaconstants_values.h -- GENERATED, do not edit.",
        f" * Values of lisaconstants {lc.__version__} for C / C++ / CUDA code.",
        " * Regenerate: python -m lisatools.utils.lisaconstants_header <this file>",
        " * (lisatools/utils/lisaconstants_header.py); a unit test checks every",
        " * value against the installed lisaconstants. */",
        "#ifndef LISACONSTANTS_VALUES_H",
        "#define LISACONSTANTS_VALUES_H",
        "",
        f'#define {PREFIX}VERSION "{lc.__version__}"',
        "",
    ]
    for name, val in sorted(numeric_constants().items()):
        lines.append(f"#define {PREFIX}{name} {val!r}")
    lines += ["", "/* derived in Python from the values above */"]
    for name, (f, desc) in DERIVED.items():
        lines.append(f"#define {PREFIX}{name} {float(f())!r}  /* {desc} */")
    lines += ["", "#endif /* LISACONSTANTS_VALUES_H */", ""]
    return "\n".join(lines)


_DEFINE = re.compile(r"^#define\s+(" + PREFIX + r"\w+)\s+([-+0-9.eE]+)")


def parse(text: str) -> dict:
    """Return ``{MACRO: float}`` from a header's numeric ``#define`` lines.

    Only ``LISACONSTANTS_*`` macros with a numeric literal value are read (the version string
    is skipped).

    Args:
        text: Header file contents.
    """
    out = {}
    for line in text.splitlines():
        m = _DEFINE.match(line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main(argv=None) -> int:
    """Write the rendered header to every path given on the command line.

    With no path, prints the ``Usage::`` tail of the module docstring to stderr.

    Args:
        argv (list[str], optional): Output paths. ``None`` (default) reads ``sys.argv[1:]``.

    Returns:
        int: ``0`` after writing, ``2`` when no path was given.
    """
    # The usage message is the text after "Usage::" in the module docstring:
    # keep that section last there.
    paths = list(sys.argv[1:] if argv is None else argv)
    if not paths:
        print(__doc__.split("Usage::")[-1].strip(), file=sys.stderr)
        return 2
    text = render()
    for p in paths:
        with open(p, "w") as fh:
            fh.write(text)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
