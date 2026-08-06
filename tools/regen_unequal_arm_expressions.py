"""Vendor segwo's sympy-generated unequal-arm XYZ covariance into LAT.

Mechanical transform of /Users/rjrosati/code/segwo/notebooks/xyzcov.py:
  * hard-coded A_oms / A_tm become the LISAModel SQUARED levels Soms_d / Sa_a
    (every occurrence in the source is `A_oms**2` / `A_tm**2`, verified, so the
    substitution is exact -- no sqrt round trip),
  * the free variable `c` (undefined in the source) is bound to C_SI,
  * the shape knees f_oms / f_tm1 / f_tm2 stay as generated.
"""
import re, textwrap

SRC = "/Users/rjrosati/code/segwo/notebooks/xyzcov.py"
DST = ("/Users/rjrosati/code/lisa_sprint_2026_clean/LISAanalysistools/src/"
       "lisatools/_unequal_arm_expressions.py")

src = open(SRC).read()
fns = re.findall(r"def (noise_cov_(\w+))\((.*?)\):\n(.*?)(?=\ndef |\Z)", src, re.S)
assert len(fns) == 6, len(fns)

header = '''"""Unequal-arm TDI-2 XYZ instrument-noise covariance elements (GENERATED).

DO NOT EDIT BY HAND. These closed forms were produced with sympy from the
six-link TDI-2 Michelson combinations, carrying every light-travel time
independently -- no equal-arm or averaged-arm assumption. Vendored from
``segwo/notebooks/xyzcov.py``; see ``tools/regen_unequal_arm_expressions.py``
for the transform applied on import into this tree.

Conventions (all verified numerically in ``tests/test_unequal_arm_noise.py``):

* ``d_ij`` is the light travel time in seconds along the link **from**
  spacecraft ``j`` **to** spacecraft ``i``, matching
  :attr:`lisatools.detector.LINKS` element ``ij``. The six are independent, so
  the Sagnac splitting ``d_ij != d_ji`` is represented exactly.
* ``Soms_d`` and ``Sa_a`` are the **squared** OMS / acceleration levels, i.e.
  the values :class:`lisatools.detector.LISAModel` stores (``(15e-12)**2`` and
  ``(3e-15)**2`` for ``scirdv1``). Each expression is exactly linear in the
  pair, which is what lets :class:`~lisatools.sensitivity.UnequalArmInstrumentNoise`
  cache two unit-level bases and recombine them per MCMC proposal.
* The result is a one-sided PSD in **relative frequency** units, the same
  convention as :meth:`lisatools.detector.LISAModel.lisanoises` with
  ``unit="relative_frequency"`` and as the stock ``X2TDISens`` family.
* Autocorrelations (XX/YY/ZZ) are real up to roundoff; cross-spectra
  (XY/XZ/YZ) are genuinely complex, with ``C_ji = conj(C_ij)``.

In the equal-arm limit these reduce to ``X2TDISens`` / ``XY2TDISens`` exactly
(ratio 1.0 to machine precision).
"""

import numpy as np

from .utils.constants import C_SI as c

__all__ = [
    "noise_cov_XX", "noise_cov_YY", "noise_cov_ZZ",
    "noise_cov_XY", "noise_cov_XZ", "noise_cov_YZ",
]

'''

out = [header]
for full, tag, args, body in fns:
    args = args.strip()
    # constants: drop the two amplitude assignments, keep the shape knees
    lines = [l for l in body.splitlines()
             if l.strip() and not re.match(r"\s*A_(oms|tm)\s*=", l)]
    body2 = "\n".join(lines)
    n_oms = body2.count("A_oms**2"); n_tm = body2.count("A_tm**2")
    assert n_oms and n_tm, (tag, n_oms, n_tm)
    # every A_oms / A_tm in the source appears squared -- assert before substituting
    assert not re.search(r"A_oms(?!\*\*2)", body2), tag
    assert not re.search(r"A_tm(?!\*\*2)", body2), tag
    body2 = body2.replace("A_oms**2", "Soms_d").replace("A_tm**2", "Sa_a")
    doc = (f'    """Unequal-arm TDI-2 ``C_{tag}`` in relative-frequency units.\n\n'
           f'    Args:\n'
           f'        f: Frequency array (Hz).\n'
           f'        d_12, d_21, d_13, d_31, d_23, d_32: Per-link light travel\n'
           f'            times (s); ``d_ij`` is the delay from S/C ``j`` to ``i``.\n'
           f'        Soms_d: Squared OMS displacement level (m^2), i.e.\n'
           f'            ``LISAModel.Soms_d``.\n'
           f'        Sa_a: Squared acceleration level, i.e. ``LISAModel.Sa_a``.\n\n'
           f'    Returns:\n'
           f'        ``C_{tag}`` evaluated on ``f``'
           + ('.' if tag in ("XX", "YY", "ZZ") else ' (complex).') + '\n    """\n')
    out.append(f"def {full}({args}, Soms_d, Sa_a):\n{doc}{body2}\n\n")

open(DST, "w").write("\n".join(out))
print("wrote", DST)
