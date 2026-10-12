"""Smaller embedded PNGs for the monitor pages.

Every figure on both pages is a base64 data URI, so PNG bytes ARE page bytes,
and the full page sits against the 16 MB artifact budget. Matplotlib writes
32-bit RGBA; these plots carry a few hundred distinct colours at most (a
colormap is 256 entries), so an 8-bit palette PNG is visually the same figure
at about a third of the size (9mo page, 2026-10-11: 5.5 MB of PNG -> 1.6 MB).

MAXCOVERAGE quantization is used because it bounds the worst pixel: on the
9mo page's time-frequency heatmap it moves no channel by more than 2/255,
where FASTOCTREE moved channels by up to 12/255.

``GF_MONITOR_PNG_QUANTIZE=0`` turns it off. Any failure, or a palette PNG that
comes out larger, returns the original bytes.
"""

from __future__ import annotations

import io
import os

__all__ = ["shrink_png"]


def shrink_png(raw: bytes) -> bytes:
    """``raw`` (a PNG) re-encoded as a 256-colour palette PNG, when smaller."""
    if os.environ.get("GF_MONITOR_PNG_QUANTIZE", "1").strip() in ("0", "false", "no", "off"):
        return raw
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(raw)).convert("RGB")
        q = im.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE,
                        dither=Image.Dither.NONE)
        buf = io.BytesIO()
        q.save(buf, format="PNG", optimize=True)
        out = buf.getvalue()
    except Exception:  # noqa: BLE001 -- a smaller page is never worth a lost figure
        return raw
    return out if len(out) < len(raw) else raw
