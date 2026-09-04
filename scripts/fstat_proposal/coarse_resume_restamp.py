#!/usr/bin/env python
"""Resume-identity preflight for the v8 submit scripts, with an opt-in
DIAGNOSTIC restamp for a deliberate coarse -> fine likelihood switch.

The noise-model identity guard (engine ``run.py`` + this cheap preflight)
normally refuses to resume a chain under a different coarse mode/Q: those
change the PSD/GALFOR *transition kernel* on identical array shapes, so
splicing a coarse-sampled chain onto a fine-likelihood resume would corrupt
the posterior. For the coarse-vs-fine drift TEST we deliberately want to
continue an existing chain under the exact-fine likelihood and watch whether
the sampled noise params move (if they do not, the coarse likelihood is
cleared of biasing them).

With ``COARSE_RESTAMP_IDENTITY=1`` this rewrites ONLY the ``coarse_*`` keys of
the stored identity to the new config so the engine's resume guard passes.
The real data-model keys (``unequal_arm``, ``wdm_psd_method``) stay strict --
those genuinely change the target and are never restamped. COPY THE STORE
FIRST: this mutates the identity record in place.
"""
import os
import sys

import h5py

# The only keys a coarse->fine switch is allowed to move. Everything else is
# a real likelihood change and must still refuse.
COARSE_KEYS = ("coarse_mode", "coarse_Q")


def _decode(v):
    """h5py attr -> plain python (mirror hdfbackend.read_noise_model_identity)."""
    if isinstance(v, bytes):
        return v.decode()
    return v


def read_identity(store):
    """Return the stored noise-model identity dict, or ``None`` if absent."""
    with h5py.File(store, "r") as f:
        grp = f.get("global_fit")
        if grp is None:
            return None
        ident = grp.get("noise_model_identity")
        if ident is None:
            return None
        return {k: _decode(v) for k, v in ident.attrs.items()}


def compute_mismatches(a, *, want_unequal_arm, want_method, want_mode, want_q):
    """Keys where the stored identity ``a`` disagrees with the wanted config."""
    m = {}
    if bool(a.get("unequal_arm")) != bool(want_unequal_arm):
        m["unequal_arm"] = (a.get("unequal_arm"), want_unequal_arm)
    if str(a.get("wdm_psd_method", "")) != str(want_method):
        m["wdm_psd_method"] = (a.get("wdm_psd_method"), want_method)
    if str(a.get("coarse_mode", "")) != str(want_mode):
        m["coarse_mode"] = (a.get("coarse_mode"), want_mode)
    if int(a.get("coarse_Q", 1)) != int(want_q):
        m["coarse_Q"] = (a.get("coarse_Q"), want_q)
    return m


def restamp_coarse(store, *, want_mode, want_q):
    """Rewrite ONLY the coarse_* identity attrs to the new config, in place.

    Mirrors how the engine originally wrote them (str mode, int Q). Drops the
    now-meaningless coarse fiducial digest when coarse is being turned off.
    """
    with h5py.File(store, "r+") as f:
        sub = f["global_fit"]["noise_model_identity"]
        sub.attrs["coarse_mode"] = str(want_mode)
        sub.attrs["coarse_Q"] = int(want_q)
        if int(want_q) <= 1 and "coarse_fiducial_digest" in sub.attrs:
            del sub.attrs["coarse_fiducial_digest"]


def preflight(store, *, want_unequal_arm, want_method, want_mode, want_q,
              restamp_enabled):
    """Cheap resume-identity check; returns None on OK, raises SystemExit(2)
    on a refused mismatch. With ``restamp_enabled`` a coarse-ONLY mismatch is
    rewritten in place instead of refused."""
    a = read_identity(store)
    if a is None:
        print(f"[V8-NOISE] REFUSING: {store!r} predates noise-model identity "
              "records -- it cannot have been sampled under the v8 noise "
              "model. Use a fresh STORE_DIR.")
        raise SystemExit(2)
    m = compute_mismatches(
        a, want_unequal_arm=want_unequal_arm, want_method=want_method,
        want_mode=want_mode, want_q=want_q,
    )
    if restamp_enabled and m and set(m).issubset(set(COARSE_KEYS)):
        restamp_coarse(store, want_mode=want_mode, want_q=want_q)
        print(f"[V8-NOISE] RESTAMPED coarse identity in {store!r} (was {m}): "
              "deliberate coarse->fine resume for the drift test; data-model "
              "keys unchanged. (The store should be a COPY.)")
        m = {}
    if m:
        print(f"[V8-NOISE] REFUSING: stored noise identity does not match this "
              f"config (stored, wanted): {m}. Full stored identity: {a}. "
              "Use a fresh STORE_DIR (or COARSE_RESTAMP_IDENTITY=1 for a "
              "deliberate coarse-only switch).")
        raise SystemExit(2)
    print(f"[V8-NOISE] resume identity OK: {read_identity(store)}")


def main(argv=None):
    """CLI: ``coarse_resume_restamp.py <store.h5>``; config from env."""
    argv = list(sys.argv[1:] if argv is None else argv)
    store = argv[0]
    if not os.path.exists(store):
        return  # fresh run: the engine writes the identity, nothing to check
    preflight(
        store,
        want_unequal_arm=(os.environ.get("UNEQUAL_ARM", "0") == "1"),
        want_method=os.environ.get("WDM_PSD_METHOD", ""),
        want_mode=os.environ.get("COARSE_GPU_MODE", "delayed_acceptance"),
        want_q=int(os.environ.get("COARSE_Q", "8")),
        restamp_enabled=(os.environ.get("COARSE_RESTAMP_IDENTITY", "0") == "1"),
    )


if __name__ == "__main__":
    main()
