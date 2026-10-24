"""The flagship run table: ONE launcher structure, six observation times.

This file is LEVEL 2 of the three adjustment levels (``README.md``):

* every run      -> edit ``launcher.sh.in`` (the shared body);
* one run        -> edit that run's :class:`Run` below (a table value, an
                    ``overrides`` entry that rewrites one export line, or an
                    ``extra_exports`` line for a knob the base does not carry);
* one run, once  -> ``KNOB=value ./submit_gf_<run>.sh`` for any ``${K:-v}``
                    knob, or ``build.py <run> --set KNOB=value --out <file>``.

Then ``python scripts/flagship/build.py`` regenerates the six committed
``submit_gf_<run>.sh`` files; ``tests/test_flagship_launchers.py`` refuses a
committed launcher that differs from what the table + template build.

THE 9MO IS THE BASE (user ruling 2026-10-11): every run follows the 9mo v9
pattern (``scripts/fstat_proposal/submit_gf_9mo_v9.sh``, job 751) and differs
from it ONLY where the observation time or the parent run demands:

* ``TOBS_TARGET`` and the knobs that follow it (sig-het stride, edge crop,
  the three GB capacity knobs), the WDM grid named by the preflights;
* ``MBHB_IDS`` by the two-week rule (every catalogue MBHB merging before
  ``days + 14 d``; ``MBH_MERGER_TIME_BUFFER`` = 14 d everywhere);
* the PARENT (3mo -> 6mo -> 9mo -> 12mo -> 18mo -> 24mo): GB warm start, noise
  pin and source warm start all read the previous run's store; the 9mo keeps
  the LIVE 6mo v9 store it was launched from;
* the 3mo has no parent: no warm start / seed stage / noise pin, psd starts
  at the unequal-arm reference fit and galfor at the INFLATED fitted
  hyperbolic tangent (the no-parent ``#else`` branches of ``launcher.sh.in``).
"""
from __future__ import annotations

import dataclasses
import typing

#: The series tag (store directories, SLURM job names, log stems). The 9mo keeps
#: its live names (``gf_prod_9mo_v9``) through its explicit fields below.
SERIES = "flagship"
OUTPUT_ROOT = "/shared/data/global_fit_output"

#: One WDM layer = Nf 1440 x dt 2.5 s = 3600 s; Nt = Tobs / LAYER_S (exact at
#: every flagship Tobs; test_flagship_launchers checks it against
#: ``WDMSettings.adjust_to_even_bins`` with the stock wavelet bounds).
NF = 1440
DT = 2.5
LAYER_S = NF * DT
#: The sig-het sparse layer stride keeps the validated 36-h parity.
SIGHET_STRIDE_S = 36 * 3600
#: The sig-het build guard: EDGE_CROP_WAVELETS >= ceil(alpha / 2 * Nt) + 8 at
#: SIGHET_TUKEY_ALPHA = 0.01 (stock/erebor/gb.py::check_sighet_build_config).
SIGHET_TUKEY_ALPHA = 0.01
#: User rule 2026-10-07: keep any MBHB merging within two weeks of the data end.
MBH_MERGER_BUFFER_DAYS = 14

#: Merger times (days from the data start) of the 20 catalogue MBHBs: the
#: mojito-lite catalogue's TimeCoalescencePhenomTPHMSSBFrame, anchored on the
#: MBH session table of 2026-10-03 (ids 0-5, 7, 9, 10, 12, 15, 16, 18 agree to
#: 0.05 d; the rest follow the same offset).
MBHB_TC_DAYS: typing.Dict[int, float] = {
    18: 92.0, 5: 104.7, 16: 111.4, 2: 173.3, 12: 243.8, 7: 263.8, 9: 285.9,
    4: 286.4, 0: 300.4, 15: 318.0, 3: 336.8, 10: 369.5, 19: 405.9, 13: 481.3,
    6: 507.4, 17: 585.5, 8: 606.6, 1: 611.7, 11: 654.6, 14: 712.7,
}
#: EMRI plunge times (days from the data start; the catalogue's
#: TimeCoalescenceSSBFrame, same anchor): names the EMRIs that plunge INSIDE a
#: window (the direct-to-WDM plunge path, header note only).
EMRI_PLUNGE_DAYS: typing.Dict[int, float] = {
    0: 346.9, 1: 518.3, 2: 711.8, 3: 256.0, 4: 969.2, 5: 908.4, 6: 427.1, 7: 665.7,
}
EMRI_IDS = "0,1,2,3,4,5,6,7"
SOBHB_IDS = "0,1,2,3,4,5"


def mbhb_ids_for(days: int) -> str:
    """The two-week rule: every catalogue MBHB with t_c < days + 14 d, ascending."""
    return ",".join(str(i) for i in sorted(MBHB_TC_DAYS)
                    if MBHB_TC_DAYS[i] < days + MBH_MERGER_BUFFER_DAYS)


def sighet_guard_min_crop(nt: int) -> int:
    """The smallest EDGE_CROP_WAVELETS the sig-het build guard accepts at Nt."""
    import math
    return int(math.ceil(0.5 * SIGHET_TUKEY_ALPHA * nt)) + 8


@dataclasses.dataclass(frozen=True)
class Parent:
    """What a run reads from its predecessor (all three warm starts + the pin)."""
    name: str                 # "6mo" (the label in comments / the candidate scan)
    store_dirname: str        # "gf_prod_6mo_v9_4gpu" (under OUTPUT_ROOT)
    base_file_name: str       # "gf_prod_6mo"
    days: int
    mbhb_ids: str             # the PARENT's MBHB_IDS (its cold chain carries those)
    emri_ids: str = EMRI_IDS
    sobhb_ids: str = SOBHB_IDS

    @property
    def tobs(self) -> int:
        return self.days * 86400

    @property
    def seed_store(self) -> str:
        return f"{OUTPUT_ROOT}/{self.store_dirname}/{self.base_file_name}_testing.h5"

    @property
    def warm_npz(self) -> str:
        return f"{self.store_dirname}_refereed.npz"


@dataclasses.dataclass(frozen=True)
class Run:
    name: str                      # "9mo" -> submit_gf_9mo.sh
    days: int
    parent: typing.Optional[typing.Union[str, Parent]]   # a run name, a Parent, or None
    gb_nleaves_max: int
    gb_n_subbands: int             # PER GPU
    gb_rj_inmodel_chunk: int
    edge_crop_wavelets: int
    ngpus_default: int = 4
    store_dirname: typing.Optional[str] = None   # default gf_prod_<name>_<SERIES>
    job_name: typing.Optional[str] = None        # default gf<name>_<SERIES>
    log_stem: typing.Optional[str] = None        # default = job_name
    #: level 2: rewrite one export line of the base in place ({KNOB: value})
    overrides: typing.Dict[str, str] = dataclasses.field(default_factory=dict)
    #: level 2: whole lines appended at the RUN_EXTRA_EXPORTS hook
    extra_exports: typing.Tuple[str, ...] = ()
    #: free-text header lines for this run (provenance, decisions)
    notes: typing.Tuple[str, ...] = ()

    # ---- derived ----------------------------------------------------------
    @property
    def tobs(self) -> int:
        return self.days * 86400

    @property
    def nt(self) -> int:
        nt, rem = divmod(self.tobs, int(LAYER_S))
        if rem:
            raise ValueError(f"{self.name}: Tobs {self.tobs} is not a whole number of "
                             f"{LAYER_S:g}-s WDM layers")
        return nt

    @property
    def sighet_nt_layer(self) -> int:
        stride, rem = divmod(self.tobs, SIGHET_STRIDE_S)
        if rem or self.nt % stride:
            raise ValueError(f"{self.name}: Tobs {self.tobs} does not carry the 36-h "
                             "sig-het stride parity")
        return stride

    @property
    def mbhb_ids(self) -> str:
        return mbhb_ids_for(self.days)

    @property
    def store_dir(self) -> str:
        return f"{OUTPUT_ROOT}/{self.store_dirname or f'gf_prod_{self.name}_{SERIES}'}/"

    @property
    def base_file_name(self) -> str:
        return f"gf_prod_{self.name}"

    @property
    def resolved_job_name(self) -> str:
        return self.job_name or f"gf{self.name}_{SERIES}"

    @property
    def resolved_log_stem(self) -> str:
        return self.log_stem or self.resolved_job_name

    def as_parent(self) -> Parent:
        """This run, seen from the next one in the chain."""
        return Parent(name=self.name,
                      store_dirname=self.store_dir.rstrip("/").rsplit("/", 1)[-1],
                      base_file_name=self.base_file_name, days=self.days,
                      mbhb_ids=self.mbhb_ids)

    def resolved_parent(self) -> typing.Optional[Parent]:
        if self.parent is None:
            return None
        if isinstance(self.parent, Parent):
            return self.parent
        return RUNS[self.parent].as_parent()

    def check(self) -> None:
        need = sighet_guard_min_crop(self.nt)
        if self.edge_crop_wavelets < need:
            raise ValueError(f"{self.name}: EDGE_CROP_WAVELETS={self.edge_crop_wavelets} < "
                             f"the sig-het guard's {need} at Nt {self.nt}")
        _ = self.sighet_nt_layer
        if self.ngpus_default not in (2, 4):
            raise ValueError(f"{self.name}: the dispatch table supports NGPUS 2 or 4 "
                             f"(got {self.ngpus_default}); extend the case table in "
                             "launcher.sh.in first (level 1)")
        if self.parent is not None and not isinstance(self.parent, Parent) \
                and self.parent not in RUNS:
            raise ValueError(f"{self.name}: unknown parent run {self.parent!r}")


# ---- THE TABLE ----------------------------------------------------------------
# Capacity knobs (GB_NLEAVES_MAX / GB_N_SUBBANDS per GPU / GB_RJ_INMODEL_CHUNK):
#   3mo   10000 / 32768 / 65536  -- the 3mo v9 (2gpu) values, run healthy in
#                                   production (jobs 636-644); 32768 is safe
#                                   ONLY with the sig-het staging knobs the base
#                                   carries (fold cap 1 GiB, SETUP_BATCH 2048,
#                                   both MEMPOOL_FREE=1), which it does
#   6mo   15000 /  8192 / 32768  -- the 6mo v9 production values (job 738 ff.)
#   9mo   20000 /  4096 / 16384  -- the 1yr values verbatim (ruling 2026-10-07)
#   12mo  20000 /  4096 / 16384  -- the 1yr values (6mo-v8 -> 1yr-v8 derivation)
#   18mo  25000 /  2048 /  8192  -- the 24mo values, as the 9mo took the 1yr's
#   24mo  25000 /  2048 /  8192  -- GB_NLEAVES_MAX 25000 = the 23mo ruling;
#                                   2048 slots = the 23mo precedent (sig-het
#                                   stash product 478 x 2048 = 0.98e6 under the
#                                   23mo's 1.1e6 OK); chunk halves again for
#                                   byte parity (cells ~2x the 1yr bytes)
# EDGE_CROP_WAVELETS: >= ceil(0.005 Nt) + 8 (19 / 30 / 41 / 52 / 73 / 95);
#   20 is the validated 3mo code default, 60 the validated 6mo / 9mo / 1yr value,
#   90 / 120 keep ~1.4 % of the data cropped at 18 / 24 mo (as the 1yr).
RUNS: typing.Dict[str, Run] = {}


def _add(run: Run) -> Run:
    RUNS[run.name] = run
    return run


_add(Run(
    name="3mo", days=90, parent=None,
    gb_nleaves_max=10000, gb_n_subbands=32768, gb_rj_inmodel_chunk=65536,
    edge_crop_wavelets=20,
    notes=(
        "THE FIRST RUN OF THE CHAIN -- no parent (user ruling 2026-10-11: start at",
        "an inflated estimate of the fitted hyperbolic tangent). The no-parent",
        "branches of launcher.sh.in apply: no GB warm start, no gb_search_seed, no",
        "seed store / noise pin, psd starts at the unequal-arm reference fit and",
        "galfor at the INFLATED 90-d add-back fixed point (amp x1.25, fk x1.10),",
        "noise_search forced AFTER source_search (position=first), gb_search_1's",
        "noise sampled once to convergence at each cycle end (as gb_search_2),",
        "sources from exact truth. MBHB 5 merges at 104.7 d, 0.7 d past the",
        "two-week window, and is left out by the rule.",
    ),
))
_add(Run(
    name="6mo", days=180, parent="3mo",
    gb_nleaves_max=15000, gb_n_subbands=8192, gb_rj_inmodel_chunk=32768,
    edge_crop_wavelets=60,
    notes=(
        "Parent = the flagship 3mo (NOT the live gf_prod_6mo_v9_4gpu lineage, which",
        "was seeded from the 3mo v8 10-walker run).",
    ),
))
_add(Run(
    name="9mo", days=270,
    # THE LIVE RUN (job 751 ff.): the 6mo v9 production store it was launched
    # from, and its own store / job / log names -- byte-for-byte the launcher
    # scripts/fstat_proposal/submit_gf_9mo_v9.sh in everything bash executes
    # (tests/test_flagship_launchers.py::NineMonthIdentityTest).
    parent=Parent(name="6mo", store_dirname="gf_prod_6mo_v9_4gpu",
                  base_file_name="gf_prod_6mo", days=180, mbhb_ids="2,5,16,18"),
    gb_nleaves_max=20000, gb_n_subbands=4096, gb_rj_inmodel_chunk=16384,
    edge_crop_wavelets=60,
    store_dirname="gf_prod_9mo_v9", job_name="gf9mo_v9",
    notes=(
        "THE BASE OF THE SERIES: the launched 9mo v9 (job 751, 2026-10-09; store",
        "gf_prod_9mo_v9, parent = the LIVE 6mo v9 store gf_prod_6mo_v9_4gpu). This",
        "file behaves exactly as scripts/fstat_proposal/submit_gf_9mo_v9.sh, and the",
        "test suite refuses any executable-line drift between the two.",
    ),
))
_add(Run(
    name="12mo", days=360,
    # The parent is a COPY of the 9mo run folder (Mike 2026-10-13: "i will copy
    # the 9mo to *_copy folder and we should run off that so we do not mess
    # that up"): every read -- the GB warm-start fit, the psd pin, the source
    # warm start, the monitor overlay -- goes to gf_prod_9mo_v9_copy, never to
    # the live gf_prod_9mo_v9 store the 9mo job keeps writing. The copy must
    # hold the 9mo's primary h5 (gf_prod_9mo_testing.h5; the running backup
    # copy beside it is the torn-primary fallback). Rename here if the copy
    # is called something else; the id lists are the 9mo's own.
    parent=Parent(name="9mo", store_dirname="gf_prod_9mo_v9_copy",
                  base_file_name="gf_prod_9mo", days=270,
                  mbhb_ids=mbhb_ids_for(270)),
    gb_nleaves_max=20000, gb_n_subbands=4096, gb_rj_inmodel_chunk=16384,
    edge_crop_wavelets=60,
    notes=(
        "PARENT = gf_prod_9mo_v9_copy, a COPY of the live 9mo run folder taken",
        "before this launch (Mike 2026-10-13) -- the live store is never read.",
        "GB_WARM_START_LAST_K = the parent's last STORED ROWS (x 4 walkers; a",
        "search stage stores 3 rows per cycle under GB_SEARCH_LEGS=1); set it on",
        "the launch line or as an override here once decided.",
        "MBHB 10 (369.5 d, SNR ~1963) is IN under the two-week rule (the 1yr v9",
        "left it out under the 7-d code default). EMRIs 0 (347 d) and 3 (256 d)",
        "plunge inside the window.",
    ),
))
_add(Run(
    name="18mo", days=540, parent="12mo",
    gb_nleaves_max=25000, gb_n_subbands=2048, gb_rj_inmodel_chunk=8192,
    edge_crop_wavelets=90,
    notes=(
        "First run past the 1yr: capacity knobs = the 24mo's (as the 9mo took the",
        "1yr's); EDGE_CROP 90 (guard needs 73). MBHB 17 merges at 585.5 d, 31.5 d",
        "past the window, and is out. UNVERIFIED AT THIS Tobs: the stage-B F-stat",
        "sky cap (512) binds harder (natural counts 2x the 1yr's), the SOBBH",
        "lookup EVAL_DT 43200 and MBH decimation 2 were validated to 1 yr / 2 yr",
        "only in parts -- read the first tar's preflight + [SOBBH_LL_TIMING] lines.",
    ),
))
_add(Run(
    name="24mo", days=720, parent="18mo",
    gb_nleaves_max=25000, gb_n_subbands=2048, gb_rj_inmodel_chunk=8192,
    edge_crop_wavelets=120,
    notes=(
        "All 20 catalogue MBHBs merge inside 720 d (the last, 14, at 712.7 d).",
        "EDGE_CROP 120 (guard needs 95). Same unverified-at-this-Tobs list as the",
        "18mo; the F-stat host-mapped float32 grid (TODO) is the real fix for the",
        "stage-B memory at 1 yr and beyond.",
    ),
))

ORDER = ("3mo", "6mo", "9mo", "12mo", "18mo", "24mo")
for _r in RUNS.values():
    _r.check()
