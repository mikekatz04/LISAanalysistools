"""PSD fan-out end to end: the noise_only synthetic fit on 2 compute ranks (fake comm).

Two arms:

* ``test_two_compute_ranks_run_the_noise_moves`` -- the historical one. Four
  walkers, two ranks, a 2-walker block each; unchanged.
* ``test_one_walker_two_replicas`` -- ONE walker on THREE ranks (head +
  compute + a dedicated saver), which is what puts ``build_layout`` in
  ONE-WALKER REPLICA mode: both compute ranks get block ``(0, 1)`` and hold
  the SAME walker's residual. The PSD family then runs its unchanged body on
  the head and scatters likelihood ROWS over the replicas through
  ``RowFanout`` (op ``psd_rows``), replaying the propose-begin prep and the
  end-of-propose noise publish on every rank (op ``psd_replay``). The inner
  proposal is the eigen-axis MH move -- at one walker there is no stretch
  complement -- which ``PSDMove._resolve_inner_kind`` picks and logs once.
  A size-2 world would NOT exercise any of this: ``resolve_roles`` aliases
  the saver onto the head below three ranks, so rank 1 would be the saver and
  there would be exactly one compute rank (no replicas at all).
"""

import logging
import os
import resource
import shutil
import tempfile
import time
import unittest

import numpy as np

RUN_GF_SMOKE = os.environ.get("RUN_GF_SMOKE", "") not in ("", "0")

#: the noise_only stage runs psd_pe AND galfor_pe, so every stored iteration
#: sends the worker two proposes (plus the one start-up ping)
MOVES_PER_ITERATION = 2

#: prefix on the measurement lines this module prints (wall / RSS / op counts),
#: so they can be grepped out of the run log
MARKER = "[noisesmoke]"


def _rss_gb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 ** 3) if ru > 1e7 else ru / (1024.0 ** 2)  # macOS bytes / linux KB


class _LogCapture(logging.Handler):
    """Every ``lisatools`` record's formatted message, for the test to grep.

    Attached to the ``lisatools`` logger, NOT the root logger: the run's
    ``globalfit.loginfo.init_logger`` sets ``lisatools.propagate = False``
    during ``fit.build()``, so a root handler would see none of these records.
    It leaves handlers other than its own managed console handler alone, so a
    capture attached BEFORE the world starts survives the build. Same trick as
    ``tests/test_multirank_gb_smoke.py::_LogCapture``.

    ``logging.Handler.handle`` takes the handler's lock around ``emit``, which
    is what makes this safe with FakeWorld's one thread per rank.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):  # pragma: no cover - exercised only under RUN_GF_SMOKE
        try:
            self.lines.append(record.getMessage())
        except Exception:
            pass

    def text(self):
        return "\n".join(self.lines)


def _eigen_health(gf):
    """``{move name: {branch: {...}}}`` — is the eigen table on the prior's scale?

    Read off the moves the recipe actually materialized, AFTER the run, so
    it describes the tables the run really proposed with. Per branch:

    * ``widths`` — what :func:`prior_box_widths` resolved;
    * ``true_widths`` — ``max - min`` read straight off the container's
      PARSED ``priors`` list, which is key-spelling agnostic;
    * ``sigma_frac`` — the largest 1-sigma step as a fraction of the prior
      box along its OWN axis (``axis_prior_bounds``). ``<= 1`` by
      construction once the widths are right; the 2026-09-16 defect had it
      at ~1.8e5 for psd, so every draw left the prior and the move accepted
      nothing.
    """
    from eryn.moves.eigenaxis import axis_prior_bounds

    from lisatools.globalfit.moves.eigen_refresh import prior_box_widths
    from lisatools.globalfit.moves.psdmove import PSDMove

    out = {}
    for stage in getattr(getattr(gf, "recipe", None), "stages", []) or []:
        for mv in stage.moves:
            runtime = getattr(mv, "runtime", None)
            if not isinstance(runtime, PSDMove):
                continue
            inner = getattr(runtime, "_eigen_inner", None)
            tables = dict(getattr(inner, "_tables", {}) or {})
            per_branch = {}
            for branch, (axes, sigmas) in tables.items():
                axes = np.asarray(axes, dtype=float)
                sigmas = np.asarray(sigmas, dtype=float)
                ndim = int(axes.shape[-1])
                widths = np.asarray(prior_box_widths(runtime.priors[branch], ndim))
                true = np.ones(ndim)
                for inds, dist in runtime.priors[branch].priors:
                    mn, mx = getattr(dist, "minimum", None), getattr(dist, "maximum", None)
                    if mn is None or mx is None:
                        continue
                    for col in np.atleast_1d(np.asarray(inds)).ravel():
                        if 0 <= int(col) < ndim:
                            true[int(col)] = float(mx) - float(mn)
                flat_ax = axes.reshape(-1, ndim, ndim)
                flat_sg = sigmas.reshape(-1, ndim)
                # bound against the TRUE box, never the resolved ``widths``:
                # the defect scaled the cap by the same wrong widths, so a
                # self-referential ratio is <= 1 even when every draw is
                # 1e5 prior widths out.
                bounds = np.asarray(axis_prior_bounds(flat_ax, true))
                per_branch[branch] = {
                    "widths": widths,
                    "true_widths": true,
                    "sigma_frac": float(np.max(flat_sg / np.maximum(bounds, 1e-300))),
                }
            if per_branch:
                out[mv.name] = per_branch
    return out


@unittest.skipUnless(RUN_GF_SMOKE, "set RUN_GF_SMOKE=1 to run the multi-rank noise smoke")
class MultiRankNoiseSmokeTest(unittest.TestCase):
    WORLD_SIZE = 2  # head + one compute rank
    ITERATIONS = 2
    NWALKERS = 4
    NTEMPS = 2

    def setUp(self):
        os.environ.setdefault("USE_GPU", "0")
        os.environ.setdefault("MAKE_DIAGNOSTIC_PLOTS", "0")
        self.tmpdir = tempfile.mkdtemp(prefix="gf_multirank_noise_")
        # the per-branch setup logs ignore file_store_dir and land in the cwd;
        # remember whether that directory was already there so tearDown only
        # removes one THIS test created
        self._stray = os.path.join(os.getcwd(), "gf_output_noise")
        self._stray_existed = os.path.isdir(self._stray)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        if not self._stray_existed:
            shutil.rmtree(self._stray, ignore_errors=True)

    def _run_world(self, size=None, nwalkers=None, iterations=None):
        """``{rank: probe}`` plus the string key ``"log"`` (every lisatools record).

        ``size``/``nwalkers`` default to the class fixture (2 ranks, 4
        walkers), so the historical arm is byte-for-byte the same run it was.
        ``nwalkers=1`` with two COMPUTE ranks (i.e. ``size >= 3``) is replica
        mode. ``iterations`` (default ``ITERATIONS``) is how many iterations
        THIS world runs (``run_mcmc``'s count): a second world on the same
        ``(size, nwalkers)`` store RESUMES it and appends that many rows.
        """
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import RankRole, prepare_rank
        from lisatools.globalfit.run import GlobalFit
        from lisatools.globalfit.stock import erebor

        size = self.WORLD_SIZE if size is None else int(size)
        nwalkers = self.NWALKERS if nwalkers is None else int(nwalkers)
        iterations = self.ITERATIONS if iterations is None else int(iterations)

        def fn(rank, comm):
            fit = erebor.noise_only(
                nwalkers=nwalkers, ntemps=self.NTEMPS, data_mode="synthetic",
                file_store_dir=os.path.join(self.tmpdir, f"n{size}w{nwalkers}"),
                make_diagnostic_plots=False,
            )
            fit.general.num_iterations = iterations
            layout = prepare_rank(fit, comm)
            fit.build()
            gf = GlobalFit(fit, comm)
            gf.run_global_fit()
            role = layout.role_of(rank)
            if role is RankRole.SAVER:
                # a dedicated saver (size >= 3) never builds an ACA: it opens
                # the HDF backend and parks in the async save loop
                return {"role": role.value,
                        "store": getattr(gf.curr.general_info, "main_file_path", None)}
            out = {"role": role.value, "acs_rows": int(gf.acs.acs_total_entries),
                   "layout": layout, "block": layout.block_of(rank),
                   # every compute rank's own residual + noise model, scored
                   # (replicas of one block must agree with their lead)
                   "acs_like": np.asarray(gf.acs.likelihood(), dtype=float).reshape(-1)}
            if hasattr(gf, "compute_service"):
                out["served"] = gf.compute_service_served
                return out
            out["log_like"] = np.array(gf.state.log_like[0], copy=True)
            out["betas"] = {
                k: np.array(s.betas, copy=True)
                for k, s in gf.state.sub_states.items()
                if s is not None and getattr(s, "betas", None) is not None
            }
            # ---- head-only extras (additive; the 4-walker arm ignores them).
            # ``run_global_fit`` throws away ``run_mcmc``'s return value, so
            # ``gf.state`` above is still the STARTING state. eryn keeps the
            # last state for its ``initial_state=None`` resume path, and that
            # in-memory object is exactly what the moves produced.
            final = getattr(gf.sampler, "_previous_state", None)
            if final is None:  # pragma: no cover - eryn API fallback
                final = gf.sampler.get_last_sample()
            out["state"] = final
            out["final_log_like"] = np.array(final.log_like[0], copy=True)
            # the CORRECTNESS probe: a fresh likelihood off the head's own ACA
            # after the run. ``PSDMove.propose_local`` ends every propose with
            # ``new_state.log_like[0] = self.acs.likelihood()`` (psdmove.py
            # :2832-2839), so on the head the final cold row and this recompute
            # must be the same numbers -- a replica whose published noise model
            # or scattered rows were wrong shows up as a mismatch here.
            out["acs_like"] = np.asarray(gf.acs.likelihood(), dtype=float).reshape(-1)
            out["psd_eigen"] = _eigen_health(gf)
            try:
                with np.errstate(all="ignore"):
                    out["acceptance_fraction"] = np.asarray(
                        gf.sampler.acceptance_fraction, dtype=float
                    )
            except Exception as exc:  # pragma: no cover - eryn API drift
                out["acceptance_fraction"] = None
                out["acceptance_fraction_error"] = f"{type(exc).__name__}: {exc}"
            return out

        capture = _LogCapture()
        logging.getLogger("lisatools").addHandler(capture)
        started = time.time()
        try:
            out = FakeWorld(size, timeout=1800.0).run(fn)
        finally:
            logging.getLogger("lisatools").removeHandler(capture)
        print(f"{MARKER} world size={size} nwalkers={nwalkers}: "
              f"wall={time.time() - started:.1f}s peak_rss={_rss_gb():.2f}GB", flush=True)
        out["log"] = capture.text()
        return out

    def test_two_compute_ranks_run_the_noise_moves(self):
        out = self._run_world()
        self.assertEqual(out[0]["role"], "head")
        self.assertEqual(out[1]["role"], "compute")
        self.assertEqual((out[0]["acs_rows"], out[1]["acs_rows"]), (2, 2))
        # ping + one propose per noise move per stored iteration
        self.assertGreaterEqual(
            out[1]["served"], 1 + MOVES_PER_ITERATION * self.ITERATIONS
        )
        self.assertTrue(np.all(np.isfinite(out[0]["log_like"])))
        self.assertEqual(out[0]["log_like"].shape, (4,))
        for betas in out[0]["betas"].values():
            self.assertTrue(np.all(np.diff(betas) <= 0))  # a valid ladder after the head's step

    def test_one_walker_two_replicas(self):
        """1 walker on 2 replicas (+ a saver): rows scatter, replays broadcast.

        THREE ranks on purpose -- head, compute, saver. ``resolve_roles``
        aliases the saver onto the head below three ranks, so a size-2 world
        at one walker would demote rank 1 to the saver and leave a single
        compute rank: no replicas, no scatter, and every assertion below would
        pass vacuously through ``RowFanout``'s ``not self.active`` local-call
        branch. That is why the spy records ``active`` per call and the
        assertions demand a call with ``active=True``.

        The claims, in order of what a defect would break first:

        * the world does not RAISE -- a ``RemoteWorkerError`` on any rank is
          re-raised by ``FakeWorld.run`` as a ``RuntimeError`` carrying the
          remote traceback, so "it returned" is the no-error claim;
        * the PSD move really resolved its inner proposal to the eigen-axis
          move (the log line ``PSDMove._resolve_inner_kind`` emits once) --
          ``stretch`` has no complement at one walker and would have raised;
        * the replicas really DID work: at least one ``psd_rows`` scatter and
          at least one ``psd_replay`` broadcast, both with the fan-out ACTIVE,
          and the compute rank answered a non-zero number of commands;
        * the numbers are sane and SELF-CONSISTENT: the head's final cold row
          equals a fresh ``acs.likelihood()`` recompute on the head;
        * the eigen inner proposal actually MIXES -- every PSDMove's own
          in-model accepted count over the whole run is > 0, its eigen steps
          sit inside the prior box, and the move never hit the
          "All points entering likelihood have a log prior of minus inf"
          dead end. Mechanical success is not enough: before the
          2026-09-16 ``prior_box_widths`` fix every claim above passed while
          the psd branch accepted exactly ZERO proposals.
        """
        from unittest import mock

        from lisatools.globalfit.communication.rowfanout import RowFanout
        from lisatools.globalfit.moves import psdmove as psdmove_mod
        from lisatools.globalfit.moves.psdmove import PSDMove

        # Spies, not stubs: the real bodies run and their return values are
        # passed through. ``replay`` does NOT go through ``run`` (it calls
        # ``self.fanout.run`` directly), so both are wrapped. Only the HEAD
        # thread calls either -- the replicas answer through
        # ``PSDMove.serve_psd_rows`` / ``serve_psd_replay`` -- and
        # ``list.append`` is atomic anyway. Installed BEFORE the world starts,
        # on the class, so every rank thread sees the same patched method.
        calls = []
        _real_run = RowFanout.run
        _real_replay = RowFanout.replay

        def _spy_run(rf_self, op, rows, *, local_body):
            calls.append(("run", op, bool(rf_self.active), rf_self.move_name))
            return _real_run(rf_self, op, rows, local_body=local_body)

        def _spy_replay(rf_self, op, payload, *, local_body):
            kind = payload.get("kind") if isinstance(payload, dict) else None
            calls.append(("replay", op, bool(rf_self.active), kind))
            return _real_replay(rf_self, op, payload, local_body=local_body)

        # The move's OWN in-model tallies are RESET per propose (psdmove.py
        # ``propose_local``), so reading them after the run would show only the
        # last propose. ``run_move`` is where they are incremented, so wrap it
        # and accumulate over the whole run instead.
        accepts = {}
        _real_run_move = PSDMove.propose_local

        def _spy_propose(mv_self, model, state):
            new_state, accepted = _real_run_move(mv_self, model, state)
            tally = getattr(mv_self, "_tally_in_model_accepted", None)
            prop = getattr(mv_self, "_tally_in_model_proposed", None)
            a, p = accepts.get(mv_self.name, (0, 0))
            accepts[mv_self.name] = (
                a + (0 if tally is None else int(np.sum(tally))),
                p + (0 if prop is None else int(np.sum(prop))),
            )
            return new_state, accepted

        # ``PSDMove.compute_log_like`` warns through the module's ``warnings``
        # when every proposed row has log prior -inf -- i.e. the eigen step
        # left the prior box on EVERY rung. That is a ``warnings.warn``, not a
        # log record, so ``_LogCapture`` cannot see it; ``catch_warnings`` is
        # process-global and unsafe across FakeWorld's rank THREADS. Wrapping
        # the module's ``warn`` (pass-through, restored by mock) counts it
        # without changing any behavior.
        minus_inf = []
        _real_warn = psdmove_mod.warnings.warn

        def _spy_warn(message, *a, **k):
            if "log prior of minus inf" in str(message):
                minus_inf.append(str(message))
            return _real_warn(message, *a, **k)

        with mock.patch.object(RowFanout, "run", _spy_run), \
                mock.patch.object(RowFanout, "replay", _spy_replay), \
                mock.patch.object(PSDMove, "propose_local", _spy_propose), \
                mock.patch.object(psdmove_mod.warnings, "warn", _spy_warn):
            out = self._run_world(size=3, nwalkers=1)

        rows_calls = [c for c in calls if c[0] == "run" and c[1] == "psd_rows"]
        replay_calls = [c for c in calls if c[0] == "replay" and c[1] == "psd_replay"]
        print(f"{MARKER} RowFanout ops: psd_rows={len(rows_calls)} "
              f"(active={sum(1 for c in rows_calls if c[2])}) "
              f"psd_replay={len(replay_calls)} "
              f"(active={sum(1 for c in replay_calls if c[2])}, "
              f"kinds={sorted({c[3] for c in replay_calls})}) "
              f"served_by_rank1={out[1].get('served')}", flush=True)

        # roles: head + compute + a dedicated saver == two compute ranks == two replicas
        self.assertEqual(
            (out[0]["role"], out[1]["role"], out[2]["role"]),
            ("head", "compute", "saver"))
        self.assertEqual((out[0]["acs_rows"], out[1]["acs_rows"]), (1, 1))

        # the eigen inner proposal was resolved and logged (psdmove.py
        # ``_resolve_inner_kind``: "[%s] resolved PSD inner proposal kind ->
        # %r (block nwalkers=%d, run nwalkers=%d)")
        log = out["log"]
        eigen_lines = [
            line for line in log.splitlines()
            if "resolved PSD inner proposal kind" in line
        ]
        for line in eigen_lines:
            print(f"{MARKER} {line}", flush=True)
        self.assertTrue(eigen_lines, "the PSD move never logged its resolved inner kind")
        for line in eigen_lines:
            self.assertIn("-> 'eigen'", line,
                          "the PSD inner proposal did not resolve to the eigen-axis move")

        # the replicas really served rows / replays, with the fan-out ACTIVE
        # (a single-compute-rank world takes RowFanout's local-call branch and
        # would report active=False)
        self.assertTrue([c for c in rows_calls if c[2]],
                        "no psd_rows scatter reached the replicas")
        self.assertTrue([c for c in replay_calls if c[2]],
                        "no psd_replay broadcast reached the replicas")
        self.assertEqual(sorted({c[3] for c in replay_calls}), ["begin", "publish"],
                         "the propose-begin prep and/or the end-of-propose publish "
                         "never replayed")
        self.assertGreater(int(out[1]["served"]), 0,
                           "the compute rank answered no commands")

        # the numbers
        final_ll = out[0]["final_log_like"]
        self.assertTrue(np.all(np.isfinite(final_ll)),
                        f"final cold-row log_like not finite: {final_ll}")
        self.assertEqual(final_ll.shape, (1,))
        recompute = out[0]["acs_like"]
        np.testing.assert_allclose(
            final_ll, recompute, rtol=0.0, atol=1e-8,
            err_msg="the head's final cold-row log_like disagrees with a fresh "
                    "acs.likelihood() recompute on the head")

        # acceptance_fraction: a RuntimeWarning in eryn's accepted/iteration
        # division surfaces as a nan, and capturing warnings across FakeWorld's
        # rank THREADS is not safe (warnings filters are process-global), so
        # the finiteness check is the reachable form of that claim.
        af = out[0]["acceptance_fraction"]
        if af is None:
            print(f"{MARKER} acceptance_fraction not reachable: "
                  f"{out[0].get('acceptance_fraction_error')}", flush=True)
        else:
            print(f"{MARKER} acceptance_fraction={np.asarray(af).ravel()}", flush=True)
            self.assertTrue(np.all(np.isfinite(af)),
                            f"acceptance_fraction has nan/inf: {af}")

        # ---- the eigen inner proposal MIXES (2026-09-16 regression) --------
        # ``prior_box_widths`` resolved prior columns off ``priors_in`` KEYS,
        # and ``psd_prior_dict`` keys are LaTeX LABELS, so the psd branch got
        # width 1.0 on levels living at ~1e-11/~1e-14 in boxes 1.9e-10/2.0e-13
        # wide. Both the finite-difference corners and the prior-box cap are
        # width-scaled, so the move stepped ~1e5 prior widths and accepted
        # NOTHING while every claim above still passed.
        print(f"{MARKER} in-model (accepted, proposed) per move: {accepts}", flush=True)
        self.assertTrue(accepts, "no PSDMove propose was observed at all")
        for name, (acc, prop) in sorted(accepts.items()):
            self.assertGreater(prop, 0, f"move {name!r} proposed nothing")
            self.assertGreater(
                acc, 0,
                f"move {name!r} accepted 0 of {prop} in-model proposals over the "
                "whole run -- the eigen inner proposal is not mixing")

        self.assertEqual(
            minus_inf, [],
            "PSDMove.compute_log_like hit 'All points entering likelihood have "
            f"a log prior of minus inf' {len(minus_inf)}x: every eigen draw left "
            "the prior box")

        health = out[0]["psd_eigen"]
        print(f"{MARKER} eigen tables: "
              + "; ".join(
                  f"{mv}/{b}: widths={np.asarray(d['widths'])} "
                  f"sigma_frac={d['sigma_frac']:.3e}"
                  for mv, per in sorted(health.items())
                  for b, d in sorted(per.items())), flush=True)
        self.assertTrue(health, "no eigen table was installed by any PSD move")
        for mv, per in sorted(health.items()):
            for b, d in sorted(per.items()):
                np.testing.assert_allclose(
                    d["widths"], d["true_widths"], rtol=1e-12,
                    err_msg=f"{mv}/{b}: prior_box_widths does not match the "
                            "branch prior box (string-keyed prior container?)")
                # every 1-sigma step is at most one prior width along its own
                # axis. 1.8e5 before the fix.
                self.assertLessEqual(
                    d["sigma_frac"], 1.0 + 1e-9,
                    f"{mv}/{b}: eigen sigma is {d['sigma_frac']:.3e} prior widths "
                    "along its own axis -- draws cannot land inside the prior")

        # The '[infomat] N/N matrices had a non-positive eigenvalue' line is
        # NOT asserted away: the observed information -d_i d_j lnL is not
        # sign-definite away from the peak (see info_matrix_ll's docstring),
        # and at iteration 0 every rung sits on a slope from a random prior
        # draw, so an indefinite matrix there is physics, not a defect. It is
        # printed for the record; the PSD-cone projection degrades those rungs
        # to prior-scale steps, which the acceptance assertion above shows are
        # perfectly usable.
        infomat_lines = [ln for ln in out["log"].splitlines()
                         if "non-positive eigenvalue" in ln]
        print(f"{MARKER} infomat PSD-cone projections: {len(infomat_lines)}", flush=True)
        for ln in infomat_lines:
            print(f"{MARKER}   {ln}", flush=True)

    # ---- GF_GPU_ROUTING: 4 walkers x R = 2 (the NGPUS=8 production shape) ----
    def _routing_world(self, iterations=None, spy=None):
        """4 walkers on 8 compute ranks + a saver, ``GF_GPU_ROUTING=1``.

        The 9mo production shape on 8 GPUs, at noise-only scale: 4 walker
        blocks x R = 2. PSD/galfor are NOT flat-capable, so they take the
        BLOCK path: each block's lead proposes, and the head publishes the
        merged noise model to the passive replica (``replica_sync``).
        ``GF_RESIDUAL_SNAPSHOT_EVERY=1`` so the snapshot fan-out runs too.
        """
        from unittest import mock

        env = {"GF_GPU_ROUTING": "1", "GF_RESIDUAL_SNAPSHOT_EVERY": "1"}
        with mock.patch.dict(os.environ, env):
            return self._run_world(size=9, nwalkers=4, iterations=iterations)

    def _assert_replicas_match_their_leads(self, out):
        layout = out[0]["layout"]
        self.assertEqual((layout.n_compute, layout.n_blocks, layout.ranks_per_block),
                         (8, 4, 2))
        self.assertTrue(layout.gpu_routing)
        final_ll = out[0]["final_log_like"]
        self.assertEqual(final_ll.shape, (4,))
        self.assertTrue(np.all(np.isfinite(final_ll)), final_ll)
        for r in range(8):
            self.assertEqual(out[r]["block"], (r // 2, r // 2 + 1), r)
            self.assertEqual(out[r]["acs_rows"], 1, r)
            # every replica -- the passive one included -- holds the noise
            # model the head stored for its walker: the publish reached it
            np.testing.assert_allclose(
                out[r]["acs_like"], final_ll[r // 2: r // 2 + 1], rtol=0.0, atol=1e-8,
                err_msg=f"rank {r} (block {r // 2}, replica {r % 2}) disagrees "
                        "with the stored cold lnL of its walker")

    def test_four_walkers_two_replicas_each(self):
        """4 walkers x R = 2: the block path, the replica publish, the snapshot."""
        from unittest import mock

        from lisatools.globalfit.moves.psdmove import PSDMove

        synced, proposed = [], []
        _real_sync = PSDMove.fanout_apply_replica_sync
        _real_body = PSDMove.propose_local

        def _spy_sync(mv, payload):
            synced.append((mv.gf_rank, mv.name))
            return _real_sync(mv, payload)

        def _spy_body(mv, model, state):
            proposed.append(mv.gf_rank)
            return _real_body(mv, model, state)

        with mock.patch.object(PSDMove, "fanout_apply_replica_sync", _spy_sync), \
                mock.patch.object(PSDMove, "propose_local", _spy_body):
            out = self._routing_world()
        print(f"{MARKER} 4x2: proposed by ranks {sorted(set(proposed))}, "
              f"replica syncs by ranks {sorted({r for r, _ in synced})} "
              f"({len(synced)} total)", flush=True)
        self.assertEqual(
            tuple(out[r]["role"] for r in range(9)),
            ("head",) + ("compute",) * 7 + ("saver",))
        self._assert_replicas_match_their_leads(out)
        # ONLY the leads ran the body; ONLY the passives applied a publish
        self.assertEqual(sorted(set(proposed)), [0, 2, 4, 6])
        self.assertEqual(sorted({r for r, _ in synced}), [1, 3, 5, 7])
        # one publish per passive per noise propose (psd_pe + galfor_pe per
        # iteration)
        self.assertEqual(len(synced), 4 * MOVES_PER_ITERATION * self.ITERATIONS)
        self.assertIn("block path with 2 rank(s) per block", out["log"])
        # the snapshot fan-out: every walker's residual, from the block leads
        from lisatools.globalfit.residual_snapshot import snapshot_path

        snap = np.load(snapshot_path(out[8]["store"]))
        self.assertEqual(snap["residuals"].shape[0], 4)
        np.testing.assert_array_equal(snap["walkers"], np.arange(4))

    def test_four_walkers_two_replicas_save_and_resume(self):
        """A store written at 4 walkers x R = 2 resumes at the same shape."""
        import h5py

        first = self._routing_world(iterations=self.ITERATIONS)
        self._assert_replicas_match_their_leads(first)
        store = first[8]["store"]

        def _group(fh):
            return fh["global_fit"] if "global_fit" in fh else fh["mcmc"]

        with h5py.File(store, "r") as fh:
            n_first = int(_group(fh).attrs["iteration"])
            rows_first = np.array(_group(fh)["log_like"][:n_first], copy=True)
        self.assertEqual(n_first, self.ITERATIONS)
        # every stored row is WHOLE (4 walkers, all finite): the head ships
        # the merged ensemble to the saver whatever the replica factor
        self.assertEqual(rows_first.shape[-1], 4)
        self.assertTrue(np.all(np.isfinite(rows_first)))
        # ``num_iterations`` is what THIS job runs (``run_mcmc``'s count); a
        # resume appends that many rows to the stored ones
        second = self._routing_world(iterations=2)
        self._assert_replicas_match_their_leads(second)
        with h5py.File(store, "r") as fh:
            g = _group(fh)
            n_total = int(g.attrs["iteration"])
            # RESUMED, not restarted: the first job's rows are untouched and
            # the second job's two rows were appended after them
            self.assertEqual(n_total, n_first + 2)
            np.testing.assert_array_equal(g["log_like"][:n_first], rows_first)
            self.assertTrue(np.all(np.isfinite(g["log_like"][:n_total])))
        print(f"{MARKER} 4x2 resume: {n_first} -> {n_total} stored rows", flush=True)


if __name__ == "__main__":
    unittest.main()
