"""CUDA-graph feasibility bench for the GB in-model repeat trains.

WHY (2026-08-27, v7 snapshot-6 profile): the in-model repeat trains run
25/250 sequential MH steps per staged block, each step a train of small
kernel launches over a (width,) source batch. Physics per step at width
2048 is ~14 ms (2048 x ~7 us/eval) under DOZENS of launches -- the step
is launch-bound, and the budgets/chunk shapes were deliberately kept
rigid "for CUDA-graph capture later" (gbspecialstretch design notes).
This bench measures, ON THE PRODUCTION GPU, how much capturing a
fixed-shape repeat train actually buys, WITHOUT touching production
code: it synthesizes a repeat step with the same structural mix
(batched proposal draw, waveform-scale elementwise work, inner-product
reduction, MH compare + masked scatter, a vertical-swap-style paired
pass), unrolls R steps into one captured graph, and compares eager
launch vs graph replay -- with a bit-identity check between the two
paths from identical inputs.

RUNBOOK (cluster, interactive GPU node, any env with cupy):

    python scripts/benchmark/cuda_graph_repeat_bench.py \
        --width 4096 --repeats 25 --blocks 20
    python scripts/benchmark/cuda_graph_repeat_bench.py \
        --width 4096 --repeats 250 --blocks 4
    python scripts/benchmark/cuda_graph_repeat_bench.py \
        --width 2048 --repeats 25 --blocks 20   # pre-4096 comparison

Interpretation: 'eager ms/step' - 'graph ms/step' is the recoverable
launch overhead per step; multiply by the production train census
(snapshot 6, row 8: 48 train blocks, sum(repeats x sources) ~ 5.5M,
1020 s wall span inside rj_fstat_search) to get the ceiling for the
real capture project. If the speedup here is < 1.3x, graph capture is
not worth the production plumbing; > 2x justifies phase 2 (capturing
the real step, which additionally requires hoisting the per-step host
counters out of the hot loop).

The synthetic kernels are NOT the production kernels -- this measures
launch-train mechanics only. Bit-identity here proves the harness, not
the physics.
"""

import argparse
import time


def build_step(cp, width, nvals, ndim):
    """One synthetic repeat step: returns fn(state, rand_r) -> None.

    Kernel mix per step (all fixed-shape, capture-safe: no host sync,
    no allocation after warmup):
      1. proposal draw: batched (width, ndim) matvec against a fixed
         per-source (ndim, ndim) factor  (~ Fisher/Cholesky draw)
      2. 'waveform' work: E elementwise passes over (width, nvals)
         (~ per-source template evaluation cost)
      3. inner-product reduction over nvals (~ <d|h>, <h|h>)
      4. MH compare + masked parameter scatter (accept application)
      5. paired swap pass (~ vertical tempering exchange)
    """
    E = 6  # elementwise passes; tuned so step compute ~ production scale

    def step(state, rand_r):
        coords, chol, work, ll, partner = state
        u, jit = rand_r
        # 1. proposal draw
        prop = coords + cp.einsum("kij,kj->ki", chol, jit)
        # 2. waveform-scale work
        acc = work
        for _ in range(E):
            acc = cp.cos(acc) * 1.0001 + prop[:, :1] * 1e-3
        # 3. reduction
        d_h = acc.sum(axis=1)
        h_h = (acc * acc).sum(axis=1)
        delta = d_h - 0.5 * h_h
        # 4. MH compare + masked scatter
        keep = delta - ll > cp.log(u)
        coords[:] = cp.where(keep[:, None], prop, coords)
        ll[:] = cp.where(keep, delta, ll)
        # 5. paired swap pass
        pll = ll[partner]
        swap = pll > ll
        ll[:] = cp.where(swap, pll, ll)
        work[:] = acc
        return None

    return step


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--nvals", type=int, default=512)
    ap.add_argument("--ndim", type=int, default=9)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    try:
        import cupy as cp
    except ImportError:
        raise SystemExit(
            "cupy required -- run on a GPU node (this bench measures real "
            "launch/graph mechanics; there is nothing to measure on CPU).")

    cp.cuda.Device(args.device).use()
    W, R, ND, NV = args.width, args.repeats, args.ndim, args.nvals
    rng = cp.random.default_rng(1234)

    def fresh_state():
        return [
            rng.standard_normal((W, ND)),                     # coords
            cp.tile(cp.eye(ND) * 0.01, (W, 1, 1)),            # chol
            rng.standard_normal((W, NV)),                     # work
            cp.full(W, -1e3),                                 # ll
            cp.roll(cp.arange(W), 1),                         # partner
        ]

    # Random pools indexed per unrolled step; refreshed between blocks
    # OUTSIDE any capture (the graph reads these buffers by reference).
    u_pool = rng.random((R, W))
    jit_pool = rng.standard_normal((R, W, ND))
    step = build_step(cp, W, NV, ND)

    def run_train_eager(state):
        for r in range(R):
            step(state, (u_pool[r], jit_pool[r]))

    # ---- warmup + bit-identity reference --------------------------------
    s_eager = fresh_state()
    run_train_eager(s_eager)          # warmup allocations
    s_eager = fresh_state()
    cp.cuda.get_current_stream().synchronize()

    # ---- capture ---------------------------------------------------------
    stream = cp.cuda.Stream(non_blocking=True)
    s_graph = fresh_state()
    with stream:
        run_train_eager(s_graph)      # warmup on the capture stream
    stream.synchronize()
    s_graph = fresh_state()
    s_check = [a.copy() for a in s_graph]
    with stream:
        stream.begin_capture()
        run_train_eager(s_graph)
        graph = stream.end_capture()
    stream.synchronize()

    # bit-identity: replay the captured graph once from the SAME inputs
    # the eager reference will use, then compare.
    for a, b in zip(s_graph, s_check):
        a[:] = b
    with stream:
        graph.launch(stream)
    stream.synchronize()
    run_train_eager(s_check)
    cp.cuda.get_current_stream().synchronize()
    ident = all(bool(cp.array_equal(a, b)) for a, b in zip(s_graph, s_check))
    print(f"bit-identity graph-vs-eager over one train: {ident}")
    if not ident:
        diffs = [float(cp.abs(a - b).max()) if a.dtype.kind == "f" else -1.0
                 for a, b in zip(s_graph, s_check)]
        print(f"  max diffs per state array: {diffs}")

    # ---- timing ----------------------------------------------------------
    def refresh_pools():
        u_pool[:] = rng.random((R, W))
        jit_pool[:] = rng.standard_normal((R, W, ND))

    cp.cuda.get_current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(args.blocks):
        refresh_pools()
        run_train_eager(s_eager)
    cp.cuda.get_current_stream().synchronize()
    t_eager = time.perf_counter() - t0

    stream.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.blocks):
        # refresh on the SAME stream the graph replays on: stream-ordered,
        # no cross-stream race between pool writes and graph reads.
        with stream:
            refresh_pools()
            graph.launch(stream)
    stream.synchronize()
    t_graph = time.perf_counter() - t0

    steps = args.blocks * R
    print(f"width={W} repeats={R} blocks={args.blocks} nvals={NV}")
    print(f"eager : {t_eager:8.3f} s total | {1e3 * t_eager / steps:7.3f} ms/step")
    print(f"graph : {t_graph:8.3f} s total | {1e3 * t_graph / steps:7.3f} ms/step")
    print(f"speedup {t_eager / t_graph:5.2f}x | recoverable "
          f"{1e3 * (t_eager - t_graph) / steps:7.3f} ms/step")
    print("production scaling hint: row-8 census had 48 train blocks, "
          "~5.5M repeat-source steps, 1020 s wall inside rj_fstat_search; "
          "multiply the recoverable ms/step by (your blocks x repeats).")


if __name__ == "__main__":
    main()
