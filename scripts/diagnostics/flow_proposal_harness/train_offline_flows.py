"""Phase 0/1 driver, part 1: train candidate flow configs on backend-derived
buffers (offline, GPU 7 via CUDA_VISIBLE_DEVICES). No MCMC involved.

Buffers mimic the ring-buffer feed: cold-chain coords per leaf over a trailing
window of steps, time-ordered (oldest first) so val_split="temporal" holds out
the newest rows, exactly as in the live trainer.

Candidates (Phase 1 config levers):
  MBH : noise0.1_w84   -- baseline, matches live guards (anchor)
        noise0_w84     -- train_noise off
        noise0.01_w84  -- fallback guard value
        noise0_w168    -- train_noise off + double buffer (max_buffer_samples lever)
        noise0_w84_e250-- train_noise off + more epochs
  EMRI: noise0.1_w84 / noise0.01_w84 / noise0_w84
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import time
import h5py
import numpy as np
import torch
torch.set_num_threads(4)
from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning, ModeMixtureFlow

TMP = "/home/asantini/.claude/jobs/f7a00a42/tmp"
OUT = f"{TMP}/offline_flows"
os.makedirs(OUT, exist_ok=True)
SEED = 103209
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

f = h5py.File(f"{TMP}/main_backend.h5", "r")
ll = f["mcmc/log_like"][:]
n = int(np.where(np.any(ll != 0.0, axis=(1, 2)))[0][-1]) + 1
log(f"backend steps: {n}")

def buffer(branch, nleaves, window):
    ch = f[f"mcmc/chain/{branch}"][n - window:n, 0]  # (window, 24, nl, nd) cold
    return {leaf: ch[:, :, leaf, :].reshape(-1, ch.shape[-1]).astype(np.float64)
            for leaf in range(nleaves)}

BRANCH = {
    "mbh": dict(dims=11, nleaves=6,
                periodic={5: (0.0, 2 * np.pi), 7: (0.0, np.pi), 8: (0.0, 2 * np.pi)}),
    "emri": dict(dims=12, nleaves=2,
                 periodic={7: (0.0, 2 * np.pi), 8: (0.0, 2 * np.pi),
                           9: (0.0, 2 * np.pi), 10: (0.0, 2 * np.pi)}),
}

def make_flow(branch, pcov=False, mixture=False, kmax=8):
    """Build a candidate flow. mixture=True mirrors the settings' MBH block
    (ModeMixtureFlow: buffer-estimated per-island conditions + exact mixture MH)."""
    b = BRANCH[branch]
    common = dict(
        flow_class="NSF", device="cuda",
        data_transform=WhiteningTransform(ndim=b["dims"], periodic=b["periodic"], shared=False,
                                          periodic_in_cholesky=pcov),
        seed=SEED, transforms=8, hidden_features=(128, 128, 128), bins=8,
    )
    if mixture:
        return ModeMixtureFlow(
            dims=b["dims"], nleaves_max=b["nleaves"], kmax=kmax, mode_floor=0.02,
            cluster_seed=SEED, periodic=b["periodic"], **common,
        )
    return ZukoFlow(
        dims=b["dims"], conditioning=OneHotLeafConditioning(nleaves_max=b["nleaves"]), **common,
    )

CANDS = {
    "mbh": [
        ("noise0.1_w84",   dict(window=84,  train_noise=0.1,  n_epochs=150)),
        ("noise0_w84",     dict(window=84,  train_noise=0.0,  n_epochs=150)),
        ("noise0.01_w84",  dict(window=84,  train_noise=0.01, n_epochs=150)),
        ("noise0_w168",    dict(window=168, train_noise=0.0,  n_epochs=150)),
        ("noise0_w84_e250",dict(window=84,  train_noise=0.0,  n_epochs=250)),
        ("noise0_w168_pcov", dict(window=168, train_noise=0.0, n_epochs=150, pcov=True)),
        ("noise0_w84_pcov",  dict(window=84,  train_noise=0.0, n_epochs=150, pcov=True)),
        # Task-7 gate: the settings' MBH config (mixture + pcov). k12 probes the
        # plan's first fallback knob if the multimodal leaves miss >= 0.08.
        ("noise0_w168_mixture", dict(window=168, train_noise=0.0, n_epochs=150,
                                     pcov=True, mixture=True)),
        ("noise0_w168_mixture_k12", dict(window=168, train_noise=0.0, n_epochs=150,
                                         pcov=True, mixture=True, kmax=12)),
        # rows-per-island test: max available window (290 steps = 6960 rows/leaf
        # -> ~1740/island at K=4, i.e. leaf-2's working regime). Staleness is
        # splice-free and the window is stationary (smear ~1.0), so the only
        # thing that changes is rows per component.
        ("noise0_w290_mixture", dict(window=290, train_noise=0.0, n_epochs=150,
                                     pcov=True, mixture=True)),
        ("noise0_w290_pcov", dict(window=290, train_noise=0.0, n_epochs=150, pcov=True)),
    ],
    "emri": [
        ("noise0.1_w84",   dict(window=84,  train_noise=0.1,  n_epochs=150)),
        ("noise0.01_w84",  dict(window=84,  train_noise=0.01, n_epochs=150)),
        ("noise0_w84",     dict(window=84,  train_noise=0.0,  n_epochs=150)),
        ("noise0_w168",    dict(window=168, train_noise=0.0,  n_epochs=150)),
        ("noise0_w240",    dict(window=240, train_noise=0.0,  n_epochs=150)),
        ("noise0_w168_pcov", dict(window=168, train_noise=0.0, n_epochs=150, pcov=True)),
        # EMRI is unimodal (settings keep plain ZukoFlow); mixture should be a
        # no-op here (K=1 per leaf) -- a control on the wrapper's neutrality.
        ("noise0_w168_mixture", dict(window=168, train_noise=0.0, n_epochs=150,
                                     pcov=True, mixture=True)),
    ],
}

for branch, cands in CANDS.items():
    for tag, cfg in cands:
        if os.path.exists(f"{OUT}/{branch}_{tag}.h5"):
            log(f"{branch:4s} {tag:16s}: exists, skipping")
            continue
        samples = buffer(branch, BRANCH[branch]["nleaves"], cfg["window"])
        rows = sum(len(v) for v in samples.values())
        flow = make_flow(branch, pcov=cfg.get("pcov", False),
                         mixture=cfg.get("mixture", False), kmax=cfg.get("kmax", 8))
        hist = flow.fit(
            samples, n_epochs=cfg["n_epochs"], batch_size=1024, lr=1e-3,
            lr_annealing=True, optimizer="adamw", patience=30,
            validation_fraction=0.15, val_split="temporal",
            train_noise=cfg["train_noise"], verbose=False,
        )
        path = f"{OUT}/{branch}_{tag}.h5"
        if os.path.exists(path):
            os.remove(path)
        flow.save(path)
        vl = np.asarray(hist.validation_loss)
        log(f"{branch:4s} {tag:16s}: rows {rows:5d}, epochs {len(vl):3d}, "
            f"best val NLL {vl.min():8.3f} (last {vl[-1]:8.3f}) -> {os.path.basename(path)}")
log("ALL TRAINED")
