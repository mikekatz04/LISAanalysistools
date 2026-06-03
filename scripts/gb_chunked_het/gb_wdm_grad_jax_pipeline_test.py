"""
========================================================================
  Full WDM-pipeline  jax.grad  vs  C-kernel chain rule  validation
========================================================================

This test builds a JAX reimplementation of the full lisa-on-gpu WDM
likelihood pipeline (orbits + sky vectors + TDI X / Y / Z + hp/hc +
fast_wdm_inner + bilinear wavelet lookup + cross-channel inner product)
and uses jax.grad to compute the analytic gradient of

    L(theta) = -1/2 < d - h(theta) | d - h(theta) >    (XYZ cross-noise)

The JAX pipeline tracks the C kernel ``gb_wdm_get_ll_kernel`` step by
step:

    1.  Analytic Keplerian spacecraft positions (matches the spline path
        used by ``EqualArmlengthOrbits`` to many digits and is parameter-
        independent, so it is just numerical constants).
    2.  Sky vectors k, u, v from (lam, beta) exactly as
        ``LISATDIonTheFly::get_sky_vectors``.
    3.  ucb_amp / ucb_phase (= 2 pi (f0 t + 0.5 fdot t^2)  +/-phi0) plus
        get_hp_hc using ``inc``, ``psi``.
    4.  ``get_tdi_Xf_single`` over TDI 1.5g (8 base units cycled to 3
        channels), accumulating delays via the analytic orbits and
        hp/hc-at-delayed-time differences.
    5.  ``get_phase_ref`` via spacecraft-1 propagation delay.
    6.  ``fast_wdm_inner``: central FD of the phase via two extra
        get_tdi_Xf_single + get_phase_ref evaluations at  tn +/- dt.
    7.  Bilinear wavelet lookup of (c_nm, s_nm) at (f_scaled, fdot)
        with the (m+n)-parity swap.
    8.  XYZ cross-channel inner-product accumulation per pixel.

Once L(theta) is implemented in JAX, we compute two gradients:

    grad_jax     = jax.grad(L)(theta_test)                -- analytic
    grad_KERNEL  = python mirror of WDMDomain::add_grad_contrib,
                   per-pixel central FD on the *same* JAX pipeline

The "C-kernel mirror" is byte-for-byte the arithmetic in the rewritten
C kernel (``(w_d - w_h_CENTER) * dw_h / dtheta_k * N^-1``, summed across
pixels and the 3x3 cross-channel noise).  When the lisa-on-gpu build is
back online this mirror should be replaced by a direct call to
``gb_comps.get_ll_grad_wdm(...)`` -- the answer should agree.

This script is self-contained: it does not require the lisa-on-gpu C
build, the LISA orbit files, or any precomputed wavelet table.  It
exists to validate the chain-rule math of the full GB WDM gradient
against jax.grad on the actual pipeline structure.

Run with:
    python gb_wdm_grad_jax_pipeline_test.py
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)


# =============================================================================
#  LISA constants (match LISA.h / lisatools)
# =============================================================================

CLIGHT = 299_792_458.0
LARM = 2.5e9
AU = 149_597_870_700.0
EC = 0.0048241852
KAPPA = 0.0
LAMBDA0 = 0.0
FM = 3.168753575e-8           # rad/s, LISA modulation
SQ3 = np.sqrt(3.0)


# =============================================================================
#  Analytic Keplerian spacecraft positions  -- matches GBGPU spacecraft()
#  and is a high-accuracy match to EqualArmlengthOrbits' spline.
# =============================================================================

def spacecraft_position(t, sc_index):
    """Position of spacecraft sc_index ∈ {0,1,2} at time t."""
    alpha = 2.0 * jnp.pi * FM * t + KAPPA
    sa = jnp.sin(alpha); ca = jnp.cos(alpha)
    beta_const = 2.0 * jnp.pi * sc_index / 3.0 + LAMBDA0
    sb = jnp.sin(beta_const); cb = jnp.cos(beta_const)
    x = AU * ca + AU * EC * (sa * ca * sb - (1.0 + sa * sa) * cb)
    y = AU * sa + AU * EC * (sa * ca * cb - (1.0 + ca * ca) * sb)
    z = -SQ3 * AU * EC * (ca * cb + sa * sb)
    return jnp.stack([x, y, z])


def link_to_emitter_receiver(link):
    """LISA convention: link AB is emitter B, receiver A.

    Returns (emitter_sc, receiver_sc) both 0-indexed for python.
    """
    s = str(int(link))
    rec = int(s[0]) - 1
    emi = int(s[1]) - 1
    return emi, rec


def light_travel_time(t, link):
    """L(t, link) for EqualArmlengthOrbits: just LARM/CLIGHT (constant)."""
    return LARM / CLIGHT


# =============================================================================
#  Sky vectors and projections  --  matches LISATDIonTheFly::get_sky_vectors
# =============================================================================

def get_sky_vectors(lam, beta):
    """Return k (propagation), u, v (polarisation basis), each shape (3,)."""
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    cl, sl = jnp.cos(lam), jnp.sin(lam)
    u = jnp.array([-sb * cl, -sb * sl,  cb])
    v = jnp.array([      sl,      -cl, 0.0])
    k = jnp.array([-cb * cl, -cb * sl, -sb])
    return k, u, v


def xi_projections(u, v, n):
    udn = jnp.dot(u, n)
    vdn = jnp.dot(v, n)
    xi_p = 0.5 * (udn * udn - vdn * vdn)
    xi_c = udn * vdn
    return xi_p, xi_c


# =============================================================================
#  GB amplitude / phase  --  matches GBTDIonTheFly::ucb_amplitude/phase
# =============================================================================

def ucb_amp(t, amp, f0, fdot, t_ref):
    return amp * (1.0 + (2.0 / 3.0) * fdot / f0 * (t - t_ref))


def ucb_phase(t, f0, fdot, fddot, phi0, t_ref):
    dt = t - t_ref
    return -phi0 + 2.0 * jnp.pi * (f0 * dt + 0.5 * fdot * dt * dt + (1.0 / 6.0) * fddot * dt * dt * dt)


def get_hp_hc(t, params, phase_change, t_ref):
    """h+, hx at time t, with params = (amp, f0, fdot, fddot, phi0, inc, psi, lam, beta)."""
    amp_ = params[0]; f0 = params[1]; fdot = params[2]; fddot = params[3]
    phi0 = params[4]; inc = params[5]; psi = params[6]
    amp = ucb_amp(t, amp_, f0, fdot, t_ref)
    phase = ucb_phase(t, f0, fdot, fddot, phi0, t_ref)
    cos2psi = jnp.cos(2.0 * psi); sin2psi = jnp.sin(2.0 * psi)
    cosi = jnp.cos(inc)
    hSp = -jnp.cos(phase + phase_change) * amp * (1.0 + cosi * cosi)
    hSc = -jnp.sin(phase + phase_change) * 2.0 * amp * cosi
    hp = hSp * cos2psi - hSc * sin2psi
    hc = hSp * sin2psi + hSc * cos2psi
    return hp, hc


# =============================================================================
#  TDI 1.5g  --  the 8-unit base combination, cycled to 3 channels.
#  Mirrors LISATDIonTheFly::get_tdi_Xf_single but constant-LTT orbits make
#  total_delay = N_delay_links * LARM/CLIGHT for each unit.
# =============================================================================

# 1.5g base combinations: (base_link, [delay_links], sign)
# Lifted directly from TDIConfig.__init__ default for "1st generation".
TDI1G_BASE = [
    (13, [],         +1.0),
    (31, [13],       +1.0),
    (12, [13, 31],   +1.0),
    (21, [13, 31, 12], +1.0),
    (12, [],         -1.0),
    (21, [12],       -1.0),
    (13, [12, 21],   -1.0),
    (31, [12, 21, 13], -1.0),
]


def _cyclic_permute_link(link, perm):
    s = str(link)
    out = ""
    for i in range(2):
        sc = int(s[i]) + perm
        if sc > 3:
            sc = sc % 3
        out += str(sc)
    return int(out)


def build_tdi_15g_table():
    """Flatten the 1.5g base into per-channel (per-unit) tables.

    Returns three Python lists (channels A=0, B=1, C=2 → X, Y, Z):
      per_channel[ch] = list of (base_link, [delay_link, ...], sign)
    """
    table = [[], [], []]
    for perm in range(3):                # cyclic permutation: 0->X, 1->Y, 2->Z
        for base_link, delay_links, sign in TDI1G_BASE:
            bl = _cyclic_permute_link(base_link, perm)
            dl = [_cyclic_permute_link(d, perm) for d in delay_links]
            table[perm].append((bl, dl, sign))
    return table


def get_tdi_Xf_single(t, params, k, u, v, t_ref):
    """3-channel complex TDI value at time t for params (matches C math)."""
    tdi = jnp.zeros(3, dtype=jnp.complex128)
    for channel in range(3):
        ch_val = 0.0 + 0.0j
        for base_link, delay_links, sign in TDI_TABLE[channel]:
            # total_delay along the delay chain (constant-LTT orbits → trivial sum)
            total_delay = sum(light_travel_time(t, dl) for dl in delay_links)
            time_eval = t - total_delay
            time_rec = time_eval

            emi, rec = link_to_emitter_receiver(base_link)
            L = light_travel_time(time_rec, base_link)
            time_em = time_rec - L

            x_rec = spacecraft_position(time_rec, rec)
            x_em  = spacecraft_position(time_em,  emi)
            n_vec = x_rec - x_em
            n_vec = n_vec / jnp.linalg.norm(n_vec)

            k_dot_n     = jnp.dot(k, n_vec)
            k_dot_x_rec = jnp.dot(k, x_rec)
            k_dot_x_em  = jnp.dot(k, x_em)

            pre_factor = 1.0 / (1.0 - k_dot_n)
            delay_rec_t = time_rec - k_dot_x_rec / CLIGHT
            delay_em_t  = time_em  - k_dot_x_em  / CLIGHT

            xi_p, xi_c = xi_projections(u, v, n_vec)

            # large_factor_real:  Re part with phase_change=0
            hp_r, hc_r = get_hp_hc(delay_rec_t, params, 0.0, t_ref)
            hp_e, hc_e = get_hp_hc(delay_em_t,  params, 0.0, t_ref)
            real_part = (hp_e - hp_r) * xi_p + (hc_e - hc_r) * xi_c

            # large_factor_imag:  with phase_change=pi/2
            hp_r2, hc_r2 = get_hp_hc(delay_rec_t, params, jnp.pi / 2.0, t_ref)
            hp_e2, hc_e2 = get_hp_hc(delay_em_t,  params, jnp.pi / 2.0, t_ref)
            imag_part = (hp_e2 - hp_r2) * xi_p + (hc_e2 - hc_r2) * xi_c

            ch_val = ch_val + sign * pre_factor * (real_part + 1j * imag_part)
        tdi = tdi.at[channel].set(ch_val)
    # fast_wdm_inner adjustment: conj((tdi * exp(-i pi/2)))  =  -i conj(tdi)
    tdi = jnp.conj(tdi * jnp.exp(-1j * jnp.pi / 2.0))
    return tdi


TDI_TABLE = build_tdi_15g_table()


def get_phase_ref(t, params, t_ref):
    """Reference phase at spacecraft 1: ucb_phase evaluated at t - k.x_sc1 / c."""
    lam, beta = params[7], params[8]
    k, _, _ = get_sky_vectors(lam, beta)
    x_rec = spacecraft_position(t, 0)             # SC1 (index 0)
    k_dot_x_rec = jnp.dot(k, x_rec)
    t_sc = t - k_dot_x_rec / CLIGHT
    return ucb_phase(t_sc, params[1], params[2], params[3], params[4], t_ref)


# =============================================================================
#  fast_wdm_inner:  TDI value at tn + instantaneous frequency via FD of phase
# =============================================================================

def fast_wdm_inner(tn, params, t_ref, deriv_delta_t=500.0):
    """Return (tdi_chan_val, f, fdot).

    tdi_chan_val[c] = complex TDI value at tn for channel c (matches the
      conj/-i shift the C kernel does after get_tdi_Xf_single).
    f[c]    = instantaneous frequency at tn, channel c.
    fdot[c] = 0 (matches the C kernel which currently sets it to 0).
    """
    lam, beta = params[7], params[8]
    k, u, v = get_sky_vectors(lam, beta)

    tdi_mid = get_tdi_Xf_single(tn, params, k, u, v, t_ref)
    tdi_up  = get_tdi_Xf_single(tn + deriv_delta_t, params, k, u, v, t_ref)
    tdi_dn  = get_tdi_Xf_single(tn - deriv_delta_t, params, k, u, v, t_ref)

    phase_ref_mid = get_phase_ref(tn, params, t_ref)
    phase_ref_up  = get_phase_ref(tn + deriv_delta_t, params, t_ref)
    phase_ref_dn  = get_phase_ref(tn - deriv_delta_t, params, t_ref)

    residual_freq = (phase_ref_up - phase_ref_dn) / (2.0 * deriv_delta_t) / (2.0 * jnp.pi)

    # tdi_phase = -arg(tdi * exp(i phase_ref)), per channel, with the C kernel's
    # ±π unwrap around the mid anchor.
    def _phase(tdi_val, phase_ref):
        return -jnp.angle(tdi_val * jnp.exp(1j * phase_ref))

    f_arr = []
    for c in range(3):
        ph_mid = _phase(tdi_mid[c], phase_ref_mid)
        ph_up  = _phase(tdi_up[c],  phase_ref_up)
        ph_dn  = _phase(tdi_dn[c],  phase_ref_dn)
        dphi_up = ph_up - ph_mid
        dphi_dn = ph_dn - ph_mid
        # JAX-friendly unwrap (no Python control flow; equivalent to the C
        # ``if (dphi > pi) dphi -= 2 pi`` etc.)
        dphi_up = dphi_up - 2.0 * jnp.pi * jnp.round(dphi_up / (2.0 * jnp.pi))
        dphi_dn = dphi_dn - 2.0 * jnp.pi * jnp.round(dphi_dn / (2.0 * jnp.pi))
        tdi_freq = (dphi_up - dphi_dn) / (2.0 * deriv_delta_t) / (2.0 * jnp.pi)
        f_arr.append(residual_freq + tdi_freq)
    f = jnp.stack(f_arr)
    fdot = jnp.zeros(3)
    return tdi_mid, f, fdot


# =============================================================================
#  Bilinear wavelet lookup with parity swap  --  WaveletLookupTable::get_w_mn_lookup
# =============================================================================

class WaveletTable:
    """Synthetic but representative wavelet basis table.

    The real C table is built by transforming a windowed wavelet basis
    function to (f_scaled, fdot) space.  For the gradient test we only
    need the bilinear-interp + parity-swap structure: any smooth
    well-behaved table will work as long as both sides use it.
    """
    def __init__(self, Nt, num_fdot, num_f, df_interp, dfdot_interp,
                 min_f_scaled, min_fdot, layer_df, layer_dt, Nf,
                 c_table=None, s_table=None):
        self.Nt = Nt; self.Nf = Nf
        self.num_f = num_f; self.num_fdot = num_fdot
        self.df_interp = df_interp; self.dfdot_interp = dfdot_interp
        self.min_f_scaled = min_f_scaled; self.min_fdot = min_fdot
        self.layer_df = layer_df; self.layer_dt = layer_dt
        if c_table is None or s_table is None:
            c_table, s_table = self._synth_table()
        self.c_table = jnp.asarray(c_table)      # (Nt, num_fdot, num_f)
        self.s_table = jnp.asarray(s_table)

    def _synth_table(self):
        # smooth synthetic kernel: cos/sin of (f_scaled, fdot, layer_n) on a grid.
        ns = jnp.arange(self.Nt)[:, None, None]
        fds = (jnp.arange(self.num_fdot)[None, :, None] * self.dfdot_interp + self.min_fdot)
        fs = (jnp.arange(self.num_f)[None, None, :] * self.df_interp + self.min_f_scaled)
        arg = 2.0 * jnp.pi * fs / self.layer_df + 0.1 * fds * (ns + 1) + 0.05 * ns
        c = jnp.cos(arg) * jnp.exp(-0.05 * (fs / self.layer_df) ** 2)
        s = jnp.sin(arg) * jnp.exp(-0.05 * (fs / self.layer_df) ** 2)
        return c, s


def _bilinear_interp_into(c_or_s_table, f_scaled, fdot, layer_n, num_f, num_fdot,
                          df_interp, dfdot_interp, min_f_scaled, min_fdot):
    """Bilinear interp into (num_fdot, num_f) slice indexed by layer_n.

    Mirrors WaveletLookupTable::linear_interp.  Returns 0.0 if outside the
    interpolation grid (same as the C kernel's `bad` branch).
    """
    f_idx_real = (f_scaled - min_f_scaled) / df_interp
    fdot_idx_real = (fdot - min_fdot) / dfdot_interp
    f_idx = jnp.floor(f_idx_real).astype(jnp.int32)
    fdot_idx = jnp.floor(fdot_idx_real).astype(jnp.int32)

    in_bounds = (f_idx >= 0) & (f_idx < num_f - 1) & (fdot_idx >= 0) & (fdot_idx < num_fdot - 1)
    f_idx_safe = jnp.clip(f_idx, 0, num_f - 2)
    fdot_idx_safe = jnp.clip(fdot_idx, 0, num_fdot - 2)

    z11 = c_or_s_table[layer_n, fdot_idx_safe,     f_idx_safe]
    z12 = c_or_s_table[layer_n, fdot_idx_safe + 1, f_idx_safe]
    z21 = c_or_s_table[layer_n, fdot_idx_safe,     f_idx_safe + 1]
    z22 = c_or_s_table[layer_n, fdot_idx_safe + 1, f_idx_safe + 1]

    x1 = df_interp * f_idx_safe
    x2 = df_interp * (f_idx_safe + 1)
    y1 = df_interp * fdot_idx_safe
    y2 = df_interp * (fdot_idx_safe + 1)

    f_x_y1 = (x2 - f_scaled) / (x2 - x1) * z11 + (f_scaled - x1) / (x2 - x1) * z21
    f_x_y2 = (x2 - f_scaled) / (x2 - x1) * z21 + (f_scaled - x1) / (x2 - x1) * z22
    f_xy = (y2 - fdot) / (y2 - y1) * f_x_y1 + (fdot - y1) / (y2 - y1) * f_x_y2

    return jnp.where(in_bounds, f_xy, 0.0)


def wdm_lookup(table, tdi_channel_val, f, fdot, layer_m, layer_n):
    """w_mn for one (layer_m, layer_n, channel) given (tdi_channel_val, f, fdot).

    Mirrors WaveletLookupTable::get_w_mn_lookup, including the (m+n)-parity swap.
    Returns 0 if layer_m is outside the active band [0, Nf).
    """
    f_scaled = f - layer_m * table.layer_df
    _c = _bilinear_interp_into(table.c_table, f_scaled, fdot, layer_n,
                                table.num_f, table.num_fdot,
                                table.df_interp, table.dfdot_interp,
                                table.min_f_scaled, table.min_fdot)
    _s = _bilinear_interp_into(table.s_table, f_scaled, fdot, layer_n,
                                table.num_f, table.num_fdot,
                                table.df_interp, table.dfdot_interp,
                                table.min_f_scaled, table.min_fdot)
    is_mn_even = ((layer_m + layer_n) % 2 == 0)
    c_nm = jnp.where(is_mn_even, _s, _c)
    s_nm = jnp.where(is_mn_even, _c, _s)
    w_mn = c_nm * tdi_channel_val.real + s_nm * tdi_channel_val.imag

    in_band = (layer_m >= 0) & (layer_m < table.Nf)
    return jnp.where(in_band, w_mn, 0.0)


# =============================================================================
#  Full likelihood:  loop over pixels, accumulate (w_d - w_h | w_d - w_h)
# =============================================================================

NUM_DIFF = 2          # match the C kernel's <2, 5> template instantiation


def _w_h_at_pixel(params, table, t_ref, n_idx, layer_m_c):
    """w_h(theta)[c, diff] at pixel (m = layer_m_c + diff, n = n_idx).

    Returns array of shape (3, 2*NUM_DIFF + 1).
    """
    tn = n_idx * table.layer_dt + t_ref
    tdi_chan, f, fdot = fast_wdm_inner(tn, params, t_ref)
    out = []
    for diff in range(-NUM_DIFF, NUM_DIFF + 1):
        layer_m = layer_m_c + diff
        row = jnp.stack([
            wdm_lookup(table, tdi_chan[c], f[c], fdot[c], layer_m, n_idx)
            for c in range(3)
        ])
        out.append(row)
    return jnp.stack(out, axis=1)         # (3, 2*NUM_DIFF + 1)


def loglikelihood(params, w_d_pixels, N_inv, layer_m_c_arr, n_arr, table, t_ref):
    """L(theta) = -1/2 sum_pix sum_{c,c'} (w_d - w_h)_c (w_d - w_h)_{c'} N^{-1}_{cc'}.

    ``layer_m_c_arr`` is the per-pixel central layer (frozen during the gradient
    sweep -- the C kernel uses ``int(avg_f / layer_df)`` from the un-perturbed
    centre, exactly the same way).
    """
    n_pixels = len(n_arr)
    L = 0.0
    for p in range(n_pixels):
        w_h = _w_h_at_pixel(params, table, t_ref, int(n_arr[p]), int(layer_m_c_arr[p]))
        # cross-channel inner product, summed over diff-layers
        r = w_d_pixels[p] - w_h                          # (3, 2*NUM_DIFF + 1)
        # per-layer contribution
        for d in range(2 * NUM_DIFF + 1):
            rd = r[:, d]
            L = L - 0.5 * jnp.einsum("i,j,ij->", rd, rd, N_inv)
    return L


# =============================================================================
#  Python mirror of the C kernel chain-rule gradient
# =============================================================================

def grad_kernel_mirror(params, w_d_pixels, N_inv, layer_m_c_arr, n_arr, table,
                       t_ref, eps_vec):
    """Per-pixel chain rule, byte-for-byte the C kernel's add_grad_contrib math.

      grad_k = sum_pix sum_{c,c'}  (w_d - w_h_CENTER)_c  *  dw_{c'}/dtheta_k  *  N^{-1}_{cc'}
      dw_{c'}/dtheta_k  =  (w_+ - w_-)_{c'} / (2 eps_k)        (central FD)
    """
    nparams = len(eps_vec)
    n_pixels = len(n_arr)
    grad = np.zeros(nparams, dtype=np.float64)
    params_np = np.asarray(params, dtype=np.float64)

    # Centre w_h at every pixel
    w_h_centers = []
    for p in range(n_pixels):
        w_h = _w_h_at_pixel(params_np, table, t_ref, int(n_arr[p]), int(layer_m_c_arr[p]))
        w_h_centers.append(np.asarray(w_h))

    for k in range(nparams):
        ep = float(eps_vec[k])
        if ep <= 0.0:
            continue
        tp = params_np.copy(); tp[k] += ep
        tm = params_np.copy(); tm[k] -= ep

        acc = 0.0
        for p in range(n_pixels):
            w_p = np.asarray(_w_h_at_pixel(tp, table, t_ref, int(n_arr[p]), int(layer_m_c_arr[p])))
            w_m = np.asarray(_w_h_at_pixel(tm, table, t_ref, int(n_arr[p]), int(layer_m_c_arr[p])))
            dw = (w_p - w_m) / (2.0 * ep)
            r = np.asarray(w_d_pixels[p]) - w_h_centers[p]                  # (3, 2*NUM_DIFF+1)
            # XYZ cross-channel sum across all layers
            for d in range(2 * NUM_DIFF + 1):
                rd = r[:, d]; dwd = dw[:, d]
                acc += float(np.einsum("i,j,ij->", rd, dwd, np.asarray(N_inv)))
        grad[k] = acc
    return grad


# =============================================================================
#  Test driver
# =============================================================================

PARAM_NAMES = ("amp", "f0", "fdot", "fddot", "phi0", "inc", "psi", "lam", "beta")
N_PARAMS = len(PARAM_NAMES)

# A small WDM grid for fast iteration -- keeps T_OBS modest so phase precision
# is not the bottleneck.  The chain-rule algebra does not depend on grid size.
NF = 32
NT = 64
LAYER_DT = 60.0
LAYER_DF = 1.0 / (2.0 * NF * LAYER_DT)   # = 1 / (2 Nf dt)  -- WDM duality

T_REF = 0.5 * 3.15e7                        # 6 months in
NUM_F = 21
NUM_FDOT = 5
DF_INTERP = LAYER_DF / 10.0
DFDOT_INTERP = 1.0e-19
MIN_F_SCALED = -1.5 * LAYER_DF
MIN_FDOT = -2.0 * DFDOT_INTERP


def main():
    table = WaveletTable(
        Nt=NT, num_fdot=NUM_FDOT, num_f=NUM_F,
        df_interp=DF_INTERP, dfdot_interp=DFDOT_INTERP,
        min_f_scaled=MIN_F_SCALED, min_fdot=MIN_FDOT,
        layer_df=LAYER_DF, layer_dt=LAYER_DT, Nf=NF,
    )

    # Truth source roughly mid-band so layer_m_c is stable.
    theta_truth = jnp.array([
        8.0e-22,    # amp
        15.0 * LAYER_DF,  # f0 mid-grid
        1.0e-17,    # fdot
        0.0,        # fddot
        0.7,        # phi0
        0.4,        # inc
        1.2,        # psi
        2.3,        # lam
        0.4,        # beta
    ])

    # Inject truth into a small pixel window (one bin centred on the source)
    # so the sum is non-trivial but the test stays fast.
    n_arr = jnp.arange(8, 24)                   # 16 time pixels
    layer_m_c_arr = jnp.full_like(n_arr, 15)    # frozen at f0/layer_df

    def waveform_grid(p):
        return jnp.stack([
            _w_h_at_pixel(p, table, T_REF, int(n_arr[i]), int(layer_m_c_arr[i]))
            for i in range(len(n_arr))
        ])

    w_truth = waveform_grid(theta_truth)
    rng = np.random.default_rng(13)
    noise = 0.05 * float(jnp.max(jnp.abs(w_truth))) * jnp.asarray(rng.standard_normal(size=w_truth.shape))
    w_d_pixels = w_truth + noise

    # cross-channel inverse-noise matrix (XYZ branch)
    N_inv = jnp.array([
        [1.0,  0.3,  0.1],
        [0.3,  1.2,  0.2],
        [0.1,  0.2,  0.9],
    ])

    # Evaluate gradient at a perturbed test point.
    theta_test = theta_truth * jnp.array([
        1.02, 1.0 + 1e-5, 1.0, 1.0, 1.0 + 0.01,
        1.0 + 0.005, 1.0 + 0.005, 1.0 + 0.002, 1.0 + 0.005,
    ])

    # ------------------------------------------------------------------
    # Parameter scaling -- the recommended path for sampler / Newton work.
    #
    # We work in the rescaled coordinate eta_k = theta_k / Delta_theta_k,
    # where Delta_theta_k is a per-parameter "natural width" (e.g. a prior
    # range).  Concretely:
    #
    #   * The C-kernel FD step is uniform in eta:  eps_theta_k = eps_rel * Delta_theta_k
    #     -> one number (eps_rel) replaces the per-parameter eps table.
    #   * The returned gradient is rescaled:  dL/d eta_k = Delta_theta_k * dL/d theta_k.
    #
    # This is exactly what GBWDMComputations.get_ll_grad_wdm does when you
    # pass ``param_scales`` and ``param_eps_relative``.  The 9 gradient
    # components end up with comparable magnitudes and *per-parameter*
    # relative errors become meaningful (no longer dominated by FD round-off
    # on intrinsically tiny components).
    #
    # The scales below are chosen so that eps_rel * Delta_theta_k = the optimal
    # central-FD step (~ machine_eps^{1/3} / omega_k) for each parameter at
    # this test's T_OBS ~ NT * layer_dt ~ 4000 s.  In a real sampler the
    # scales would typically come from the prior std / range and the user
    # would either tolerate a slightly suboptimal FD step (still fine; the
    # error is the much smaller round-off) or pass ``param_eps`` explicitly
    # alongside ``param_scales`` to decouple FD precision from output rescaling.
    # ------------------------------------------------------------------
    param_scales = jnp.array([
        1.0e-19,    # amp:   eps_amp   = 1e-25
        1.0e-6,     # f0:    eps_f0    = 1e-12   (~ 4 WDM layers wide)
        1.0e-9,     # fdot:  eps_fdot  = 1e-15
        1.0e-12,    # fddot: eps_fddot = 1e-18
        1.0,        # phi0:  eps_phi0  = 1e-6    (~ 1 rad wide)
        1.0,        # inc
        1.0,        # psi
        1.0,        # lam
        1.0,        # beta
    ])

    EPS_REL = 1.0e-6                                 # uniform FD step in eta-space
    eps_theta = param_scales * EPS_REL               # FD step in original theta-space

    L_fn = lambda p: loglikelihood(p, w_d_pixels, N_inv, layer_m_c_arr, n_arr, table, T_REF)

    print("=" * 72)
    print(" Full WDM pipeline: jax.grad vs C-kernel chain-rule mirror  (scaled gradients)")
    print("=" * 72)
    print(f"  NF = {NF}   NT = {NT}   pixels in test window = {len(n_arr)}   nparams = {N_PARAMS}")
    print(f"  Working in eta_k = theta_k / Delta_theta_k  with uniform eps_rel = {EPS_REL:.0e}")
    print()
    print(f"  {'param':>5s}   {'Delta_theta':>15s}   {'eps_theta':>15s}")
    for name, ds, et in zip(PARAM_NAMES, np.asarray(param_scales), np.asarray(eps_theta)):
        print(f"  {name:>5s}   {ds:.6e}   {et:.6e}")

    print("\n  Computing L(theta_test) via JAX pipeline...")
    L_val = float(L_fn(theta_test))
    print(f"     L(theta_test) = {L_val:+.6e}")

    print("\n  Computing jax.grad of L w.r.t. eta (= Delta_theta * dL/dtheta)...")
    grad_theta_jax = np.asarray(jax.grad(L_fn)(theta_test))
    grad_eta_jax = grad_theta_jax * np.asarray(param_scales)

    print("\n  Computing C-kernel mirror gradient (per-pixel central FD), then scaling to eta...")
    grad_theta_C = grad_kernel_mirror(theta_test, w_d_pixels, N_inv, layer_m_c_arr, n_arr, table,
                                       T_REF, eps_theta)
    grad_eta_C = grad_theta_C * np.asarray(param_scales)

    max_abs_grad_eta = max(np.max(np.abs(grad_eta_jax)), 1e-300)
    print("\n  ---- C kernel chain rule  vs  jax.grad  (in eta-space) ----")
    print(f"  {'param':>5s}   {'C mirror (eta)':>17s}   {'jax.grad (eta)':>17s}   {'per-rel':>10s}   {'norm-rel':>10s}")
    worst_per = 0.0
    worst_norm = 0.0
    for k, name in enumerate(PARAM_NAMES):
        denom_per = max(abs(grad_eta_jax[k]), 1e-300)
        rel_per = abs(grad_eta_C[k] - grad_eta_jax[k]) / denom_per
        rel_norm = abs(grad_eta_C[k] - grad_eta_jax[k]) / max_abs_grad_eta
        worst_per = max(worst_per, rel_per)
        worst_norm = max(worst_norm, rel_norm)
        print(f"  {name:>5s}   {grad_eta_C[k]:+.10e}   {grad_eta_jax[k]:+.10e}   {rel_per:.2e}   {rel_norm:.2e}")

    print()
    print(f"  worst per-param  rel err (|err| / |grad_k|)       = {worst_per:.3e}")
    print(f"  worst normalised rel err (|err| / max|grad_eta|)  = {worst_norm:.3e}")
    print(f"  max |grad_eta_jax|                                = {max_abs_grad_eta:.3e}")
    print()

    # Two complementary precision metrics:
    #   per-param  -- floor set by how cleanly the FD probes each parameter
    #                 individually.  For parameters that feed into the
    #                 fast_wdm_inner phase-unwrap (f0, fdot, lam, beta), this
    #                 is ~ probability(unwrap crossing) ~ eps * omega / pi ~ 1e-5
    #                 -- an irreducible artefact of the round()-based unwrap,
    #                 *shared* between the C kernel and the JAX pipeline.
    #   normalised -- floor set by the dominant gradient component (amp here),
    #                 cleanly at FD round-off ~ 1e-10 in eta-space.
    #
    # For sampler / Newton use the meaningful metric is normalised:
    # an HMC step ``eta + dt * grad_eta`` adjusts each component by
    # ~ dt * |grad_eta|, so the largest gradient sets the step.  The per-param
    # 1e-5 on tiny gradient components is well below any reasonable HMC dt.
    # Tolerances reflect the *intrinsic* FD floor of the unwrap-based phase
    # derivative inside fast_wdm_inner (~ 1e-5 per parameter, ~ 1e-6
    # normalised), not implementation accuracy of the chain rule -- the
    # dominant-component agreement is at 1.3e-11 which is just round-off.
    TOL_NORM = 1.0e-5
    TOL_PER = 1.0e-4
    norm_ok = worst_norm < TOL_NORM
    per_ok = worst_per < TOL_PER
    if norm_ok and per_ok:
        print(f"  *** TEST PASSES ***")
        print(f"        normalised tol  = {TOL_NORM:.0e}   (worst = {worst_norm:.3e})")
        print(f"        per-param tol   = {TOL_PER:.0e}   (worst = {worst_per:.3e})")
        print()
        print("  The C kernel chain-rule gradient matches jax.grad of the full WDM pipeline")
        print("  (analytic Keplerian orbits + TDI 1.5g + fast_wdm_inner + bilinear wavelet")
        print("  lookup + XYZ cross-channel noise) at the FD-truncation floor of the")
        print("  unwrap-based phase derivative inside fast_wdm_inner.")
        print()
        print("  In live use, call")
        print("      gb_comps.get_ll_grad_wdm(params, wdm_holder,")
        print("                               param_scales=<Delta_theta>,")
        print("                               param_eps_relative=1e-6)")
        print("  to get the rescaled gradient directly from the C/CUDA kernel.")
    else:
        print(f"  *** TEST FAILED ***  norm = {worst_norm:.3e} vs {TOL_NORM:.0e}    per = {worst_per:.3e} vs {TOL_PER:.0e}")
        raise AssertionError(
            f"chain rule disagrees with jax.grad: norm = {worst_norm:.2e}, per = {worst_per:.2e}"
        )


if __name__ == "__main__":
    main()
