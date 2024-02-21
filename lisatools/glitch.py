from lisaconstants import *
import numpy as np

co = C_SI


def HeavisideTheta(x, xp=None):
    if xp is None:
        xp = np

    squeeze = True if x.ndim == 1 else False

    x = xp.atleast_1d(x)

    out = 1.0 * (x >= 0.0)

    if squeeze:
        out = out.squeeze()

    return out


def DiracDelta(x, xp=None):
    if xp is None:
        xp = np
    squeeze = True if x.ndim == 1 else False

    x = xp.atleast_1d(x)

    # 1e-10 to guard against numerical error
    out = 1.0 * (xp.abs(x) < 1e-10)

    if squeeze:
        out = out.squeeze()

    return out


tau_2_default = 5585.708541201614
Deltav_default = 1.1079114425597559 * 10 ** (-9)

tau_2_default = 1.9394221536001746
Deltav_default = 2.22616837 * 10 ** (-11)
t0 = 600


def tdi_glitch_XYZ1(
    t_in, T=8.3, tau_2=tau_2_default, Deltav=Deltav_default, t0=600, mtm=1.982, xp=None
):
    # def tdi_glitch_XYZ1(t_in, T=8.3, tau_1=480.0, tau_2=100.0, Deltav=1e-12, t0=600.0, mtm=1.982, xp=None):

    if xp is None:
        xp = np

    tau_2 = xp.atleast_1d(tau_2)
    t0 = xp.atleast_1d(t0)
    Deltav = xp.atleast_1d(Deltav)

    assert tau_2.shape == t0.shape == Deltav.shape

    out = xp.zeros((3, len(t0), len(t_in)))

    run = ~(
        xp.isinf(xp.exp((-t_in[None, :] + t0[:, None])))
        | xp.isnan(xp.exp((-t_in[None, :] + t0[:, None])))
    )

    tdiX1link12 = (
        mtm
        * Deltav[:, None]
        * (
            t_in[None, :]
            - t0[:, None]
            - 2 * tau_2[:, None]
            + xp.exp((-t_in[None, :] + t0[:, None]) / tau_2[:, None])
            * (t_in[None, :] - t0[:, None] + 2 * tau_2[:, None])
        )
        * DiracDelta(t_in[None, :] - t0[:, None])
        + mtm
        * Deltav[:, None]
        * (
            1
            + (
                xp.exp((-t_in[None, :] + t0[:, None]) / tau_2[:, None])
                * (-t_in[None, :] + t0[:, None] - tau_2[:, None])
            )
            / tau_2[:, None]
        )
        * HeavisideTheta(t_in[None, :] - t0[:, None])
    ) / co - (
        mtm
        * Deltav[:, None]
        * (
            t_in[None, :]
            + 4 * T
            - t0[:, None]
            - 2 * tau_2[:, None]
            + xp.exp((-t_in[None, :] - 4 * T + t0[:, None]) / tau_2[:, None])
            * (t_in[None, :] + 4 * T - t0[:, None] + 2 * tau_2[:, None])
        )
        * DiracDelta(t_in[None, :] + 4 * T - t0[:, None])
        + mtm
        * Deltav[:, None]
        * (
            1
            - (
                xp.exp((-t_in[None, :] - 4 * T + t0[:, None]) / tau_2[:, None])
                * (t_in[None, :] + 4 * T - t0[:, None] + tau_2[:, None])
            )
            / tau_2[:, None]
        )
        * HeavisideTheta(t_in[None, :] + 4 * T - t0[:, None])
    ) / co

    tdiY1link12 = (
        -2
        * (
            mtm
            * Deltav[:, None]
            * (
                t_in[None, :]
                + T
                - t0[:, None]
                - 2 * tau_2[:, None]
                + (t_in[None, :] + T - t0[:, None] + 2 * tau_2[:, None])
                / xp.exp((t_in[None, :] + T - t0[:, None]) / tau_2[:, None])
            )
            * DiracDelta(t_in[None, :] + T - t0[:, None])
            + mtm
            * Deltav[:, None]
            * (
                1
                - (t_in[None, :] + T - t0[:, None] + tau_2[:, None])
                / (
                    xp.exp((t_in[None, :] + T - t0[:, None]) / tau_2[:, None])
                    * tau_2[:, None]
                )
            )
            * HeavisideTheta(t_in[None, :] + T - t0[:, None])
        )
    ) / co + (
        2
        * (
            mtm
            * Deltav[:, None]
            * (
                t_in[None, :]
                + 3 * T
                - t0[:, None]
                - 2 * tau_2[:, None]
                + xp.exp((-t_in[None, :] - 3 * T + t0[:, None]) / tau_2[:, None])
                * (t_in[None, :] + 3 * T - t0[:, None] + 2 * tau_2[:, None])
            )
            * DiracDelta(t_in[None, :] + 3 * T - t0[:, None])
            + mtm
            * Deltav[:, None]
            * (
                1
                - (
                    xp.exp((-t_in[None, :] - 3 * T + t0[:, None]) / tau_2[:, None])
                    * (t_in[None, :] + 3 * T - t0[:, None] + tau_2[:, None])
                )
                / tau_2[:, None]
            )
            * HeavisideTheta(t_in[None, :] + 3 * T - t0[:, None])
        )
    ) / co

    tdiZ1link12 = xp.zeros_like(tdiX1link12)

    tdiX1link12[~run] = 0.0
    tdiY1link12[~run] = 0.0
    tdiZ1link12[~run] = 0.0

    out[0, run] = tdiX1link12[run]
    out[1, run] = tdiY1link12[run]
    out[2, run] = tdiZ1link12[run]

    return out


if __name__ == "__main__":
    dt = 1 / 0.3
    t = np.arange(2e3) * dt
    check = tdi_glitch_XYZ1(
        t
    )  # , T=8.3, tau_1=480.0, tau_2=100.0, Deltav=1e-12, t0=600.0, mtm=1.982)
    import matplotlib.pyplot as plt

    plt.loglog(np.fft.rfftfreq(len(check[0]), dt), np.abs(np.fft.rfft(check[0])))
    plt.loglog(np.fft.rfftfreq(len(check[0]), dt), np.abs(np.fft.rfft(check[1])))
    plt.savefig("plot1.png")
    breakpoint()
