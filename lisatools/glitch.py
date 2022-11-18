
from lisatools.utils.constants import *
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
Deltav_default = 1.1079114425597559*10**(-9)
    
def tdi_glitch_XYZ1(t_in, T=8.3, tau_2=tau_2_default, Deltav=Deltav_default, t0=600, mtm=1.982, xp=None):    

#def tdi_glitch_XYZ1(t_in, T=8.3, tau_1=480.0, tau_2=100.0, Deltav=1e-12, t0=600.0, mtm=1.982, xp=None):

    if xp is None:
        xp = np

    out = xp.zeros((3, len(t_in)))

    run = ~(xp.isinf(xp.exp((-t_in + t0))) | xp.isnan(xp.exp((-t_in + t0))))

    t = t_in[run]

    tdiX1link12 =  (mtm*Deltav*(t - t0 - 2*tau_2 + xp.exp((-t + t0)/tau_2)*(t - t0 + 2*tau_2))*DiracDelta(t - t0) +  mtm*Deltav*(1 + (xp.exp((-t + t0)/tau_2)*(-t + t0 - tau_2))/tau_2)*HeavisideTheta(t - t0))/co - (mtm*Deltav*(t + 4*T - t0 - 2*tau_2 + xp.exp((-t - 4*T + t0)/tau_2)*(t + 4*T - t0 + 2*tau_2))*DiracDelta(t + 4*T - t0) +  mtm*Deltav*(1 - (xp.exp((-t - 4*T + t0)/tau_2)*(t + 4*T - t0 + tau_2))/tau_2)*HeavisideTheta(t + 4*T - t0))/co
    


    tdiY1link12 =    (-2*(mtm*Deltav*(t + T - t0 - 2*tau_2 + (t + T - t0 + 2*tau_2)/xp.exp((t + T - t0)/tau_2))*DiracDelta(t + T - t0) + mtm*Deltav*(1 - (t + T - t0 + tau_2)/(xp.exp((t + T - t0)/tau_2)*tau_2))*HeavisideTheta(t + T - t0)))/co + (2*(mtm*Deltav*(t + 3*T - t0 - 2*tau_2 + xp.exp((-t - 3*T + t0)/tau_2)*(t + 3*T - t0 + 2*tau_2))*DiracDelta(t + 3*T - t0) +  mtm*Deltav*(1 - (xp.exp((-t - 3*T + t0)/tau_2)*(t + 3*T - t0 + tau_2))/tau_2)*HeavisideTheta(t + 3*T - t0)))/co

    tdiZ1link12 = xp.zeros_like(tdiX1link12)

    out[0, run] = tdiX1link12
    out[1, run] = tdiY1link12
    out[2, run] = tdiZ1link12

    return out



if __name__ == "__main__":
    dt = 1/0.3
    t = np.arange(2e3) * dt
    check = tdi_glitch_XYZ1(t)  # , T=8.3, tau_1=480.0, tau_2=100.0, Deltav=1e-12, t0=600.0, mtm=1.982)
    import matplotlib.pyplot as plt
    plt.loglog(np.fft.rfftfreq(len(check[0]), dt), np.abs(np.fft.rfft(check[0])))
    plt.loglog(np.fft.rfftfreq(len(check[0]), dt), np.abs(np.fft.rfft(check[1])))
    plt.savefig("plot1.png")
    breakpoint()
