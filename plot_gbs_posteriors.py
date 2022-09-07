

import sys

# make the plots look a bit nicer with some defaults
import matplotlib.pyplot as plt
import matplotlib as mpl
rcparams = {}
rcparams['axes.linewidth'] = 0.5
rcparams['font.family'] = 'serif'
rcparams['font.size'] = 18
rcparams['legend.fontsize'] = 14
rcparams['mathtext.fontset'] = "stix"
mpl.rcParams.update(rcparams) # update plot parameters

usetex=True
useserif=True
plottempsetc=False

mpl.rcParams['text.usetex']=usetex

import numpy as np
import glob
import pickle
from chainconsumer import ChainConsumer

# Set the path to Eryn (not using intallations just yet)
ERYNPATH = '/home/karnesis/work/Git/Eryn'
# ERYNPATH = '/Users/nikos/work/Git/LISA/Eryn'
SAVEDIR  = '/home/karnesis/work/Git/LISAanalysistools/'
PLOTDIR  = '/home/karnesis/work/Git/LISAanalysistools/'
DOPLOT   = True
SAVEDATA = True
USEDR    = False
# clrpltt  = 'plasma'
nbrsmx = 20
truenumberofgbs = 10
burnin=100
thin=10
ndim_gbs=8

#gbs_parameters = [r"$A^\ast$", r"$f_0~[\mathrm{mHz}]$", r"$\dot{f}_0$", r"$\phi_0$", r"$\cos\iota$", r"$\psi$", r"$\lambda$", r"$\sin\beta$"]
gbs_parameters = [r"$\rho$", r"$f_\mathrm{gw}~[\mathrm{mHz}]$", r"$\dot{f}_0$", r"$\phi_0$", r"$\cos\iota$", r"$\psi$", r"$\lambda$", r"$\sin\beta$"]
# r"$\ddot{f}_0$"
sys.path.append(ERYNPATH)
from eryn.backends import HDFBackend

# get input arguments
argv = sys.argv

scriptname = argv[0].split('.')[0]

print(' * Running {}.py'.format(scriptname))

print(' * Eryn path: {}'.format(ERYNPATH))

if len(argv) < 3:
    sys.exit(" ### Error: I need 2 inputs: Some hint for the h5, also a tag ... ")
else:
    h5file = str(argv[1])
    tag = str(argv[2])

# data = {}
# for fname in glob.glob('{}{}*.pkl'.format(SAVEDIR, acratefile)):
#     print('loading ', fname)
#     with open(fname, 'rb') as filehandle:
#         d = pickle.load(filehandle)
#         data[d['a'].shape[1]] = d

# # Sort the items
# data = dict(sorted(data.items()))
# mintemps = np.min(np.array(list(data.keys())))

# try:
#     print(data[mintemps]['a'].shape, data[mintemps]['ar'].shape)
# except:
#     print('Did not manage to print yeyoo')

# palette = plt.cm.plasma(np.linspace(0,1, len(data.keys())))
# # palette  = sns.color_palette(clrpltt, len(data.keys()))
# nwalkers = len(data[mintemps]['a'][0,0,:])

# #print(' * Palette : {}'.format(clrpltt))
# print(' * N walkers: {}'.format(nwalkers))

# print(' * Plotting a for each T')

# for T in [0]:
#     plt.figure(figsize=(9,6))
#     for i, temp in enumerate(data):
#         plt.plot(data[temp]['a'][:,T,0], label='T={}, $n_T={}$'.format(T, temp), color=palette[i])

#     # plt.yscale('log')
#     # plt.legend(loc='upper right', frameon=False)
#     plt.xlim(0, data[temp]['a'].shape[0])
#     plt.xlabel('Samples')
#     plt.ylabel('Acc. rate  (in model)')
#     plt.savefig('{}mh_acc_rate_{}_T{}Nw{}.pdf'.format(PLOTDIR, tag, temp, nwalkers), bbox_inches='tight', dpi=600)

# ntemps = np.min(np.array(list(data.keys())))

# print(' * Plotting a prime for each T')

# for T in range(ntemps):
#     plt.figure(figsize=(9,6))
#     for i, temp in enumerate(data):
#         plt.plot(data[temp]['ar'][:,T,0], label='T={}, $n_T={}$'.format(T, temp), color=palette[i])

#     plt.legend(loc='upper right', frameon=False)
#     plt.xlim(0, data[temp]['ar'].shape[0])
#     plt.xlabel('Samples')
#     plt.ylabel('Acc. rate  (between model)')
#     plt.savefig('{}rj_acc_rate_{}_T{}Nw{}.pdf'.format(PLOTDIR, tag, T, nwalkers), bbox_inches='tight', dpi=600)

# plt.close('all')

# palette = plt.cm.plasma(np.linspace(0,1, len(data.keys())))
# # palette  = sns.color_palette(clrpltt, len(data.keys()))

# print(' * Plotting swap acc rate for each T')

# for T in range(ntemps-1):
#     plt.figure(figsize=(9,6))
#     for i, temp in enumerate(data):
#         #print('1:', palette[i])
#         #print('2:', data[temp]['ta'][:,T].shape)
#         plt.plot(data[temp]['ta'][:,T]/nwalkers, label='T={}, $n_T={}$'.format(T, temp), color=palette[i])

#     # plt.yscale('log')
#     plt.legend(loc='upper right', frameon=False)
#     plt.xlim(0, data[temp]['ta'].shape[0])
#     plt.xlabel('Samples')
#     plt.ylabel('Acc. Swap rate per waker (in model)')
#     plt.savefig('{}mh_swap_rate_{}_T{}Nw{}.pdf'.format(PLOTDIR, tag, T, nwalkers), bbox_inches='tight', dpi=600)

# plt.close('all')

# print(' * Plotting rj swap acc rate for each T')

# #for i, temp in enumerate(data):
# #    plt.figure(figsize=(6,6))    
#     # clrs = sns.color_palette('plasma', temp-1)
# #    clrs = plt.cm.plasma(np.linspace(0,1, temp-1))
# #    for T in range(temp-1):
# #        plt.plot(data[temp]['ra'][:,T]/nwalkers, label='T={}, $n_T={}$'.format(T, temp), color=clrs[T], alpha=0.9, lw=2)
# #    plt.xlim(0, data[temp]['ra'].shape[0])
# #    plt.xlabel('Samples')
# #    plt.ylabel('Acc. Swap rate per waker (between model)')
# #    plt.savefig('{}rj_swap_rate_{}_Nt{}Nw{}.pdf'.format(PLOTDIR, tag, temp, nwalkers), bbox_inches='tight', dpi=600)
# #    plt.show()
# #plt.close('all')

# LOAD the BACKEND
backend = HDFBackend(SAVEDIR + h5file)

# # info = backend.get_info(discard=0, thin=1)  # NOTE: Here we want to see how the temperatures adjust, thus discard=1
# allbetas = backend.get_value('betas', discard=0, thin=1)

# # make a trace plot for each temperature
# ntemps = allbetas.shape[1]

# # Define a colormap
# clrs = plt.cm.plasma(np.linspace(0, 1, ntemps))

# print(' * Plotting T chains')

# fig = plt.figure(figsize=(12, 6))
# plt.ylabel(r"$\beta$")
# plt.xlabel("Samples")

# # Loop over the temperatures
# for temp in range(1, ntemps - 1):
#     # get the samples to plot
#     betas = allbetas[:, temp].flatten()
#     # Build the figure.
#     plt.plot(
#         np.arange(0, betas.shape[0]),
#         betas,
#         color=clrs[temp],
#         alpha=0.9,
#         label=r"$\beta_{{{}}}$".format(temp),
#     )
#     plt.xlim(0, betas.shape[0])
# plt.savefig('{}temperature_chains_{}_Nt{}Nw{}.pdf'.format(PLOTDIR, tag, temp, nwalkers), bbox_inches='tight', dpi=600)
# plt.close('all')

# # Get the accrate data corresponding to the particular h5
# mydata = data[ntemps]

# print(' * Plotting T, ta, ra')

# if plottempsetc:

#     # Make a plot with the swap acc rate
#     fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

#     # Loop over the temperatures
#     for temp in range(1, ntemps - 1):
#         # get the samples to plot
#         betas = allbetas[:, temp].flatten()
#         # Build the figure.
#         ax1.plot(
#         np.arange(0, betas.shape[0]),
#         betas,
#         color=clrs[temp],
#         alpha=0.9,
#         label=r"$\beta_{{{}}}$".format(temp),
#         )
#         plt.xlim(1, betas.shape[0])

#     ax1.set_xscale('log')
#     ax1.set_ylabel(r"$\beta$")


#     # Loop over the temperatures
#     for temp in range(1, ntemps - 1):
#         # Build the figure.
#         ax2.plot(
#         np.arange(0, betas.shape[0]),
#         mydata['ta'][burnin:,temp]/nwalkers,
#         color=clrs[temp],
#         alpha=0.9,
#         label=r"$T_{{{}}}$".format(temp),
#         )
#         plt.xlim(1, betas.shape[0])

#     ax2.set_ylabel(r"$\alpha_{i,j}$")
#     ax2.set_xscale('log')


#     # Loop over the temperatures
#     for temp in range(1, ntemps - 1):
#         # Build the figure.
#         ax3.plot(
#         np.arange(0, betas.shape[0]),
#         mydata['ra'][burnin:,temp]/nwalkers,
#         color=clrs[temp],
#         alpha=0.9,
#         label=r"$T_{{{}}}$".format(temp),
#         )
#         plt.xlim(1, betas.shape[0])

#     ax3.set_ylabel(r"$\alpha\prime_{i,j}$")
#     ax3.set_xlabel("Samples")
#     ax3.set_xscale('log')

#     plt.savefig('{}temp_and_swap_acc_rates_chains_{}_Nt{}Nw{}.pdf'.format(PLOTDIR, tag, temp, nwalkers), bbox_inches='tight', dpi=600)
#     plt.close('all')

# print(' * Plotting T, a, ar')

# if plottempsetc:
#     # Make a second on RJ acc rate
#     # Make a nice subplot together with the acceptance rate
#     fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

#     # Loop over the temperatures
#     for temp in range(1, ntemps - 1):
#         # get the samples to plot
#         betas = allbetas[:, temp].flatten()
#         # Build the figure.
#         ax1.plot(np.arange(0, betas.shape[0]),betas,
#         color=clrs[temp],
#         alpha=0.9,
#         label=r"$\beta_{{{}}}$".format(temp),
#         )
#         plt.xlim(1, betas.shape[0])

#     ax1.set_xscale('log')
#     ax1.set_ylabel(r"$\beta$")


#     # Loop over the temperatures
#     for temp in range(1, ntemps - 1):
#         # Build the figure.
#         ax2.plot(
#         np.arange(0, betas.shape[0]),
#         mydata['a'][burnin:,temp,0],
#         color=clrs[temp],
#         alpha=0.9,
#         label=r"$T_{{{}}}$".format(temp),
#         )
#         plt.xlim(1, betas.shape[0])

#     ax2.set_ylabel(r"$\alpha$")
#     ax2.set_xscale('log')


#     # Loop over the temperatures
#     for temp in range(1, ntemps - 1):
#          # Build the figure.
#         ax3.plot(
#         np.arange(0, betas.shape[0]),
#         mydata['ar'][burnin:,temp,0],
#         color=clrs[temp],
#         alpha=0.9,
#         )
#         plt.xlim(1, betas.shape[0])

#     ax3.set_ylabel(r"$\alpha\prime$")
#     ax3.set_xlabel("Samples")
#     ax3.set_xscale('log')

#     plt.savefig('{}temp_and_all_acc_rates_chains_{}_Nt{}Nw{}.pdf'.format(PLOTDIR, tag, temp, nwalkers), bbox_inches='tight', dpi=600)
#     plt.close('all')


#     print(' * Plotting paper plot')

#     # Make a third for the paper 
#     # Make a plot with the swap acc rate
#     fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(15, 5), sharex=True)

#     # Loop over the temperatures
#     for temp in range(1, ntemps - 1):
#         # get the samples to plot
#         betas = allbetas[:, temp].flatten()
#         # Build the figure.
#         ax1.plot(
#         np.arange(0, betas.shape[0]),
#         betas,
#         color=clrs[temp],
#         alpha=0.9,
#         label=r"$\beta_{{{}}}$".format(temp),
#         )
#         plt.xlim(1, betas.shape[0])

#     ax1.set_xscale('log')
#     ax1.set_ylabel(r"$\beta$")

#     # Loop over the temperatures
#     for temp in range(1, ntemps - 1):
#         # Build the figure.
#         ax2.plot(
#         np.arange(0, betas.shape[0]),
#         mydata['ta'][burnin:,temp]/nwalkers,
#         color=clrs[temp],
#         alpha=0.9,
#         )
#         plt.xlim(1, betas.shape[0])

#     ax2.set_ylabel(r"$\alpha_{i,j}/n_w$")
#     ax2.set_xscale('log')


#     # Loop over the temperatures
#     for temp in range(1, ntemps - 1):
#         # Build the figure.
#         ax3.plot(np.arange(0, betas.shape[0]),
#         mydata['a'][burnin:,temp,0],
#         color=clrs[temp],
#         alpha=0.9)
#         plt.xlim(1, betas.shape[0])

#     ax3.set_ylabel(r"$\alpha$")
#     ax3.set_xscale('log')
#     ax3.set_xlabel("Samples")

#     plt.savefig('{}paper_temp_and_swap_acc_rates_chains_{}_Nt{}Nw{}.pdf'.format(PLOTDIR, tag, temp, nwalkers), bbox_inches='tight', dpi=600)
#     plt.close('all')

def get_clean_k_chains(backend, temp=0):
    inds = backend.get_value("inds")  # Get the leaves out
    branches = {name: np.sum(inds[name], axis=-1, dtype=int) for name in inds}
    bns = (np.arange(1, nbrsmx + 2) - 0.5)  # Get maximum allowed number of leaves for the given branch 
    for (branch) in (branches):  # Get the total number of components/branches per temperature
        if branch == list(branches.keys())[0]:
            k_chain = branches[branch][:, temp].flatten()
        else:
            k_chain += branches[branch][:, temp].flatten()
    return k_chain, bns

gbs_k_chain_baseline, bns = get_clean_k_chains(backend)
print(gbs_k_chain_baseline)

fig = plt.figure(figsize=(5, 5))
plt.hist(
        gbs_k_chain_baseline-1,
        bins=bns,
        color='dodgerblue',
        # edgecolor='cornflowerblue',
        alpha=0.9,
        lw=2,
        histtype='step',
        density=True,
        hatch='////'
        # label='No DR'
        )

plt.axvline(x=int(truenumberofgbs), linestyle='--', lw=2, color='crimson')

# plt.hist(
#         gauss_k_chain_dr,
#         bins=bns,
#         color='crimson',
#         edgecolor='crimson',
#         alpha=0.2,
#         lw=2,
#         density=True,
#         label='DR'
#         )

# plt.legend(loc='upper left')
#plt.xticks(np.arange(0, nbrsmx))
plt.xticks(fontsize=12)
plt.yticks([])
# plt.yscale('log')
plt.xlim(5, nbrsmx+5)
plt.xlabel("$\#$ of sources in the data")
plt.savefig(PLOTDIR + "{}_k_hist.pdf".format(tag), bbox_inches='tight', dpi=600)
#plt.show()

plt.close('all')

print(' * Plotting posteriors')
# Plot posteriors below

def get_clean_chain(coords, ndim, temp=0):
    print(' - Shape of coords: {}'.format(coords[:, temp, :, :, :].shape))
    naninds    = np.logical_not(np.isnan(coords[:, temp, :, :, 0].flatten()))
    samples_in = np.zeros((coords[:, temp, :, :, 0].flatten()[naninds].shape[0], ndim))  # init the chains to plot
    # get the samples to plot
    for d in range(ndim):
        givenparam = coords[:, temp, :, :, d].flatten()
        samples_in[:, d] = givenparam[
            np.logical_not(np.isnan(givenparam))
        ]  # Discard the NaNs, each time they change the shape of the samples_in
    return samples_in

temp = 0 

samples_gbs_baseline = get_clean_chain(backend.get_chain(thin=thin)['gb'], ndim_gbs)

c = ChainConsumer()

c.add_chain(samples_gbs_baseline[:,:2], parameters=gbs_parameters[:2]) # , name='UCBs'

c.configure(bar_shade=True, summary=False, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=usetex, serif=useserif)
c.plotter.plot_summary(figsize=4, filename=PLOTDIR + "{}_gbs_posteriors_1_v2.pdf".format(tag),);

from lisatools.sampling.prior import AmplitudeFromSNR
L = 2.5e9
Tobs = 1.0 * 31457280.0
amp_transform = AmplitudeFromSNR(L, Tobs)

try:
    out_params = np.load("out_params2.npy")
    out_params[:, 3] = 0.0
    amp_in, f0_in = out_params.T.copy()[:2]
    A = amp_transform.forward(amp_in, f0_in)[0]
    for ii in range(len(A)):
        c.add_marker([A[ii], f0_in[ii]*1e3], \
        parameters=gbs_parameters[:2], marker_style="x", \
        marker_size=100, color="#F4ABF6") # #DC143C DAF7A6

    fig = c.plotter.plot(filename=PLOTDIR + "{}_gbs_posteriors_2_v2.pdf".format(tag), legend=False);
except:
    print('Did not manage to plot chain consumer 1 yeyoo')

plt.close('all')






### Plot all of them, or a subset

plt.close('all')

c = ChainConsumer()

# reminder: gbs_parameters = [A, f0, fd, p0, cosiota, psi, lam, sinbeta]
paramindices = [0, 1, 4, 6, 7]

# this is stupid!!!! Im so angry
prms = [] 
for ind in paramindices:
    prms.append(gbs_parameters[ind])

print(samples_gbs_baseline.shape)

c.add_chain(samples_gbs_baseline[:,paramindices], parameters=prms) # , name='UCBs'

c.configure(bar_shade=True, summary=False, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=usetex, serif=useserif)

try:
    amp_in, f0_in, _, _, _, iota_in, _, lam_in, beta_sky_in = out_params.T.copy()
    A = amp_transform.forward(amp_in, f0_in)[0]
    for ii in range(len(A)):
        c.add_marker([A[ii], f0_in[ii]*1e3, np.cos(iota_in[ii]), lam_in[ii], np.sin(beta_sky_in[ii])], \
        parameters=prms, marker_style="x", \
        marker_size=100, color="#F4ABF6") # #DC143C DAF7A6

    fig = c.plotter.plot(filename=PLOTDIR + "{}_gbs_posteriors_all_v1.pdf".format(tag), legend=False);
except:
    print('Did not manage to plot chain consumer 2 yeyoo')

plt.close('all')

print('### DONE!')
