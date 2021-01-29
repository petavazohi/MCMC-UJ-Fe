# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 19:13:57 2020

@author: ptava
"""


import numpy as np
import matplotlib.pylab as plt
from scipy import stats
import matplotlib.gridspec as gridspec
import matplotlib.font_manager 
from matplotlib.legend import Legend
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.default"] = "regular"
plt.rc('font', size=16)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize


lda = np.loadtxt('LDA.txt')  # [1500:]
pbe = np.loadtxt('PBE.txt')  # [2000:]
pbesol = np.loadtxt('PBEsol.txt')  # [2000:]


lda = lda[lda[:, 0] < lda[:, 1]]
pbe = pbe[pbe[:, 0] < pbe[:, 1]]
pbesol = pbesol[pbesol[:, 0] < pbesol[:, 1]]

xcs = ['LDA', 'PBEsol', 'PBE']

main_dict = {
    'LDA': {
        'x': lda[:, 1],
        'y': lda[:, 0],
        'cmap': 'Blues',
        'step': 0.09,
        'linestyle': "-.",
        'alpha': 1,
        'color': 'blue',
        'loc': 4},
    'PBE': {
        'x': pbe[:, 1],
        'y': pbe[:, 0],
        'data': pbe,
        'cmap': 'Greens',
        'step': 1.00,
        'linestyle': "solid",
        'alpha': 1,
        'color': 'green',
        'loc': 3},
    'PBEsol': {
        'x': pbesol[:, 1],
        'y': pbesol[:, 0],
        'cmap': 'Reds',
        'step': 0.10,
        'linestyle': 'dashed',
        'alpha': 1,
        'color': 'red',
        'loc': 8}}

lines = []
legends = []
legend_frames = []
legend_texts = []

# fig = plt.figure(constrained_layout=True, figsize=(14, 8))
# heights = [4, 10]
# widths = [5, 2]

# gs = gridspec.GridSpec(2, 2, width_ratios=widths,
#                        height_ratios=heights, wspace=0.0, hspace=0.0)
# #gs.update(wspace=0.001, hspace=0.001)
# axes = []
# ax_2d = fig.add_subplot(gs[1, 0])  # , sharex=True, sharey=True)
# ax_U = fig.add_subplot(gs[0, 0])
# ax_J = fig.add_subplot(gs[1, 1])
# ax_legend = fig.add_subplot(gs[0, 1])

fig = plt.figure(constrained_layout=True, figsize=(10, 10))


# gs = gridspec.GridSpec(15, 15, wspace=0.0, hspace=0.0)
# # gs.update(wspace=0.001, hspace=0.001)
# axes = []
# # , sharex=True, sharey=True)
# ax_2d = fig.add_subplot(gs[4:, :-2], aspect='equal')
# ax_U = fig.add_subplot(gs[:4, :-2], aspect='equal')
# ax_J = fig.add_subplot(gs[4:, -2:], aspect='equal')
# ax_legend = fig.add_subplot(gs[-1, 1:-2], aspect='equal')
# ax_2d = fig.add_subplot(gs[4:, :-2], aspect='equal')
# ax_U = fig.add_subplot(gs[:4, :-2], aspect='equal')
# ax_J = fig.add_subplot(gs[4:, -2:], aspect='equal')
# ax_legend = fig.add_subplot(gs[-1, 1:-2], aspect='equal')

nx, ny = 10, 10
gs = gridspec.GridSpec(nx,
                       ny,
                       wspace=0.3,
                       hspace=0.3,
                       width_ratios=[1] * ny,
                       height_ratios=[1] * nx)
# gs.update(wspace=0.001, hspace=0.001)
axes = []
# , sharex=True, sharey=True)
ax_2d = fig.add_subplot(gs[5:, :-2])
ax_U = fig.add_subplot(gs[3:5, :-2])
ax_J = fig.add_subplot(gs[5:, -2:])
ax_legend = fig.add_subplot(gs[-1, :-2])


xlim = [2.5, 7.75]
ylim = [0.5, 3.7]


npoints = 5000
for ixc in xcs:
    x = main_dict[ixc]['x']
    span = np.linspace(x.min(), x.max(), npoints)
    kernel = stats.gaussian_kde(x)
    distribution = kernel(span)
    # dx = span[1] - span[0]
    # integral = distribution * dx
    # integral = integral.sum()
    # print(integral)
    ax_U.plot(span, distribution, color=main_dict[ixc]['color'])
ax_U.set_xlim(xlim[0], xlim[1])
ax_U.set_ylim(0, )
fig.patch.set_visible(False)
# ax_U.axis('off')
# ax_U.set_xticks([])
# ax_U.set_yticks([])
ax_U.set_ylabel('$\mathcal{P}_{\mathrm{U}}$(1/eV)')

ax_U.xaxis.set_minor_locator(MultipleLocator(0.1))
ax_U.xaxis.set_major_locator(MultipleLocator(1.0))
ax_U.yaxis.set_minor_locator(MultipleLocator(0.1))
ax_U.yaxis.set_major_locator(MultipleLocator(1.0))

ax_U.tick_params(
    which='major',
    axis="y",
    direction="inout",
    width=1,
    length=5,
    labelright=False,
    right=True,
    left=True)  # ,labelsize='x-large')

ax_U.tick_params(
    which='major',
    axis="x",
    direction="inout",
    width=1,
    length=5,
    labeltop=False,
    labelbottom=False,
    bottom=True,
    top=True)  # ,labelsize='x-large')

ax_U.tick_params(
    which='minor',
    axis="y",
    direction="in",
    left=True,
    right=True)  # ,labelsize='x-large')

ax_U.tick_params(
    which='minor',
    axis="x",
    direction="in",
    bottom=True,
    top=True)  # ,labelsize='x-large')


for ixc in xcs:
    y = main_dict[ixc]['y']
    span = np.linspace(y.min(), y.max(), npoints)
    kernel = stats.gaussian_kde(y)
    distribution = kernel(span)
    # dy = span[1] - span[0]
    # integral = distribution * dy
    # integral = integral.sum()
    ax_J.plot(distribution, span, color=main_dict[ixc]['color'])
ax_J.set_ylim(ylim[0], ylim[1])
ax_J.set_xlim(0, )
# ax_J.axis('off')
# ax_J.set_xticks([])
# ax_J.set_yticks([])
ax_J.set_xlabel('$\mathcal{P}_{\mathrm{J}}$(1/eV)')

ax_J.xaxis.set_minor_locator(MultipleLocator(0.2))
ax_J.xaxis.set_major_locator(MultipleLocator(1.0))
ax_J.yaxis.set_minor_locator(MultipleLocator(0.1))
ax_J.yaxis.set_major_locator(MultipleLocator(1.0))

ax_J.tick_params(
    which='major',
    axis="x",
    direction="inout",
    width=1,
    length=5,
    labeltop=False,
    top=True,
    bottom=True)  # ,labelsize='x-large')

ax_J.tick_params(
    which='major',
    axis="y",
    direction="inout",
    width=1,
    length=5,
    labelleft=False,
    right=True,
    left=True)  # ,labelsize='x-large')

ax_J.tick_params(
    which='minor',
    axis="x",
    direction="in",
    labeltop=False,
    top=True,
    bottom=True)  # ,labelsize='x-large')

ax_J.tick_params(
    which='minor',
    axis="y",
    direction="in",
    right=True,
    left=True)  # ,labelsize='x-large')

for ixc in xcs:
    x = main_dict[ixc]['x']
    y = main_dict[ixc]['y']
    xy = np.vstack([x, y])
    kde = stats.gaussian_kde(xy)
    X, Y = np.mgrid[0:8:500j, 0:4:250j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)
    m = np.amax(Z)
    step = main_dict[ixc]['step']
    levels = np.arange(0.0, m, step) + step
    ax_2d.contour(X, Y, Z, levels, colors='black',
                  linestyles=main_dict[ixc]['linestyle'])
    ax_2d.contourf(
        X,
        Y,
        Z,
        levels,
        cmap=main_dict[ixc]['cmap'],
        alpha=main_dict[ixc]['alpha'])

ax_2d.set_xlim(xlim[0], xlim[1])
ax_2d.set_ylim(ylim[0], ylim[1])
ax_2d.set_xlabel("U(eV)")
ax_2d.set_ylabel("J(eV)")

ax_2d.xaxis.set_minor_locator(MultipleLocator(0.1))
ax_2d.xaxis.set_major_locator(MultipleLocator(1.0))
ax_2d.yaxis.set_minor_locator(MultipleLocator(0.1))
ax_2d.yaxis.set_major_locator(MultipleLocator(1.0))


ax_2d.tick_params(
    which='major',
    axis="x",
    direction="inout",
    width=1.5,
    length=5)  # ,labelsize='x-large')
ax_2d.tick_params(
    which='minor',
    axis="x",
    direction="in")  # ,labelsize='x-large')

ax_2d.tick_params(
    which='major',
    axis="y",
    direction="inout",
    width=1.5,
    length=5)  # ,labelsize='x-large')
ax_2d.tick_params(
    which='minor',
    axis="y",
    direction="in")  # ,labelsize='x-large')

ax_2d.xaxis.set_ticks_position('both')
ax_2d.yaxis.set_ticks_position('both')

ax_legend.axis('off')
ax_legend.set_xticks([])
ax_legend.set_yticks([])

for ixc in xcs:

    if len(ixc) < 6:
        label = ixc + '     \n$\delta = ${:.2f}'.format(main_dict[ixc]['step'])
    else:
        label = ixc + "\n$\delta = ${:.2f}".format(main_dict[ixc]['step'])
    lines.append(ax_legend.plot([-100, -101], [-1000, -10002],
                                linestyle=main_dict[ixc]['linestyle'],
                                label=label,
                                color='black')[0])

    # ax, lines[2:], ['line C', 'line D'],
    #          loc='lower right', frameon=False

    legends.append(Legend(ax_legend,
                          [lines[-1]],
                          [label],
                          frameon=True,
                          loc=main_dict[ixc]['loc']))
    ax_legend.add_artist(legends[-1])
    legend_frames.append(legends[-1].get_frame())
    legend_frames[-1].set_facecolor(main_dict[ixc]['color'])
    legend_frames[-1].set_edgecolor('white')
    legend_texts.append(legends[-1].get_texts())
    for text in legend_texts[-1]:
        text.set_color("white")
ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)
# plt.tight_layout()
plt.savefig('distribution_1.pdf')
# plt.show()
# l3 = plt.legend(handles=[line3],fontsize=14,frameon=1,loc='upper right',
# bbox_transform=(1, 1))
# frame3 = l3.get_frame()
# frame3.set_facecolor('red')
# frame3.set_edgecolor('white')
# text3 = l3.get_texts()
# for text in text3:
#     text.set_color("white")
