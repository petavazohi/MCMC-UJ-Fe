# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:38:44 2020

@author: ptava
"""

import numpy as np
import matplotlib.pylab as plt
from scipy import stats


def plot_dist2d(x, y,title):
    xy = np.vstack([x, y])
    kde = stats.gaussian_kde(xy)
    z = kde(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, c=z, s=4, edgecolor='',cmap='jet')
    ax.set_xlim(0,5)
    ax.set_ylim(0,9)
    ax.set_title(title)

def plot_contour(x, y,linestyles,cmap,step,label,alpha):
    xy = np.vstack([x, y])
    kde = stats.gaussian_kde(xy)
    X, Y = np.mgrid[0:8:100j, 0:4:50j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)
    
    m = np.amax(Z)
    levels = np.arange(0.0, m, step) + step
    print(label, levels)
#    plt.contourf(xx, yy, f, levels, cmap=cmap, alpha=0.5)
    plt.contour(X, Y, Z, levels, colors='black',linestyles=linestyles)
    plt.contourf(X, Y, Z, levels, cmap=cmap, alpha=alpha)
#    plt.imshow(np.rot90(Z), cmap=cmap, extent=[0, 8, 0, 4])
#    z = kde(xy)
#    idx = z.argsort()
#    x, y, z = x[idx], y[idx], z[idx]
#    plt.scatter(x, y, c=cmap.lower()[:-1], s=1, edgecolor='',cmap=cmap)
#    plt.xlim(0,8)
#    plt.ylim(0,4)

def plot_evolution(x, y, title):
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))
    axes[0].plot(x)
    axes[0].set_xlim(0,)
    axes[1].plot(y)
    axes[1].set_xlim(0,)
    axes[0].set_title(title)


def get_mode(x, npoints):
    span = np.linspace(x.min(), x.max(), npoints)
    kernel = stats.gaussian_kde(x)
    distribution = kernel(span)
    return np.round(span[distribution.argmax()], 1)


def plot_dist1d(x, npoints, title):
    span = np.linspace(x.min(), x.max(), npoints)
    kernel = stats.gaussian_kde(x)
    distribution = kernel(span)
    mode = span[distribution.argmax()]
    plt.figure()
    plt.plot(span, distribution)
    plt.title(title)
    plt.axvline(mode, color='r')

def std2d(xi,yi):
    x_ = np.mean(xi,axis=0)
    y_ = np.mean(yi,axis=0)
    n = len(xi)
    rms = 1/n*((xi-x_)**2+(yi-y_)**2).sum()
    return rms
    
lda = np.loadtxt('LDA.txt')#[1500:]
pbe = np.loadtxt('PBE.txt')#[2000:]
pbesol = np.loadtxt('PBEsol.txt')#[2000:]


lda = lda[lda[:, 0] < lda[:, 1]]
pbe = pbe[pbe[:, 0] < pbe[:, 1]]
pbesol = pbesol[pbesol[:, 0] < pbesol[:, 1]]


#pbe_dudarev = pbe[:, 1] - pbe[:, 0]
#print("mode lda u = {}".format(get_mode(lda[:, 1], 5000)))
#print("mode lda j = {}".format(get_mode(lda[:, 0], 5000)))
#print("-----------------------------------------------------")
#print("mode pbe u = {}".format(get_mode(pbe[:, 1], 5000)))
#print("mode pbe j = {}".format(get_mode(pbe[:, 0], 5000)))
#print("mode pbe u-j = {}".format(get_mode(pbe_dudarev, 5000)))
#print("-----------------------------------------------------")
#print("mode pbesol u = {}".format(get_mode(pbesol[:, 1], 5000)))
#print("mode pbesol j = {}".format(get_mode(pbesol[:, 0], 5000)))
#print("-----------------------------------------------------")
#
#print("mean, std lda u = {}, {}".format(np.mean(lda[:, 1]).round(1), np.std(lda[:, 1]).round(1)))
#print("mean, std lda j = {}, {}".format(np.mean(lda[:, 0]).round(1), np.std(lda[:, 0]).round(1)))
#print("RMS 2d lda = {}".format(std2d(lda[:, 0], lda[:, 1]).round(1)))
#print("-----------------------------------------------------")
#print("mean, std pbe u = {}, {}".format(np.mean(pbe[:, 1]).round(1), np.std(pbe[:, 1]).round(1)))
#print("mean, std pbe j = {}, {}".format(np.mean(pbe[:, 0]).round(1), np.std(pbe[:, 0]).round(1)))
#print("RMS 2d pbe = {}".format(std2d(pbe[:, 0], pbe[:, 1]).round(1)))
#print("mean, std pbe u-j = {}, {}".format(np.mean(pbe_dudarev).round(1), np.std(pbe_dudarev).round(1)))
#print("-----------------------------------------------------")
#print("mean, std pbesol u = {}, {}".format(np.mean(pbesol[:, 1]).round(1), np.std(pbesol[:, 1]).round(1)))
#print("mean, std pbesol j = {}, {}".format(np.mean(pbesol[:, 0]).round(1), np.std(pbesol[:, 0]).round(1)))
#print("RMS 2d pbesol = {}".format(std2d(pbesol[:, 0], pbesol[:, 1]).round(1)))
#print("-----------------------------------------------------")
#print("mean, std lda u = {}, {}".format(np.mean(lda[:, 1]).round(1), np.std(lda[:, 1]).round(1)))
#print("mean, std lda j = {}, {}".format(np.mean(lda[:, 0]).round(1), np.std(lda[:, 0]).round(1)))
#print("-----------------------------------------------------")
#print("mean, std pbe u = {}, {}".format(np.mean(pbe[:, 1]).round(1), np.std(pbe[:, 1]).round(1)))
#print("mean, std pbe j = {}, {}".format(np.mean(pbe[:, 0]).round(1), np.std(pbe[:, 0]).round(1)))
#print("mean, std pbe u-j = {}, {}".format(np.mean(pbe_dudarev), np.std(pbe_dudarev)))
#print("-----------------------------------------------------")
#print("mean, std pbesol u = {}, {}".format(np.mean(pbesol[:, 1]).round(1), np.std(pbesol[:, 1]).round(1)))
#print("mean, std pbesol j = {}, {}".format(np.mean(pbesol[:, 0]).round(1), np.std(pbesol[:, 0]).round(1)))



#plot_dist1d(lda[:,1], 5000, 'lda_u')
#plot_dist1d(lda[:,0], 5000, 'lda_j')
#
#plot_dist1d(pbe[:,1], 5000, 'pbe_u')
#plot_dist1d(pbe[:,0], 5000, 'pbe_j')
#plot_dist1d(pbe_dudarev, 5000, 'pbe_dudarev')
#
#plot_dist1d(pbesol[:,1], 5000, 'pbesol_u')
#plot_dist1d(pbesol[:,0], 5000, 'pbesol_j')

# plt.figure()
#plt.plot(span_lda_u, distribution_lda_u)
# plt.axvline(span_lda_u[distribution_lda_u.argmax()],color='r')
# plt.figure()
#plt.plot(span_lda_j, distribution_lda_j)
# plt.axvline(span_lda_j[distribution_lda_j.argmax()],color='r')

#fig = plt.figure(constrained_layout=True,figsize=(8,14))
#heights = [2, 10]
#widths = [5, 2]
#
#gs = fig.add_gridspec(2, 2, width_ratios=widths,
#                          height_ratios=heights)

plt.figure(figsize=(10,5))
plot_contour(lda[:,1], lda[:,0],'-.','Blues',0.09,'LDA',1)
line1, = plt.plot([-100,-101],[-1000,-10002],linestyle='-.',label='LDA',color='black')

plot_contour(pbesol[:,1], pbesol[:,0],'solid' ,'Reds',0.1,'PBEsol', 1)
line3, = plt.plot([-100,-101],[-1000,-10002],linestyle='solid' ,label='PBEsol',color='black')

plot_contour(pbe[:,1], pbe[:,0],'--','Greens',1,'PBE',1)
line2, = plt.plot([-100,-101],[-1000,-10002],linestyle='--',label='PBE',color='black')

#plt.plot(np.linspace(0,10,50),np.linspace(0,10,50))

plt.xlim(2,8)
plt.ylim(1,4)   
plt.xlabel("U(eV)")
plt.ylabel("J(eV)")

l1 = plt.legend(handles=[line1],fontsize=14,frameon=1, loc=(0.01, 0.48))
frame1 = l1.get_frame()
frame1.set_facecolor('blue')
frame1.set_edgecolor('white')
text1 = l1.get_texts()
for text in text1:
    text.set_color("white")




l3 = plt.legend(handles=[line3],fontsize=14,frameon=1,loc='upper right', bbox_transform=(1, 1))
frame3 = l3.get_frame()
frame3.set_facecolor('red')
frame3.set_edgecolor('white')
text3 = l3.get_texts()
for text in text3:
    text.set_color("white")


l2 = plt.legend(handles=[line2],fontsize=16,frameon=1)
frame2 = l2.get_frame()
frame2.set_facecolor('green')
frame2.set_edgecolor('white')
text2 = l2.get_texts()
for text in text2:
    text.set_color("white")
#plot_dist2d(lda[:,0], lda[:,1],'lda')
#plot_dist2d(pbe[:,0], pbe[:,1],'pbe')
#plot_dist2d(pbesol[:,0], pbesol[:,1],'pbesol')
#
#plot_evolution(lda[:,0], lda[:,1],'LDA')
#plot_evolution(pbe[:,0], pbe[:,1],'PBE')
#plot_evolution(pbesol[:,0], pbesol[:,1],'PBEsol')

#corr_lda = np.corrcoef(lda[:, 0], lda[:, 1])
#corr_pbe = np.corrcoef(pbe[:, 0], pbe[:, 1])
#corr_pbesol = np.corrcoef(pbesol[:, 0], pbesol[:, 1])
#
#print("correlations for lda:{}, pbe:{}, pbesol:{}".format(corr_lda[0, 1].round(1),
#                                                          corr_pbe[0, 1].round(1),
#                                                          corr_pbesol[0, 1].round(1)))
