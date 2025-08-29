# -*- coding: utf-8 -*-


import os, h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils, pgpy_utils, tools
from typing import List
from sklearn.cluster import HDBSCAN


"""
========================================================
Plotting utilities
========================================================
"""

plt.ion()

def plot_eigenspectra_multi_res(res_meta: List, *eig_res: np.ndarray, ax: plt.Axes = None):
    
    res_style_list = [
        {"s": 30, "marker": 'o', "facecolor": "none"},
        {"s": 30, "marker": 'x'}
    ]
    
    classify_list = [
        [lambda x: (x.imag < 0), {"color": 'm'}],
        [lambda x: (x.imag >= 0), {"color": 'c'}],
        # [lambda x: np.abs(x.imag) >= 1e-9, {'color': 'tab:blue'}]
    ]
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8,8))
    
    for i_res, spectrum in enumerate(eig_res):
        for classifier in classify_list:
            idx_class = classifier[0](spectrum)
            ax.scatter(np.abs(np.imag(spectrum[idx_class])), np.abs(np.real(spectrum[idx_class])), 
                **classifier[1], **res_style_list[i_res], label=f"N={res_meta[i_res]['N']}, L={res_meta[i_res]['L']}", zorder=4)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot([1e-15, 1e+8], [1e-15, 1e+8], '--', color="gray")
    ax.set_xlabel(r"$|\omega| = |\mathrm{Im}[\lambda]|$", fontsize=14)
    ax.set_ylabel(r"$\sigma = -\mathrm{Re}[\lambda]$", fontsize=14)
    ax.set_title("Eigenvalue spectrum for different resolutions", fontsize=14)
    ax.grid(True, which="both")
    ax.legend(fontsize=12)
    return ax


def plot_eigenspectra_select(eig_vals, idx_select, ax: plt.Axes = None):
    
    classify_list = [
        [lambda x: (x.imag < 0) & (~idx_select), {"s": 30, "marker": 'o', "label": "Prograde (filtered out)", "color": 'thistle', 'zorder': 1}],
        [lambda x: (x.imag >= 0) & (~idx_select), {"s": 30, "marker": '^', "label": "Retrograde (filtered out)", "color": 'turquoise', 'zorder': 1}],
        [lambda x: (x.imag < 0) & idx_select, {"s": 30, "marker": 'o', "label": "Prograde", "color": 'm', 'zorder': 5}],
        [lambda x: (x.imag >= 0) & idx_select, {"s": 30, "marker": '^', "label": "Retrograde", "color": 'teal', 'zorder': 5}],
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(8,8))
    
    for classifier in classify_list:
        idx_class = classifier[0](eig_vals)
        ax.scatter(np.abs(np.imag(eig_base[idx_class])), np.abs(np.real(eig_base[idx_class])), **classifier[1])
    
    ax.plot([1e-6, 1e+8], [1e-6, 1e+8], '--', color="gray")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$|\omega| = |\mathrm{Im}[\lambda]|$", fontsize=14)
    ax.set_ylabel(r"$\sigma = -\mathrm{Re}[\lambda]$", fontsize=14)
    ax.set_title("Filtered eigenvalue spectrum (No. = {})".format(np.sum(idx_select)), fontsize=14)
    ax.grid(True, which="both")
    ax.legend(fontsize=12)
    return ax


def plot_eigenspectra_conv(eig_vals, idx_select, color_arr, ax: plt.Axes = None):
    
    vrange = (np.min(color_arr[idx_select]), np.max(color_arr[idx_select]))
    classify_list = [
        [lambda x: ((x.imag < 0) & (~idx_select), 'lightgray'), {"s": 30, "marker": 'o', "label": "Prograde (filtered out)", "vmin": vrange[0], "vmax": vrange[1], 'zorder': 3}],
        [lambda x: ((x.imag >= 0) & (~idx_select), 'lightgray'), {"s": 30, "marker": '^', "label": "Retrograde (filtered out)", "vmin": vrange[0], "vmax": vrange[1], 'zorder': 3}],
        [lambda x: ((x.imag < 0) & (idx_select), None), {"s": 30, "marker": 'o', "label": "Prograde", "vmin": vrange[0], "vmax": vrange[1], 'zorder': 5}],
        [lambda x: ((x.imag >= 0) & (idx_select), None), {"s": 30, "marker": '^', "label": "Retrograde", "vmin": vrange[0], "vmax": vrange[1], 'zorder': 5}],
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(8,8))
    
    for classifier in classify_list:
        idx_class, c_literal = classifier[0](eig_vals)
        if c_literal is None:
            c_literal = color_arr[idx_class]
        im = ax.scatter(np.abs(np.imag(eig_base[idx_class])), np.abs(np.real(eig_base[idx_class])), c=c_literal, **classifier[1], cmap='inferno_r')
    
    plt.colorbar(im, ax=ax)
    ax.plot([1e-6, 1e+8], [1e-6, 1e+8], '--', color="gray")
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r"$|\omega| = |\mathrm{Im}[\lambda]|$", fontsize=14)
    ax.set_ylabel(r"$\sigma = -\mathrm{Re}[\lambda]$", fontsize=14)
    ax.grid(which="both")
    ax.legend(fontsize=12)
    return ax


def plot_cluster(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        ax.scatter(X[class_index, 0], X[class_index, 1], 50, marker=("x" if k == -1 else "o"),
            color=tuple(col), edgecolor="k", label='class {}'.format(k))
        # for ci in class_index:
        #     ax.plot(
        #         X[ci, 0],
        #         X[ci, 1],
        #         "x" if k == -1 else "o",
        #         markerfacecolor=tuple(col),
        #         markeredgecolor="k",
        #         markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
        #         label='class {}'.format(k)
        #     )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    ax.legend()
    return ax


"""
========================================================
Data configuration
========================================================
"""

# File name
model_dir = './runs/Axial_Le1e-4_Eeta1e-6_m1/'
fname = os.path.join(model_dir, 'eigenspectra.h5')
suffix = '_asym'

# Parity of magnetic eigenmode
parity_mode = 'QP'

timer = tools.ProcTimer(start=True)

meta = list()
spectra = list()
with h5py.File(fname, 'r') as fread:
    n_res = len([key for key in fread.keys() if key[:9] == "eigenspec"])
    for i_res in range(n_res):
        gp = fread[f"eigenspec_res{i_res}"]
        meta.append({key: gp.attrs[key] for key in gp.attrs.keys()})
        if parity_mode is None:
            spectra.append(gp["spectrum"][()])
        elif parity_mode == "both":
            spectra.append(np.concatenate((gp["spectrum_QP"][()], gp["spectrum_DP"][()])))
        else:
            spectra.append(gp["spectrum_%s" % parity_mode][()])

for i_w, w in enumerate(spectra):
    sort_idx = np.argsort(np.imag(w))
    spectra[i_w] = w[sort_idx][np.abs(w[sort_idx]) > 1e-10]

timer.flag(loginfo="Eigenvalues loaded resolutions:\n{}".format(str(meta)), print_str=True, mode='0+')


"""
========================================================
Eigenvalue filtering
========================================================
"""

# By default, use the last two resolutions for filtering
eig_base, eig_comp = spectra[-1], spectra[-2]

drift_ratio, nearest_idx = pgpy_utils.eigen_drift(eig_base, eig_comp, mode='global')
drift_abs = np.abs(eig_base - eig_comp[nearest_idx])
drift_rel = drift_abs/np.abs(eig_base)
reciproc = 1./drift_ratio
sig_digits = np.abs(eig_base)/(drift_abs + 1e-16*np.abs(eig_base))

timer.flag(loginfo="Eigenvalue drifts calculated", print_str=True, mode='0+')

threshold = 3e+3
c_mode = 'reciprocal_drift'
if c_mode == 'reciprocal_drift':
    c_arr = np.log10(reciproc)
else:
    c_arr = np.log10(sig_digits)

confirm_flag = False
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8), layout='constrained')
fig = plt.figure(figsize=(24, 8), layout='constrained')

while not confirm_flag:
    
    idx_select = (reciproc > threshold) & (np.abs(np.imag(eig_base)/np.real(eig_base)) > 1)

    fig.clear()
    axes = fig.subplots(nrows=1, ncols=3)
    
    plot_eigenspectra_multi_res(meta[-2:], eig_comp, eig_base, ax=axes[0])
    plot_eigenspectra_select(eig_base, idx_select, ax=axes[1])
    plot_eigenspectra_conv(eig_base, idx_select, c_arr, ax=axes[2])
    for ax in axes:
        ax.set_xlim([1e-4, 3.])
        ax.set_ylim([1e-10, 3.])
    if c_mode == 'reciprocal_drift':
        axes[2].set_title("Reciprocal drift of filtered eigenspectrum (Boyd)", fontsize=14)
    else:
        axes[2].set_title("No. significant digits of filtered eigenspectrum", fontsize=14)
    
    new_thresh = float(input("Update threshold?: "))
    if new_thresh < 0:
        confirm_flag = True
    else:
        threshold = new_thresh

save_name = os.path.join(model_dir, 'eigenspectrum_filtered')
plt.savefig(save_name + suffix + '.png', dpi=150, bbox_inches='tight')
plt.savefig(save_name + suffix + '.pdf', bbox_inches='tight')
plt.close(fig=fig)

eig_filtered = eig_base[idx_select]
eig_comp_filtered = eig_comp[nearest_idx[idx_select]]


"""
========================================================
Eigenvalue clustering
========================================================
"""

eig_clustered = dict()
eig_comp_clustered = dict()

X_eigen = np.stack([
    np.log10(np.abs(np.imag(eig_base[idx_select & (np.imag(eig_base) < 0)]))),
    np.log10(np.abs(np.real(eig_base[idx_select & (np.imag(eig_base) < 0)])))
], axis=-1)
k_slope = 0.
y_scale = 1.

confirm_flag = False
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), layout='constrained')

while not confirm_flag:
    X_feature = np.stack([k_slope*X_eigen[:,0] + X_eigen[:,1], (X_eigen[:,0] - k_slope*X_eigen[:,1])/y_scale], axis=1)
    clusterer = HDBSCAN(algorithm='auto').fit(X_feature)
    for ax in axes:
        ax.clear()
    
    plot_cluster(X_feature, clusterer.labels_, ax=axes[0])
    plot_cluster(X_eigen, clusterer.labels_, ax=axes[1])

    new_slope = input("Update slope?: ")
    new_scale = input("Update scale?: ")
    if new_slope == 'n' and new_scale == 'n':
        confirm_flag = True
        save_name = os.path.join(model_dir, 'eigenspectrum_clusters_E')
        plt.savefig(save_name + suffix + '.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_name + suffix + '.pdf', bbox_inches='tight')
        plt.close(fig=fig)
    else:
        if new_slope != 'n':
            k_slope = float(new_slope)
        if new_scale != 'n':
            y_scale = float(new_scale)

for k in set(clusterer.labels_):
    class_index = np.where(clusterer.labels_ == k)[0]
    eig_tmp = eig_filtered[np.imag(eig_filtered) < 0][class_index]
    eig_clustered[f'e{k}'] = eig_tmp[np.flip(np.argsort(np.abs(np.imag(eig_tmp))))]
    eig_tmp = eig_comp_filtered[np.imag(eig_filtered) < 0][class_index]
    eig_comp_clustered[f'e{k}'] = eig_tmp[np.flip(np.argsort(np.abs(np.imag(eig_tmp))))]

X_eigen = np.stack([
    np.log10(np.abs(np.imag(eig_base[idx_select & (np.imag(eig_base) >= 0)]))),
    np.log10(np.abs(np.real(eig_base[idx_select & (np.imag(eig_base) >= 0)])))
], axis=-1)
k_slope = 0.
y_scale = 1.

confirm_flag = False
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), layout='constrained')

while not confirm_flag:
    X_feature = np.stack([k_slope*X_eigen[:,0] + X_eigen[:,1], (X_eigen[:,0] - k_slope*X_eigen[:,1])/y_scale], axis=1)
    clusterer = HDBSCAN(algorithm='auto').fit(X_feature)
    for ax in axes:
        ax.clear()
    
    plot_cluster(X_feature, clusterer.labels_, ax=axes[0])
    plot_cluster(X_eigen, clusterer.labels_, ax=axes[1])

    new_slope = input("Update slope?: ")
    new_scale = input("Update scale?: ")
    if new_slope == 'n' and new_scale == 'n':
        confirm_flag = True
        save_name = os.path.join(model_dir, 'eigenspectrum_clusters_W')
        plt.savefig(save_name + suffix + '.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_name + suffix + '.pdf', bbox_inches='tight')
        plt.close(fig=fig)
    else:
        if new_slope != 'n':
            k_slope = float(new_slope)
        if new_scale != 'n':
            y_scale = float(new_scale)
            
for k in set(clusterer.labels_):
    class_index = np.where(clusterer.labels_ == k)[0]
    eig_tmp = eig_filtered[np.imag(eig_filtered) >= 0][class_index]
    eig_clustered[f'w{k}'] = eig_tmp[np.flip(np.argsort(np.abs(np.imag(eig_tmp))))]
    eig_tmp = eig_comp_filtered[np.imag(eig_filtered) >= 0][class_index]
    eig_comp_clustered[f'w{k}'] = eig_tmp[np.flip(np.argsort(np.abs(np.imag(eig_tmp))))]


fig, ax = plt.subplots(figsize=(8, 8))

for key, eigs in eig_clustered.items():
    ax.scatter(np.abs(np.imag(eigs)), np.abs(np.real(eigs)), 30, marker=('o' if key[0] == 'e' else '^'), label=key, zorder=5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"$|\omega| = |\mathrm{Im}[\lambda]|$", fontsize=14)
ax.set_ylabel(r"$\sigma = -\mathrm{Re}[\lambda]$", fontsize=14)
ax.grid(which="both")
ax.set_xlim([1e-4, 3.])
ax.set_ylim([1e-10, 3.])
ax.legend(fontsize=12)
save_name = os.path.join(model_dir, 'eigenspectrum_clustered')
plt.savefig(save_name + suffix + '.png', dpi=150, bbox_inches='tight')
plt.savefig(save_name + suffix + '.pdf', bbox_inches='tight')


"""
========================================================
Eigenvalue output
========================================================
"""

# Sorting
label_list = list(eig_clustered.keys())
min_freq = [np.abs(np.imag(eig_clustered[label][-1])) for label in label_list]
idx_classes = np.argsort(min_freq)

data_df = {'Class': list(), 'res2': list(), 'res1': list()}
for idx in idx_classes:
    class_label = label_list[idx]
    data_df['Class'] += [class_label]*len(eig_clustered[class_label])
    data_df['res2'].append(eig_clustered[class_label])
    data_df['res1'].append(eig_comp_clustered[class_label])
data_df['res2'] = np.concatenate(data_df['res2'])
data_df['res1'] = np.concatenate(data_df['res1'])

df = pd.DataFrame(data=data_df)
with open(os.path.join(model_dir, 'eig-filter' + suffix + '.txt'), 'x') as fwrite:
    str_df = df.to_string(col_space=[8] + [40]*2, float_format=lambda x: "{0.real:.15e}{0.imag:+.15e}j".format(x))
    str_df = '\n'.join([
        "========================================================", 
        "List of converged(?) eigenvalues with Q >= 1", 
        "========================================================", 
        str_df
    ])
    fwrite.write(str_df)

plt.show()
