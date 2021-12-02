import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stumpy
import glob
import ipywidgets as widgets
import peakutils
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KernelKMeans
from tslearn.clustering import TimeSeriesKMeans
from scipy.signal import savgol_filter
from functools import partial
from joblib import delayed, Parallel


def plot_motifs(motif_df, path):
    perdrug = pd.DataFrame()
    for drug in np.sort(motif_df.drugs.unique()):
        sub = motif_df[motif_df["drugs"] == drug]
        gb = sub.groupby("labels").count()
        to_add = list(gb["drugs"].values)
        if len(to_add) != 15:
            to_add.append(0)
        perdrug[drug] = to_add

    perdrug_percentage = (perdrug / perdrug.sum(axis = 0))*100

    fig, ax = plt.subplots(figsize = (6, 3), constrained_layout = True)
    perdrug_percentage.T.plot(kind = "bar", stacked = True, cmap = "tab20", ax = ax)
    legend = ax.legend(bbox_to_anchor = (1., 1.), ncol = 2)
    ax.set_title(path)
    ax.set_ylabel("Percentage represented")
    savepath = path.rstrip("hdf5") + "png"
    fig.savefig(savepath, dpi = 300)
    
    fig, ax = plt.subplots(figsize = (6,3), constrained_layout = True)
    perdrug.T.plot(kind = "bar", stacked = True, cmap = "tab20", ax = ax)
    legend = ax.legend(bbox_to_anchor = (1., 1.), ncol = 2)
    ax.set_title(path)
    ax.set_ylabel("Percentage represented")
    fig.savefig("absolute"+savepath, dpi = 300)
    
    fig, ax = plt.subplots(figsize = (6,3), constrained_layout = True)
    perdrug[[c for c in perdrug.columns if c.lower() != "none"]].T.plot(kind = "bar", stacked = True, cmap = "tab20", ax = ax)
    legend = ax.legend(bbox_to_anchor = (1., 1.), ncol = 2)
    ax.set_title(path)
    ax.set_ylabel("Percentage represented")
    fig.savefig("absolute_nonone"+savepath, dpi = 300)

    
paths = glob.glob("motifs*labeled*.hdf5")
for path in paths:
    motif_df = pd.read_hdf(path)
    plot_motifs(motif_df, path)
    