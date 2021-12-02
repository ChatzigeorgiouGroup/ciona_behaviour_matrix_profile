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
import tqdm
from joblib import delayed, Parallel

path = "/share/data/temp/athira/July17_features_combined_noLightStimuli.pickle"
df = pd.read_pickle(path)

def get_curvature_data(f):
    """
    function to extract a specific timeseries from the main dataframe.
    """
    df_f = df[df["filename"] == f]
    cols_to_return = [c for c in df.columns if "curv" in c]
    return df_f[cols_to_return]

def do_work(filename, window = 150):
    try:
        data = get_curvature_data(filename)
        for c in data.columns:
            data[c] = data[c].rolling(10).mean()
        data = data[10:]
        mp, ind = stumpy.mstump(data, m = window)
        peaks = peakutils.indexes(1-mp[:,-1], min_dist = window, thres = 0.7)
        motifs = np.stack([data.values[peak:peak+window, :] for peak in peaks])
        return [filename, motifs, peaks]
    except Exception as e:
        return( (filename, str(e)))
    
to_do = df["filename"].unique()
windows = [150]
n_clusters = 15

for window in windows:
    motifs = Parallel(n_jobs=15, verbose = 5)(delayed(partial(do_work, window = window))(f) for f in to_do)
    
    motifs = [m for m in motifs if type(m) == list]
    motif_df = pd.concat([pd.DataFrame([[m[0], x, m[-1]] for x in m[1]], columns = ["filename", "motifs", "frame"]) for m in motifs], ignore_index = True)
    
    motifs_scaled = TimeSeriesScalerMeanVariance().fit_transform(np.stack(motif_df["motifs"].values))
    
    for metric in ["dtw"]:
        model = TimeSeriesKMeans(n_jobs = 30, 
                                 n_clusters = n_clusters, 
                                 metric = metric, 
                                 n_init = 5, max_iter = 100, 
                                 max_iter_barycenter = 150, 
                                 verbose = 1)
        model.fit(motifs_scaled)
        model.to_json(f"model_with_context_n{n_clusters}_w{window}.json")
    
        motif_df[f"labels_{metric}"] = model.labels_
    
    drugs_col = []
    for f in tqdm.tqdm(motif_df["filename"]):
        drugs_col.append(df["drug"][df["filename"] == f].values[0])
        
    motif_df["drugs"] = drugs_col
    motif_df.to_hdf(f"motifs_{window}_nclusters_{n_clusters}_meanvarnorm_rollingmean.hdf5", key = "data")