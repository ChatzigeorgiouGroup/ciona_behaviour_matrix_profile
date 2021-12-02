import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from functools import partial
from joblib import delayed, Parallel
import tqdm
import pandas as pd
import time

def run_cluster(n_clusters, motifs):
    try:
        motifs = np.stack(motifs["motifs"].values)
        ks = TimeSeriesKMeans(n_clusters=n_clusters, n_jobs=10, metric = "dtw", verbose = 1)
        ks.fit(motifs)
        fig, axes = plt.subplots(7, n_clusters, figsize = (10,5), sharex = True, sharey = True)
        fig.subplots_adjust(hspace = 0, wspace = 0)
        for label, motif in zip(ks.labels_, motifs):
            for i in range(7):
                axes[i,label].plot(motif[:,i], c = "gray", alpha = 0.01, lw = 0.4)


        for j, center in enumerate(ks.cluster_centers_):
            for i in range(7):
                axes[i,j].plot(center[:,i], c = "r", lw = 0.5)
        #         axes[i,j].set_aspect("equal")

        for i in range(n_clusters):
            axes[0,i].set_title(f"cluster {i}", fontsize = 6)
        for i in range(7):
            axes[i,0].set_ylabel(f"dim {i}")
        print("Saving plot")
        fig.savefig(f"./plots_in_context/{n_clusters}clusters_{window}window_meanvarnorm_all_rollingmean_HQ.png", dpi = 300)
        print("Saving Model")
        ks.to_json(f"./plots_in_context/model_{n_clusters}clusters_{window}window_rollingmean.json")
        return(n_clusters)
    except Exception as e:
        with open(f"./plots_in_context/{time.strftime('%H:%M')}_{window}_{n_clusters}_errormessage.txt", "w") as f:
            f.write(str(e))
        return str(e)
    

for window in [30, 150]:
    try:
        print(f"Running clusters for {window} ")
        motifs = pd.read_hdf(f"motifs_window{window}_in_context.hdf5", key = "data")
        f = partial(run_cluster, motifs = motifs)
        to_do = [5, 10, 15, 20, 25, 30, 35, 40] 
        results = Parallel(n_jobs = 4, verbose = 10)(delayed(f)(x) for x in to_do)
        with open(f"result_file_window_{window}.txt", "w") as f:
            f.write(f"RESULTS FOR {window}\n\n {10*'*'}\n\n")
            for result in results:
                f.write(f"{str(result)}\n\n")
        
    except Exception as e:
        print(str(e))