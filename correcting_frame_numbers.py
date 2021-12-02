import pandas as pd
import numpy as np
import glob

def correct_frames(f):
    test = motif_df[motif_df["filename"] == f]
    test["frame"] = test["frame"].iloc[0]
    return test

paths = glob.glob("*.hdf5")

for path in paths:
    motif_df = pd.read_hdf(path, key = "data")
    motif_df_frames = pd.concat([correct_frames(f) for f in motif_df["filename"].unique()])
    motif_df_frames.to_hdf(f"correct_frame_{path}", key = "data")
    