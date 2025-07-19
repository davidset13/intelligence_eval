import pandas as pd
import os

hle_dataset = pd.read_parquet("hf://datasets/cais/hle/data/test-00000-of-00001.parquet")
image_series = hle_dataset.loc[hle_dataset["image"].astype(bool)]['image']

with open(os.path.join(os.getcwd(), "utility", "hle_dataset.csv"), "w", encoding="utf-8") as f:
    hle_dataset.to_csv(f, index=False)
    f.flush()
    os.fsync(f.fileno())

print("Dataset Downloaded")
