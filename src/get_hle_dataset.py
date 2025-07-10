import pandas as pd
import os
from base64 import b64decode

hle_dataset = pd.read_parquet("hf://datasets/cais/hle/data/test-00000-of-00001.parquet")
image_series = hle_dataset.loc[hle_dataset["image"].astype(bool)]['image']

print("Dataset Downloaded")

for idx in image_series.index:
    header, encoded = image_series[idx].split(",", 1)
    extension = "jpg" if "jpeg" in header or "jpg" in header else "png"
    with open(os.path.join(os.getcwd(), "utility", 'images', f"Q{idx}.{extension}"), "wb") as f:
        f.write(b64decode(encoded))

with open(os.path.join(os.getcwd(), "utility", "hle_dataset.csv"), "w", encoding="utf-8") as f:
    hle_dataset.to_csv(f, index=False)
    f.flush()
    os.fsync(f.fileno())

print("Images Saved")

