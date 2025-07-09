import pandas as pd
import os
from base64 import b64decode

hle_dataset = pd.read_parquet("hf://datasets/cais/hle/data/test-00000-of-00001.parquet")

with open(os.path.join(os.getcwd(), "utility", "hle_dataset.csv"), "w", encoding="utf-8") as f:
    hle_dataset.to_csv(f, index=False)
    f.flush()
    os.fsync(f.fileno())

print("Dataset Downloaded")

image_series = hle_dataset.loc[hle_dataset["image"].astype(bool)]['image']

for idx, row in image_series.items():
    header, encoded = row.split(",", 1)
    extension = "jpg" if "jpeg" in header or "jpg" in header else "png"
    with open(os.path.join(os.getcwd(), "utility", 'images', f"Q{idx}.{extension}"), "wb") as f:
        f.write(b64decode(encoded))

print("Images Saved")

