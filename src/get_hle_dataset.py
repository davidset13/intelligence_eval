import pandas as pd
import os

df = pd.read_parquet("hf://datasets/cais/hle/data/test-00000-of-00001.parquet")
df.to_csv(os.path.join(os.getcwd(), "utility", "hle_dataset.csv"), index=False)