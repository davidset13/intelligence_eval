import pandas as pd
import os

hle_dataset = pd.read_parquet("hf://datasets/cais/hle/data/test-00000-of-00001.parquet")

with open(os.path.join(os.getcwd(), "utility", "hle_dataset.csv"), "w", encoding="utf-8") as f:
    hle_dataset.to_csv(f, index=False)
    f.flush()
    os.fsync(f.fileno())

print("HLE Dataset Downloaded")

splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
mmlu_pro_dataset = pd.read_parquet("hf://datasets/TIGER-Lab/MMLU-Pro/" + splits["test"])

with open(os.path.join(os.getcwd(), "utility", "mmlu_pro_dataset.csv"), "w", encoding="utf-8") as f:
    mmlu_pro_dataset.to_csv(f, index=False)
    f.flush()
    os.fsync(f.fileno())

print("MMLU-Pro Dataset Downloaded")

gpqa_dataset = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv")

with open(os.path.join(os.getcwd(), "utility", "gpqa_dataset.csv"), "w", encoding="utf-8") as f:
    gpqa_dataset.to_csv(f, index=False)
    f.flush()
    os.fsync(f.fileno())

print("GPQA Dataset Downloaded")

