import pandas as pd
import os
import asyncio
from datasets import load_dataset


def os_flush(name: str, dataset: pd.DataFrame) -> None:
    
    with open(os.path.join(os.getcwd(), "utility", f"{name}.csv"), "w", encoding="utf-8") as f:
        dataset.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())

    print(f"Dataset {name} flushed to disk")


def download_dataset(scripts: list[str], name: str) -> None:
    
    ns = {}
    code = "\n".join(scripts)
    exec(code, {"pd": pd, "load_dataset": load_dataset}, ns)

    df = ns.get(name)
    if isinstance(df, pd.DataFrame):
        os_flush(name, df)
        return
    
    raise ValueError(f"No DataFrame found in {name}")


async def main() -> None:
    if not os.path.exists(os.path.join(os.getcwd(), "utility")):
        os.makedirs(os.path.join(os.getcwd(), "utility"))
    
    args: list[tuple[list[str], str]] = [
        (
            [
                "hle_dataset = pd.read_parquet('hf://datasets/cais/hle/data/test-00000-of-00001.parquet')",
                "hle_dataset['category'] = hle_dataset['category'].map(lambda x: x[:3].upper())",
                "hle_dataset = hle_dataset[['question', 'image', 'answer_type', 'answer', 'category']]"
            ],
            "hle_dataset"
        ),
        (
            [
                "splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}",
                "mmlu_pro_dataset = pd.read_parquet('hf://datasets/TIGER-Lab/MMLU-Pro/' + splits['test'])",
                "mmlu_pro_dataset['category'] = mmlu_pro_dataset['category'].map(lambda x: x[:3].upper())",
                "mmlu_pro_dataset = mmlu_pro_dataset[['question', 'options', 'answer', 'category']]"
            ],
            "mmlu_pro_dataset"
        ),
        (
            [
                "gpqa_dataset = pd.read_csv('hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv')",
                "gpqa_dataset = gpqa_dataset[['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']]"
            ],
            "gpqa_dataset"
        ),
        (
            [
                "reasoning = pd.read_parquet('hf://datasets/livebench/reasoning/data/test-00000-of-00001.parquet')",
                "data_analysis = pd.read_parquet('hf://datasets/livebench/data_analysis/data/test-00000-of-00001.parquet')",
                "instruction_following = pd.read_parquet('hf://datasets/livebench/instruction_following/data/test-00000-of-00001.parquet')",
                "math = pd.read_parquet('hf://datasets/livebench/math/data/test-00000-of-00001.parquet')",
                "language = pd.read_parquet('hf://datasets/livebench/language/data/test-00000-of-00001.parquet')",
                "livebench_dataset = pd.concat([reasoning, data_analysis, instruction_following, math, language], ignore_index=True)",
                "livebench_dataset = livebench_dataset.loc[livebench_dataset['livebench_removal_date'].str.len() == 0]",
                "livebench_dataset['category'] = livebench_dataset['category'].map(lambda x: x[:3].upper())",
                "livebench_dataset = livebench_dataset[['turns', 'ground_truth', 'category', 'task_prompt']]"
            ],
            "livebench_dataset"
        ),
    ]


    seen_datasets: list[str] = []
    for _, name in args:
        if name in seen_datasets:
            raise ValueError(f"Dataset {name} already exists, will create race condition")
        seen_datasets.append(name)

    await asyncio.gather(*(asyncio.to_thread(download_dataset, scripts, name) for scripts, name in args))
    print("\nDatasets Downloaded!")


if __name__ == "__main__":
    asyncio.run(main())





