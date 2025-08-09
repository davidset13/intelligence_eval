import pandas as pd
import os
import asyncio


def os_flush(name: str, dataset: pd.DataFrame) -> None:
    
    with open(os.path.join(os.getcwd(), "utility", f"{name}.csv"), "w", encoding="utf-8") as f:
        dataset.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())

    print(f"Dataset {name} flushed to disk")


def download_dataset(scripts: list[str], name: str) -> None:
    
    ns = {}
    code = "\n".join(scripts)
    exec(code, {"pd": pd}, ns)

    df = ns.get(name)
    if isinstance(df, pd.DataFrame):
        os_flush(name, df)
        return
    
    raise ValueError(f"No DataFrame found in {name}")


async def main() -> None:
    args: list[tuple[list[str], str]] = [
        (
            [
                "hle_dataset = pd.read_parquet('hf://datasets/cais/hle/data/test-00000-of-00001.parquet')"
            ],
            "hle_dataset"
        ),
        (
            [
                "splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}",
                "mmlu_pro_dataset = pd.read_parquet('hf://datasets/TIGER-Lab/MMLU-Pro/' + splits['test'])"
            ],
            "mmlu_pro_dataset"
        ),
        (
            [
                "gpqa_dataset = pd.read_csv('hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv')"
            ],
            "gpqa_dataset"
        ),
        (
            [
                "reasoning = pd.read_parquet('hf://datasets/livebench/reasoning/data/test-00000-of-00001.parquet')",
                "data_analysis = pd.read_parquet('hf://datasets/livebench/data_analysis/data/test-00000-of-00001.parquet')",
                "livebench_dataset_gen = pd.concat([reasoning, data_analysis], ignore_index=True)",
            ],
            "livebench_dataset_gen"
        ),
        (
            [
                "livebench_dataset_coding = pd.read_parquet('hf://datasets/livebench/coding/data/test-00000-of-00001.parquet')",
            ],
            "livebench_dataset_coding"
        )
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





