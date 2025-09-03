import os
import sys
import pandas as pd
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmn_pckgs.python.logger import get_logger

crawl_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
crawl_path = os.path.join(crawl_path, "conc_web_crawler")

logger = get_logger("ml_pipeline")


logger.info("Reading data.jsonl.gz")
text_dataset = pd.read_json(os.path.join(crawl_path, "data.jsonl.gz"), lines=True, compression="gzip")
logger.info("data.jsonl.gz read")

code_dataset: pd.Series = pd.read_csv(os.path.join(os.getcwd(), "utility", "code_snippets_for_ml.csv"), encoding="utf-8").squeeze("columns") #type: ignore
final_dataset = pd.DataFrame(columns= ["text", "start_index", "end_index"])

def main() -> None:
    for i in range(len(text_dataset)):
        logger.info(f"Processing {i+1} of {len(text_dataset)}")
        try:
            random_index = random.randint(0, len(text_dataset.loc[i, "text"]) - 1) #type: ignore
            random_code_snippet_idx = random.randint(0, len(code_dataset) - 1)
            random_code_snippet = code_dataset[random_code_snippet_idx].strip()
            final_dataset.loc[len(final_dataset)] = [f"{text_dataset.loc[i, 'text'][:random_index]}{random_code_snippet}{text_dataset.loc[i, 'text'][random_index:]}", random_index, random_index + len(random_code_snippet)] #type: ignore
        except Exception as e:
            logger.error(f"Error Processing {i+1}: {e}")
            continue

if __name__ == "__main__":
    main()
    final_dataset.to_csv(os.path.join(os.getcwd(), "utility", "code_detection_dataset.csv"), index=False, encoding="utf-8")