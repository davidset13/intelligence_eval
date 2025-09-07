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

code_dataset: pd.Series = pd.Series(pd.read_csv(os.path.join(os.getcwd(), "utility", "code_snippets_for_ml.csv"), encoding="utf-8").squeeze("columns"))
final_dataset = pd.DataFrame(columns= ["text", "start_index", "end_index"])

def main() -> None:
    for i in range(len(text_dataset)):
        logger.info(f"Processing {i+1} of {len(text_dataset)}")
        try:
            total_text_len = len(str(text_dataset.loc[i, "text"]))
            random_start_idx = int(random.uniform(0, 1) * min(total_text_len / 2, 2048))
            random_end_idx = int(random.uniform(0, 1) * min(total_text_len / 2, 2048))

            random_code_snippet_idx = random.randint(0, len(code_dataset) - 1)
            random_code_snippet = code_dataset[random_code_snippet_idx].strip()
            
            final_str = (
                f"{str(text_dataset.loc[i, 'text'])[:random_start_idx]}"
                f"{random_code_snippet}"
                f"{str(text_dataset.loc[i, 'text'])[total_text_len//2: total_text_len//2 + random_end_idx]}"
            )

            final_dataset.loc[len(final_dataset)] = [final_str, random_start_idx, random_start_idx + len(random_code_snippet)]

        except Exception as e:
            logger.error(f"Error Processing {i+1}: {e}")
            continue

if __name__ == "__main__":
    main()
    final_dataset.to_csv(os.path.join(os.getcwd(), "utility", "code_detection_dataset.csv"), index=False, encoding="utf-8")