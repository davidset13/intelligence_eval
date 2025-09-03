import os
import pandas as pd
import sys
from dotenv import load_dotenv
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.async_llm_call import response_generator_openrouter
from cmn_pckgs.python.logger import get_logger

logger = get_logger("ml_pipeline")

if not load_dotenv(os.path.join(os.getcwd(), ".private.env")):
    raise FileNotFoundError("Private Environment File Not Found")

openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""

crawl_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
crawl_path = os.path.join(crawl_path, "conc_web_crawler")

logger.info("Reading LCB Codegen")
lcb_codegen = pd.read_csv(os.path.join(os.getcwd(), "utility", "lcb_codegen.csv"))
code_results = pd.Series(dtype = str)
logger.info("LCB Codegen Read")

async def main() -> None:
    for i in range(5):
        logger.info(f"Processing Question {i}")
        question = str(lcb_codegen.loc[i, "question_content"])
        starter_code = str(lcb_codegen.loc[i, "starter_code"])
        public_test_cases = str(lcb_codegen.loc[i, "public_test_cases"])
        payload = {
            "model": "google/gemini-flash-1.5-8b",
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant that can generate code. You are to answer the question based on public test_cases, potiential starter_code, and public_test_cases.
                                    When outputting your code, do just the code, with no explanations, and use traditional ```python notation for doing so"""
                },
                {
                    "role": "user",
                    "content": f"""Question: {question}
                                    Starter Code: {starter_code if starter_code else "No starter code provided"}
                                    Public Test Cases: {public_test_cases}"""
                }
            ]
        }

        resp = await response_generator_openrouter(openrouter_api_key, payload, logger)
        if not resp["success"] or resp["content"] is None:
            logger.warning(f"Error Processing Question (Bad Response) {i}, Skipping")
            continue

        if not isinstance(resp["content"], str):
            logger.warning(f"Error Processing Question (Bad Response) {i}, Skipping")
            continue

        if not resp["content"][0:9] == "```python":
            logger.warning(f"Error Processing Question (String Improperly Formatted) {i}, Skipping")
            continue
        
        if not resp["content"][-3:] == "```":
            logger.warning(f"Error Processing Question (String Improperly Formatted) {i}, Skipping")
            continue

        code_results[len(code_results)] = resp["content"][9:-3]
        logger.info(f"Code Results: {code_results[len(code_results) - 1]}")


if __name__ == "__main__":
    asyncio.run(main())