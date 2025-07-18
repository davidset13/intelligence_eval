import pandas as pd
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from payloads import GeneralHLEEval
from async_llm_call import total_eval_process
from logging import getLogger
import uvicorn
import logging
import time
import asyncio
from typing import Any, Coroutine

logging.basicConfig(level=logging.INFO)
logger = getLogger("__main__")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

hle_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "hle_dataset.csv")) # type: ignore

hle_dataset = hle_dataset_full.sample(n=100)

if not load_dotenv(os.path.join(os.getcwd(), ".private.env")):
    raise FileNotFoundError("Private Environment File Not Found")

openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""

hle_app = FastAPI()

@hle_app.post("/llm/general")
async def general_llm_eval(payload: GeneralHLEEval) -> dict[str, float]:
    time_start = time.time()
    
    llm_tasks: list[Coroutine[Any, Any, bool | str | None]] = [total_eval_process(openrouter_api_key, logger, payload.model, "google/gemini-flash-1.5-8b", row) for _, row in hle_dataset.iterrows()]
    results = await asyncio.gather(*llm_tasks)
    results = [result for result in results if (result is not None and result is not False)]

    successful_results: int = 0
    total_results: int = len(results)
    for result in results:
        if result == 'yes':
            successful_results += 1

    try:
        accuracy = successful_results / total_results
    except ZeroDivisionError:
        accuracy = 0
    
    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start} seconds")
    
    return {"accuracy": accuracy}


if __name__ == "__main__":
    try:
        logger.info("HLE Server Starting...")
        uvicorn.run(hle_app, host="127.0.0.1", port=3000, log_level="info")
    except Exception as e:
        logger.error(f"Error in HLE Server: {e}")
        raise HTTPException(status_code=500)