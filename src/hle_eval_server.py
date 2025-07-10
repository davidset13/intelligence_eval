import pandas as pd
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from payloads import GeneralHLEEval, create_general_eval_payload
from async_llm_call import response_generator_openrouter
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

hle_dataset: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "hle_dataset.csv")) # type: ignore

hle_test = hle_dataset.sample(20)

if not load_dotenv(os.path.join(os.getcwd(), ".private.env")):
    raise FileNotFoundError("Private Environment File Not Found")

openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""

hle_app = FastAPI()

@hle_app.post("/llm/general")
async def general_llm_eval(payload: GeneralHLEEval):
    time_start = time.time()
    payload_creation_tasks: list[Coroutine[Any, Any, dict[str, Any]]] = [create_general_eval_payload(payload.model, row, payload.config or {}) for _, row in hle_test.iterrows()]
    payloads = await asyncio.gather(*payload_creation_tasks)
    llm_tasks: list[Coroutine[Any, Any, tuple[str | None, bool]]] = [response_generator_openrouter(openrouter_api_key, payload, logger) for payload in payloads]
    llm_results = await asyncio.gather(*llm_tasks)
    time_end = time.time()
    logger.info(f"Total Time: {time_end - time_start} seconds")
    return llm_results

if __name__ == "__main__":
    try:
        logger.info("HLE Server Starting...")
        uvicorn.run(hle_app, host="127.0.0.1", port=3000, log_level="info")
    except Exception as e:
        logger.error(f"Error in HLE Server: {e}")
        raise HTTPException(status_code=500)
        


