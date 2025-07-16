import pandas as pd
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from payloads import GeneralHLEEval, create_general_eval_payload, create_general_score_payload
from async_llm_call import response_generator_openrouter
from logging import getLogger
import uvicorn
import logging
import time
import asyncio
from typing import Any, Coroutine
import ast

logging.basicConfig(level=logging.INFO)
logger = getLogger("__main__")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

hle_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "hle_dataset.csv")) # type: ignore

hle_dataset = hle_dataset_full.sample(200)

if not load_dotenv(os.path.join(os.getcwd(), ".private.env")):
    raise FileNotFoundError("Private Environment File Not Found")

openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""

hle_app = FastAPI()

@hle_app.post("/llm/general")
async def general_llm_eval(payload: GeneralHLEEval):
    time_start = time.time()
    hle_dataset_copy = hle_dataset.copy()
    successful_results: list[tuple[int, str | dict[str, Any]]] = []
    
    payload_creation_tasks: list[Coroutine[Any, Any, dict[str, Any]]] = [create_general_eval_payload(payload.model, row, payload.config or {}) for _, row in hle_dataset.iterrows()]
    payloads = await asyncio.gather(*payload_creation_tasks)
    llm_tasks: list[Coroutine[Any, Any, tuple[str | None, bool] | tuple[dict[str, Any] | None, bool]]] = [response_generator_openrouter(openrouter_api_key, payload, logger) for payload in payloads]
    llm_results = await asyncio.gather(*llm_tasks)
    llm_success_results = [(idx, result[0]) for idx, result in enumerate(llm_results) if (result[1] and result[0] is not None)]
    successful_results.extend(llm_success_results)
    hle_dataset_copy = hle_dataset_copy.drop(hle_dataset_copy.index[[idx for idx, _ in llm_success_results]])
    logger.info(f"Successfully Evaluated: {len(successful_results)}")
    logger.info(f"Remaining to Evaluate: {len(hle_dataset_copy)}")

    time_end = time.time()
    logger.info(f"Total Time: {time_end - time_start} seconds")
    time_start = time.time()
    logger.info(f"Starting Eval Processsing...")
    score_payload_creation_tasks = [create_general_score_payload("deepseek/deepseek-chat-v3-0324", row, str(successful_results[i][1]), payload.config or {}) for i, (_, row) in enumerate(hle_dataset.iloc[[idx for idx, _ in successful_results]].iterrows())]
    score_payloads = await asyncio.gather(*score_payload_creation_tasks)
    score_llm_tasks: list[Coroutine[Any, Any, tuple[str | None, bool] | tuple[dict[str, Any] | None, bool]]] = [response_generator_openrouter(openrouter_api_key, payload, logger) for payload in score_payloads]
    score_llm_results = await asyncio.gather(*score_llm_tasks)
    score_llm_success_results = [result[0] for result in score_llm_results if (result[1] and result[0] is not None)]
    score_llm_success_results = [ast.literal_eval(result)['correct'] if isinstance(result, str) else result['correct'] for result in score_llm_success_results]
    total_correct = 0
    total_questions = len(score_llm_success_results)
    for i in score_llm_success_results:
        if i == "yes":
            total_correct += 1
    try:
        accuracy = total_correct / total_questions
    except ZeroDivisionError:
        accuracy = 0
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Total Correct: {total_correct}")
    logger.info(f"Total Questions: {total_questions}")
    time_end = time.time()
    logger.info(f"Total Time: {time_end - time_start} seconds")

    return {"HLE_Accuracy": accuracy}

if __name__ == "__main__":
    try:
        logger.info("HLE Server Starting...")
        uvicorn.run(hle_app, host="127.0.0.1", port=3000, log_level="info")
    except Exception as e:
        logger.error(f"Error in HLE Server: {e}")
        raise HTTPException(status_code=500)