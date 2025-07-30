import pandas as pd
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from hle.total_eval_process_hle import hle_scoring
from logging import getLogger
import uvicorn
import logging
from pydantic import BaseModel
from typing import Any, Optional
from sklearn.model_selection import train_test_split  # type: ignore
from math_evals.MLE import min_sample_size_safe_mle_wald
from typing import Coroutine
from mmlu_pro.total_eval_process_mmlu_pro import mmlu_pro_scoring
import asyncio

class IntelligenceEvalInput(BaseModel):
    model: str
    config: Optional[dict[str, Any]] = {}
    hle: bool
    mmlu_pro: bool

class IntelligenceEvalOutput(BaseModel):
    hle_accuracy: float | None = None
    hle_ci: tuple[float, float] | None = None
    mmlu_pro_accuracy: float | None = None
    mmlu_pro_ci: tuple[float, float] | None = None

logging.basicConfig(level=logging.INFO)
logger = getLogger("__main__")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

hle_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "hle_dataset.csv")) # type: ignore
mmlu_pro_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "mmlu_pro_dataset.csv"), sep="\t") # type: ignore

if not load_dotenv(os.path.join(os.getcwd(), ".private.env")):
    raise FileNotFoundError("Private Environment File Not Found")

openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""

hle_app = FastAPI()

@hle_app.post("/llm/general")
async def general_llm_eval(payload: IntelligenceEvalInput) -> IntelligenceEvalOutput:
    async_tasks: list[Coroutine[Any, Any, dict[str, float | tuple[float, float] | None] | None]] = []

    if payload.hle:
        hle_dataset, _ = train_test_split(hle_dataset_full, train_size = min_sample_size_safe_mle_wald("bernoulli", len(hle_dataset_full), eps = 0.04), stratify = hle_dataset_full["category"], random_state = None)  # type: ignore
        async_tasks.append(hle_scoring(openrouter_api_key, logger, payload.model, "google/gemini-flash-1.5-8b", hle_dataset)) # type: ignore
    
    if payload.mmlu_pro:
        mmlu_pro_dataset, _ = train_test_split(mmlu_pro_dataset_full, train_size = min_sample_size_safe_mle_wald("bernoulli", len(mmlu_pro_dataset_full), eps = 0.04), stratify = mmlu_pro_dataset_full["category"], random_state = None)  # type: ignore
        async_tasks.append(mmlu_pro_scoring(openrouter_api_key, logger, payload.model, "google/gemini-flash-1.5-8b", mmlu_pro_dataset)) # type: ignore

    results = await asyncio.gather(*async_tasks)
    
    hle_accuracy = None
    hle_ci = None
    mmlu_pro_accuracy = None
    mmlu_pro_ci = None
    
    for result in results:
        if result is not None:
            if 'hle_accuracy' in result and isinstance(result['hle_accuracy'], float):
                hle_accuracy = result['hle_accuracy']
            if 'hle_ci' in result and isinstance(result['hle_ci'], tuple):
                hle_ci = result['hle_ci']
            if 'mmlu_pro_accuracy' in result and isinstance(result['mmlu_pro_accuracy'], float):
                mmlu_pro_accuracy = result['mmlu_pro_accuracy']
            if 'mmlu_pro_ci' in result and isinstance(result['mmlu_pro_ci'], tuple):
                mmlu_pro_ci = result['mmlu_pro_ci']

    return IntelligenceEvalOutput(
        hle_accuracy=hle_accuracy,
        hle_ci=hle_ci,
        mmlu_pro_accuracy=mmlu_pro_accuracy,
        mmlu_pro_ci=mmlu_pro_ci
    )

if __name__ == "__main__":
    try:
        logger.info("HLE Server Starting...")
        uvicorn.run(hle_app, host="127.0.0.1", port=3000, log_level="info")
    except Exception as e:
        logger.error(f"Error in HLE Server: {e}")
        raise HTTPException(status_code=500)