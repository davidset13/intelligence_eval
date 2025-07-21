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
from math_evals.MLE import min_sample_size_safe_mle_wald, Wald_CI

class IntelligenceEvalInput(BaseModel):
    model: str
    config: Optional[dict[str, Any]] = {}
    hle: bool

class IntelligenceEvalOutput(BaseModel):
    hle_accuracy: float | None
    hle_ci: tuple[float, float] | None

logging.basicConfig(level=logging.INFO)
logger = getLogger("__main__")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

hle_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "hle_dataset.csv")) # type: ignore

if not load_dotenv(os.path.join(os.getcwd(), ".private.env")):
    raise FileNotFoundError("Private Environment File Not Found")

openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""

hle_app = FastAPI()

@hle_app.post("/llm/general")
async def general_llm_eval(payload: IntelligenceEvalInput) -> IntelligenceEvalOutput:
    if payload.hle:
        hle_dataset, _ = train_test_split(hle_dataset_full, train_size = min_sample_size_safe_mle_wald("bernoulli", 2500, eps = 0.04), stratify = hle_dataset_full["category"], random_state = None)  # type: ignore
        hle_accuracy = await hle_scoring(openrouter_api_key, logger, payload.model, "google/gemini-flash-1.5-8b", hle_dataset)  # type: ignore
        hle_ci = Wald_CI("bernoulli", 2500, len(hle_dataset), hle_accuracy) if hle_accuracy is not None else None # type: ignore
    else:
        hle_accuracy = None
        hle_ci = None

    return IntelligenceEvalOutput(hle_accuracy=hle_accuracy, hle_ci=hle_ci)

if __name__ == "__main__":
    try:
        logger.info("HLE Server Starting...")
        uvicorn.run(hle_app, host="127.0.0.1", port=3000, log_level="info")
    except Exception as e:
        logger.error(f"Error in HLE Server: {e}")
        raise HTTPException(status_code=500)