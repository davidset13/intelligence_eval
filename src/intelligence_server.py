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

class IntelligenceEvalInput(BaseModel):
    model: str
    config: Optional[dict[str, Any]] = {}
    hle: bool

class IntelligenceEvalOutput(BaseModel):
    hle_accuracy: float | None

logging.basicConfig(level=logging.INFO)
logger = getLogger("__main__")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

hle_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "hle_dataset.csv")) # type: ignore

hle_dataset = hle_dataset_full.sample(20)

if not load_dotenv(os.path.join(os.getcwd(), ".private.env")):
    raise FileNotFoundError("Private Environment File Not Found")

openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""

hle_app = FastAPI()

@hle_app.post("/llm/general")
async def general_llm_eval(payload: IntelligenceEvalInput) -> IntelligenceEvalOutput:
    if payload.hle:
        hle_accuracy = await hle_scoring(openrouter_api_key, logger, payload.model, "google/gemini-flash-1.5-8b", hle_dataset)
    else:
        hle_accuracy = None

    return IntelligenceEvalOutput(hle_accuracy=hle_accuracy)

if __name__ == "__main__":
    try:
        logger.info("HLE Server Starting...")
        uvicorn.run(hle_app, host="127.0.0.1", port=3000, log_level="info")
    except Exception as e:
        logger.error(f"Error in HLE Server: {e}")
        raise HTTPException(status_code=500)