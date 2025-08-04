import pandas as pd
import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from hle.total_eval_process_hle import hle_scoring
import uvicorn
from typing import Any, Coroutine
from sklearn.model_selection import train_test_split
from math_evals.MLE import min_sample_size_safe_mle_wald
from typing import Coroutine
from mmlu_pro.total_eval_process_mmlu_pro import mmlu_pro_scoring
import asyncio
from payloads import IntelligenceEvalInput, IntelligenceEvalOutput

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmn_pckgs.logger import get_logger

logger = get_logger("intelligence_server")

hle_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "hle_dataset.csv"))
mmlu_pro_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "mmlu_pro_dataset.csv"), sep="\t")

if not load_dotenv(os.path.join(os.getcwd(), ".private.env")):
    raise FileNotFoundError("Private Environment File Not Found")

openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""

with open(os.path.join(os.getcwd(), "prompts", "hle", "HLE_SYS_PROMPT_MC.md"), "r", encoding="utf-8") as f:
    hle_sys_prompt_mc = f.read()
with open(os.path.join(os.getcwd(), "prompts", "hle", "HLE_SYS_PROMPT_EX.md"), "r", encoding="utf-8") as f:
    hle_sys_prompt_ex = f.read()
with open(os.path.join(os.getcwd(), "prompts", "mmlu_pro", "MMLU_PRO_5SHOT.md"), "r", encoding="utf-8") as f:
    mmlu_pro_5shot = f.read()

hle_app = FastAPI()

@hle_app.post("/llm/general")
async def general_llm_eval(payload: IntelligenceEvalInput):
    async_tasks: list[Coroutine[Any, Any, dict[str, float | tuple[float, float] | None] | None]] = []

    if payload.hle:
        hle_dataset, _ = train_test_split(hle_dataset_full, train_size = min_sample_size_safe_mle_wald("bernoulli", len(hle_dataset_full), eps = 0.04), stratify = hle_dataset_full["category"], random_state = None)
        hle_dataset = pd.DataFrame(hle_dataset, columns=hle_dataset_full.columns)
        async_tasks.append(hle_scoring(openrouter_api_key, payload.agent_url, payload.agent_params, logger, "google/gemini-flash-1.5-8b", hle_dataset, hle_sys_prompt_mc, hle_sys_prompt_ex, payload.prompt_param_name, payload.image_param_name, payload.images_enabled))
    
    if payload.mmlu_pro:
        mmlu_pro_dataset, _ = train_test_split(mmlu_pro_dataset_full, train_size = min_sample_size_safe_mle_wald("bernoulli", len(mmlu_pro_dataset_full), eps = 0.04), stratify = mmlu_pro_dataset_full["category"], random_state = None)
        mmlu_pro_dataset = pd.DataFrame(mmlu_pro_dataset, columns=mmlu_pro_dataset_full.columns)
        async_tasks.append(mmlu_pro_scoring(openrouter_api_key, payload.agent_url, payload.agent_params, logger, "google/gemini-flash-1.5-8b", mmlu_pro_dataset, payload.prompt_param_name))
    
    try:
        results = await asyncio.gather(*async_tasks)
    except Exception as e:
        logger.error(f"Error in Intelligence Server: {e}")
        raise HTTPException(status_code=500)
    
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
        agent_name=payload.agent_name if payload.agent_name else None,
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