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
from gpqa_diamond.total_eval_process_gpqa import gpqa_scoring
from livebench.total_eval_process_livebench import livebench_scoring
import asyncio
from payloads import IntelligenceEvalInput, IntelligenceEvalOutput

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmn_pckgs.python.logger import get_logger

logger = get_logger("intelligence_server")

hle_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "hle_dataset.csv"))
mmlu_pro_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "mmlu_pro_dataset.csv"))
gpqa_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "gpqa_dataset.csv"))
livebench_dataset_full: pd.DataFrame = pd.read_csv(os.path.join(os.getcwd(), "utility", "livebench_dataset.csv"))

if not load_dotenv(os.path.join(os.getcwd(), ".private.env")):
    raise FileNotFoundError("Private Environment File Not Found")

openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""

with open(os.path.join(os.getcwd(), "prompts", "hle", "HLE_SYS_PROMPT_MC.md"), "r", encoding="utf-8") as f:
    hle_sys_prompt_mc = f.read()
with open(os.path.join(os.getcwd(), "prompts", "hle", "HLE_SYS_PROMPT_EX.md"), "r", encoding="utf-8") as f:
    hle_sys_prompt_ex = f.read()

hle_app = FastAPI()

@hle_app.post("/llm/general")
async def general_llm_eval(payload: IntelligenceEvalInput):
    async_tasks: list[Coroutine[Any, Any, dict[str, float | tuple[float, float] | None] | None]] = []
    
    if payload.hle:
        if payload.hle_categories == ["all"]:
            hle_dataset_mod = hle_dataset_full
        else:
            hle_dataset_mod = pd.DataFrame()
            for category in payload.hle_categories:
                hle_dataset_mod = pd.concat([hle_dataset_mod, hle_dataset_full[hle_dataset_full["category"] == category]])
        
        if not payload.images_enabled:
            hle_dataset_mod = hle_dataset_mod[hle_dataset_mod["image"].isna()]

        #hle_dataset, _ = train_test_split(hle_dataset_mod, train_size = min_sample_size_safe_mle_wald("bernoulli", len(hle_dataset_mod), eps = 0.1, alpha = 0.1), stratify = hle_dataset_mod["category"], random_state = None)
        hle_dataset, _ = train_test_split(hle_dataset_mod[0:250], train_size = 8, stratify = hle_dataset_mod[0:250]["category"], random_state = None)
        hle_dataset = pd.DataFrame(hle_dataset, columns=hle_dataset_full.columns)
        async_tasks.append(hle_scoring(openrouter_api_key, payload.agent_url, payload.agent_params, logger, "google/gemini-flash-1.5-8b", hle_dataset, hle_sys_prompt_mc, hle_sys_prompt_ex, payload.prompt_param_name, payload.image_param_name, payload.images_enabled, len(hle_dataset_mod)))
    
    if payload.mmlu_pro:
        if payload.mmlu_pro_categories == ["all"]:
            mmlu_pro_dataset_mod = mmlu_pro_dataset_full
        else:
            mmlu_pro_dataset_mod = pd.DataFrame()
            for category in payload.mmlu_pro_categories:
                mmlu_pro_dataset_mod = pd.concat([mmlu_pro_dataset_mod, mmlu_pro_dataset_full[mmlu_pro_dataset_full["category"] == category]])

        mmlu_pro_dataset, _ = train_test_split(mmlu_pro_dataset_mod, train_size = min_sample_size_safe_mle_wald("bernoulli", len(mmlu_pro_dataset_mod), eps = 0.1, alpha = 0.1), stratify = mmlu_pro_dataset_mod["category"], random_state = None)
        mmlu_pro_dataset = pd.DataFrame(mmlu_pro_dataset, columns=mmlu_pro_dataset_full.columns)
        async_tasks.append(mmlu_pro_scoring(openrouter_api_key, payload.agent_url, payload.agent_params, logger, "google/gemini-flash-1.5-8b", mmlu_pro_dataset, payload.prompt_param_name, len(mmlu_pro_dataset_mod)))

    if payload.gpqa:
        gpqa_dataset, _ = train_test_split(gpqa_dataset_full, train_size = min_sample_size_safe_mle_wald("bernoulli", len(gpqa_dataset_full), eps = 0.1, alpha = 0.1), random_state = None)
        gpqa_dataset = pd.DataFrame(gpqa_dataset, columns=gpqa_dataset_full.columns)
        async_tasks.append(gpqa_scoring(openrouter_api_key, payload.agent_url, payload.agent_params, logger, "google/gemini-flash-1.5-8b", gpqa_dataset, payload.prompt_param_name, len(gpqa_dataset_full)))

    if payload.livebench:
        if payload.livebench_categories == ["all"]:
            livebench_dataset_mod = livebench_dataset_full
        else:
            livebench_dataset_mod = pd.DataFrame()
            for category in payload.livebench_categories:
                livebench_dataset_mod = pd.concat([livebench_dataset_mod, livebench_dataset_full[livebench_dataset_full["category"] == category]])

        livebench_dataset, _ = train_test_split(livebench_dataset_mod, train_size = min_sample_size_safe_mle_wald("bernoulli", len(livebench_dataset_mod), eps = 0.1, alpha = 0.1), stratify = livebench_dataset_mod["category"], random_state = None)
        livebench_dataset = pd.DataFrame(livebench_dataset, columns=livebench_dataset_full.columns)
        async_tasks.append(livebench_scoring(openrouter_api_key, payload.agent_url, payload.agent_params, logger, "google/gemini-flash-1.5-8b", livebench_dataset, payload.prompt_param_name, len(livebench_dataset_mod)))
    
    try:
        results = await asyncio.gather(*async_tasks)
    except Exception as e:
        logger.error(f"Error in Intelligence Server: {e}")
        raise HTTPException(status_code=500)
        
    final_resp = IntelligenceEvalOutput(agent_name=payload.agent_name)
    for result in results:
        if result is not None:
            for key, value in result.items():
                if key in IntelligenceEvalOutput.model_fields.keys():
                    setattr(final_resp, key, value)

    return final_resp

if __name__ == "__main__":
    try:
        logger.info("Intelligence Server Starting...")
        uvicorn.run(hle_app, host="127.0.0.1", port=3000, log_level="info")
    except Exception as e:
        logger.error(f"Error in HLE Server: {e}")
        raise HTTPException(status_code=500)