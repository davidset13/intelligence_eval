import asyncio
import pandas as pd
from logging import Logger
from async_llm_call import response_generator_openrouter
from mmlu_pro.payloads_mmlu_pro import create_mmlu_pro_score_payload
import time
from typing import Any, Coroutine
from math_evals.MLE import Wald_CI
import requests
import copy
from collections import defaultdict
from eval_json_parser import parse_eval_json
from payloads import gpt_dataset, gpt_ds_name
import os

value_counts = {
    "MAT": 1351,
    "PHY": 1299,
    "CHE": 1132,
    "LAW": 1101,
    "ENG": 969,
    "OTH": 924,
    "ECO": 844,
    "HEA": 818,
    "PSY": 798,
    "BUS": 789,
    "BIO": 717,
    "PHI": 499,
    "COM": 410,
    "HIS": 381
}


async def init_call_mmlu_pro(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, row: pd.Series, prompt_param_name: Any) -> tuple[bool, str] | None:
    try:
        agent_params_copy = copy.deepcopy(agent_params)

        question = str(row["question"])
        options = str(row["options"])
        correct_answer = str(row["answer"])
        category = str(row["category"])

        agent_params_copy[prompt_param_name] = f"Please keep your response short and concise. \n\n Question: {question} \n\n Options: {options}"
        response = await asyncio.to_thread(requests.post, agent_url, json=agent_params_copy)
        
        try:
            response_content = response.json()
        except:
            response_content = response.text
        
        if len(response_content) == 0 or response_content is None:
            return None
    except Exception as e:
        logger.error(f"Error Calling Agent: {e}")
        return None
    try:
        payload_eval = create_mmlu_pro_score_payload(model_eval, question, correct_answer, options, response_content)
        response_eval = await response_generator_openrouter(openrouter_key, payload_eval, logger)
        if not response_eval["success"] or response_eval["content"] is None:
            return None
        else:
            correct = parse_eval_json(response_eval["content"])

        gpt_dataset.loc[len(gpt_dataset)] = [question, category, response_content, correct_answer, correct]
        logger.info(len(gpt_dataset))
        
        return correct, category
    except Exception as e:
        logger.error(f"Error Evaluating Agent: {e}")
        return None


async def mmlu_pro_scoring(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, mmlu_pro_dataset: pd.DataFrame, prompt_param_name: Any, total_dataset_size: int) -> dict[str, float | tuple[float, float] | None] | None:

    try:
        time_start = time.time()
        llm_tasks: list[Coroutine[Any, Any, tuple[bool, str] | None]] = [init_call_mmlu_pro(openrouter_key, agent_url, agent_params, logger, model_eval, row, prompt_param_name) for _, row in mmlu_pro_dataset.iterrows()]
        results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        results = [result for result in results if (result is not None and not isinstance(result, BaseException))]
        category_results: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for result in results:
            if result[0]:
                category_results[result[1]][0] += 1
            category_results[result[1]][1] += 1

    except Exception as e:
        logger.error(f"Error in mmlu_pro_scoring: {e}")
        return

    total_success = sum(result[0] for result in category_results.values())
    total_results = sum(result[1] for result in category_results.values())

    try:
        accuracy = total_success / total_results
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Successful Results: {total_success}")
        logger.info(f"Total Results: {total_results}")
        mmlu_pro_ci = Wald_CI("bernoulli", total_dataset_size, total_results, accuracy)
        resp_dict = {"mmlu_pro_accuracy": round(accuracy, 4), "mmlu_pro_ci": mmlu_pro_ci}
    except ZeroDivisionError:
        logger.info("No results found. Invalid LLM calls.")
        return
    
    category_accuracy = {}
    for category, result in category_results.items():
        try:
            accuracy = result[0] / result[1]
            category_accuracy[category] = (round(accuracy, 4), Wald_CI("bernoulli", value_counts[category], result[1], accuracy))
        except ZeroDivisionError:
            logger.info(f"No results found for {category}. Invalid LLM calls.")
            category_accuracy[category] = (Exception, "No results found")
    
    resp_dict['mmlu_pro_categories'] = category_accuracy

    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start} seconds")

    gpt_dataset.to_csv(os.path.join(os.getcwd(), "agent_ans", f"{gpt_ds_name}.csv"), encoding="utf-8", index=False)

    return resp_dict