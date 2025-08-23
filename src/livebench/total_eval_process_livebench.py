import asyncio
import pandas as pd
from logging import Logger
from async_llm_call import response_generator_openrouter
from livebench.payloads_livebench import create_livebench_score_payload
import time
from typing import Any, Coroutine
from math_evals.MLE import Wald_CI
import requests
import copy
from collections import defaultdict
from eval_json_parser import parse_eval_json, eval_livebench_if

value_counts = {
    "INS": 200,
    "MAT": 182,
    "DAT": 150,
    "REA": 100,
    "LAN": 50
}


async def init_call_livebench(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, row: pd.Series, prompt_param_name: Any) -> tuple[bool | float, str] | None:
    
    try:
        agent_params_copy = copy.deepcopy(agent_params)

        func_args = {}
        category = str(row["category"])
        if category != "INS":
            func_args["question"] = str(row["turns"])
            func_args["correct_answer"] = str(row["ground_truth"])
            agent_params_copy[prompt_param_name] = f"Please keep your response short and concise. Question: {func_args["question"]}"
        else:
            func_args["task"] = str(row["task_prompt"])
            func_args["input_text"] = str(row["turns"])
            agent_params_copy[prompt_param_name] = f"Please keep your response short and concise. Task: {func_args["task"]} \n\n Input Text: {func_args["input_text"]}"
        
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
        payload_eval = create_livebench_score_payload(model = model_eval, category = category, response = response_content, **func_args)
        response_eval = await response_generator_openrouter(openrouter_key, payload_eval, logger)
        if not response_eval["success"] or response_eval["content"] is None:
            return None
        else:
            if category != "INS":
                correct = parse_eval_json(response_eval["content"])
                return correct, category
            else:
                correct = eval_livebench_if(response_eval["content"])
                return correct, category
    except Exception as e:
        logger.error(f"Error Evaluating Agent: {e}")
        return None


async def livebench_scoring(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, livebench_dataset: pd.DataFrame, prompt_param_name: Any, total_dataset_size: int) -> dict[str, float | tuple[float, float] | None] | None:

    try:
        time_start = time.time()
        llm_tasks: list[Coroutine[Any, Any, tuple[bool | float | int, str] | None]] = [init_call_livebench(openrouter_key, agent_url, agent_params, logger, model_eval, row, prompt_param_name) for _, row in livebench_dataset.iterrows()]
        results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        results = [result for result in results if (result is not None and not isinstance(result, BaseException))]
        category_results: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
        for result in results:
            if isinstance(result[0], bool) and result[0]:
                category_results[result[1]][0] += 1
            elif isinstance(result[0], float) or isinstance(result[0], int) and result[0] > 0:
                category_results[result[1]][0] += result[0]
            category_results[result[1]][1] += 1

    except Exception as e:
        logger.error(f"Error in livebench_scoring: {e}")
        return

    total_success = sum(result[0] for result in category_results.values())
    total_results = sum(result[1] for result in category_results.values())

    try:
        accuracy = total_success / total_results
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Successful Results: {total_success}")
        logger.info(f"Total Results: {total_results}")
        livebench_ci = Wald_CI("bernoulli", total_dataset_size, int(total_results), accuracy)
        resp_dict = {"livebench_accuracy": round(accuracy, 4), "livebench_ci": livebench_ci}
    except ZeroDivisionError:
        logger.info("No results found. Invalid LLM calls.")
        return
    
    category_accuracy = {}
    for category, result in category_results.items():
        try:
            accuracy = result[0] / result[1]
            category_accuracy[category] = (round(accuracy, 4), Wald_CI("bernoulli", value_counts[category], int(result[1]), accuracy))
        except ZeroDivisionError:
            logger.info(f"No results found for {category}. Invalid LLM calls.")
            category_accuracy[category] = (Exception, "No results found")
    
    resp_dict['livebench_categories'] = category_accuracy

    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start} seconds")

    return resp_dict