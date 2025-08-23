import asyncio
import pandas as pd
from logging import Logger
from async_llm_call import response_generator_openrouter
from gpqa_diamond.payloads_gpqa import create_gpqa_score_payload
import time
from typing import Any, Coroutine
from math_evals.MLE import Wald_CI
import requests
import copy
import random
from eval_json_parser import parse_eval_json

async def init_call_gpqa(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, row: pd.Series, prompt_param_name: Any) -> bool | None:
    try:
        agent_params_copy = copy.deepcopy(agent_params)
        
        question = str(row["Question"])
        correct_answer = str(row["Correct Answer"])
        incorrect_answers = [str(row["Incorrect Answer 1"]), str(row["Incorrect Answer 2"]), str(row["Incorrect Answer 3"])]
        answer_list = [correct_answer] + incorrect_answers
        random.shuffle(answer_list)

        for idx, option in enumerate(["A) ", "B) ", "C) ", "D) "]):
            labeled_option = option + answer_list[idx]
            if answer_list[idx] == correct_answer:
                correct_answer = labeled_option
            answer_list[idx] = labeled_option
        options = "\n".join(answer_list)
        question_with_options = f"Please keep your response short and concise. Question: {question} \n\n Options: {options}"
        agent_params_copy[prompt_param_name] = question_with_options
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
        payload_eval = create_gpqa_score_payload(model_eval, question_with_options, correct_answer, response_content)
        response_eval = await response_generator_openrouter(openrouter_key, payload_eval, logger)
        if not response_eval["success"] or response_eval["content"] is None:
            return None
        else:
            correct = parse_eval_json(response_eval["content"])
        
        return correct
    except Exception as e:
        logger.error(f"Error Evaluating Agent: {e}")
        return None


async def gpqa_scoring(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, gpqa_dataset: pd.DataFrame, prompt_param_name: Any, total_dataset_size: int) -> dict[str, float | tuple[float, float] | None] | None:
    
    try:
        time_start = time.time()
        llm_tasks: list[Coroutine[Any, Any, bool | None]] = [init_call_gpqa(openrouter_key, agent_url, agent_params, logger, model_eval, row, prompt_param_name) for _, row in gpqa_dataset.iterrows()]
        results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        results = [result for result in results if (result is not None and not isinstance(result, BaseException))]
        successful_results: int = 0
        total_results: int = len(results)
        for result in results:
            if result:
                successful_results += 1
    except Exception as e:
        logger.error(f"Error in gpqa_scoring: {e}")
        return

    try:
        accuracy = successful_results / total_results
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Successful Results: {successful_results}")
        logger.info(f"Total Results: {total_results}")
    except ZeroDivisionError:
        logger.info("No results found. Invalid LLM calls.")
        return
    
    gpqa_ci = Wald_CI("bernoulli", total_dataset_size, total_results, accuracy)
    
    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start} seconds")

    return {"gpqa_accuracy": round(accuracy, 4), "gpqa_ci": gpqa_ci}