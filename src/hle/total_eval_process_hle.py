import asyncio
import ast
import pandas as pd
from logging import Logger
from async_llm_call import response_generator_openrouter
from hle.payloads_hle import create_hle_score_payload
import time
from typing import Any, Coroutine
from math_evals.MLE import Wald_CI
import requests
import copy
from collections import defaultdict

value_counts = {
    "MAT": 1021,
    "BIO": 280,
    "COM": 241,
    "OTH": 233,
    "PHY": 230,
    "HUM": 219,
    "CHE": 165,
    "ENG": 111
}

async def init_call_hle(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, row: pd.Series, hle_sys_prompt_mc: str, hle_sys_prompt_ex: str, prompt_param_name: Any, image_param_name: Any, images_enabled: bool) -> tuple[bool, str] | None:
    
    try:
        agent_params_copy = copy.deepcopy(agent_params)
        
        question = str(row["question"])
        image = str(row["image"])
        answer_type = str(row["answer_type"])
        correct_answer = str(row["answer"])
        category = str(row["category"])
        
        if image != "nan":
            if not images_enabled:
                return None
            agent_params_copy[image_param_name] = image
        if answer_type == "multiple_choice":
            agent_params_copy[prompt_param_name] = hle_sys_prompt_mc + "\n\n" + question
        else:
            agent_params_copy[prompt_param_name] = hle_sys_prompt_ex + "\n\n" + question
        response = await asyncio.to_thread(requests.post, agent_url, json=agent_params_copy)
        response_content = response.json()
        if len(response_content) == 0 or response_content is None:
            return None
    except Exception as e:
        logger.error(f"Error Calling Agent: {e}")
        return None
    try:
        payload_eval = create_hle_score_payload(model_eval, question, correct_answer, response_content)
        response_eval = await response_generator_openrouter(openrouter_key, payload_eval, logger)
        if not response_eval["success"] or response_eval["content"] is None:
            return None
        else:
            try:
                correct = ast.literal_eval(response_eval["content"])['correct']
                if correct == 'yes':
                    correct = True
                else:
                    correct = False
            except:
                try:
                    if ('"correct":' in response_eval["content"] or "'correct':" in response_eval["content"]):
                        idx1 = response_eval["content"].find('"correct":')
                        idx2 = response_eval["content"].find("'correct':")
                        if idx1 != -1 and idx2 != -1:
                            correct = False
                        elif idx1 != -1:
                            if_yes = response_eval["content"][idx1:].find('yes')
                            if_no = response_eval["content"][idx1:].find('no')
                            if if_yes == -1:
                                correct = False
                            elif if_yes < if_no:
                                correct = True
                            else:
                                correct = False
                        elif idx2 != -1:
                            if_yes = response_eval["content"][idx2:].find('yes')
                            if_no = response_eval["content"][idx2:].find('no')
                            if if_yes == -1:
                                correct = False
                            elif if_yes < if_no:
                                correct = True
                            else:
                                correct = False
                        else:
                            correct = False
                    else:
                        correct = False
                except:
                    correct = False
        
        return correct, category
    except Exception as e:
        logger.error(f"Error Evaluating Agent: {e}")
        return None


async def hle_scoring(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, hle_dataset: pd.DataFrame, hle_sys_prompt_mc: str, hle_sys_prompt_ex: str, prompt_param_name: Any, image_param_name: Any, images_enabled: bool) -> dict[str, float | tuple[float, float] | None] | None:
    
    try:
        time_start = time.time()
        llm_tasks: list[Coroutine[Any, Any, tuple[bool, str] | None]] = [init_call_hle(openrouter_key, agent_url, agent_params, logger, model_eval, row, hle_sys_prompt_mc, hle_sys_prompt_ex, prompt_param_name, image_param_name, images_enabled) for _, row in hle_dataset.iterrows()]
        results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        results = [result for result in results if (result is not None and not isinstance(result, BaseException))]
        category_results: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for result in results:
            if result[0]:
                category_results[result[1]][0] += 1
            category_results[result[1]][1] += 1

    except Exception as e:
        logger.error(f"Error in hle_scoring: {e}")
        return

    total_success = sum(result[0] for result in category_results.values())
    total_results = sum(result[1] for result in category_results.values())

    try:
        accuracy = total_success / total_results
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Successful Results: {total_success}")
        logger.info(f"Total Results: {total_results}")
        hle_ci = Wald_CI("bernoulli", 2500, total_results, accuracy)
        resp_dict = {"hle_accuracy": round(accuracy, 4), "hle_ci": hle_ci}
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
    
    resp_dict['hle_categories'] = category_accuracy
    
    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start} seconds")

    return resp_dict