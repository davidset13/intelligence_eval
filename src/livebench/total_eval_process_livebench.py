import asyncio
import ast
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

value_counts = {
    "INS": 200,
    "DAT": 150,
    "REA": 100,
}

async def init_call_livebench(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, row: pd.Series, prompt_param_name: Any) -> tuple[bool, str] | None:
    try:
        agent_params_copy = copy.deepcopy(agent_params)

        category = str(row["category"])
        if category == "REA" or category == "DAT":
            question = str(row["turns"])
            correct_answer = str(row["ground_truth"])

        agent_params_copy[prompt_param_name] = f"Question: {question}"
        response = await asyncio.to_thread(requests.post, agent_url, json=agent_params_copy)
        response_content = response.json()
        if len(response_content) == 0 or response_content is None:
            return None
    except Exception as e:
        logger.error(f"Error Calling Agent: {e}")
        return None
    try:
        payload_eval = create_livebench_score_payload(model_eval, question, correct_answer, response_content)
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


async def livebench_scoring(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, livebench_dataset: pd.DataFrame, prompt_param_name: Any, total_dataset_size: int) -> dict[str, float | tuple[float, float] | None] | None:

    try:
        time_start = time.time()
        llm_tasks: list[Coroutine[Any, Any, tuple[bool, str] | None]] = [init_call_livebench(openrouter_key, agent_url, agent_params, logger, model_eval, row, prompt_param_name) for _, row in livebench_dataset.iterrows()]
        results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        results = [result for result in results if (result is not None and not isinstance(result, BaseException))]
        category_results: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for result in results:
            if result[0]:
                category_results[result[1]][0] += 1
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
        livebench_ci = Wald_CI("bernoulli", total_dataset_size, total_results, accuracy)
        resp_dict = {"livebench_accuracy": round(accuracy, 4), "livebench_ci": livebench_ci}
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
    
    resp_dict['livebench_categories'] = category_accuracy

    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start} seconds")

    return resp_dict