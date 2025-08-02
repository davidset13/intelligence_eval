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

async def init_call_hle(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, row: pd.Series, hle_sys_prompt_mc: str, hle_sys_prompt_ex: str, prompt_param_name: Any, image_param_name: Any, images_enabled: bool) -> bool | None:
    try:
        question = str(row["question"])
        image = str(row["image"])
        answer_type = str(row["answer_type"])
        if image != "nan":
            if images_enabled:
                return None
            agent_params[image_param_name] = image
        if answer_type == "multiple_choice":
            agent_params[prompt_param_name] = hle_sys_prompt_mc + "\n\n" + question
        else:
            agent_params[prompt_param_name] = hle_sys_prompt_ex + "\n\n" + question
        response = await asyncio.to_thread(requests.post, agent_url, json=agent_params)
        response_content = response.json()
        if len(response_content) == 0 or response_content is None:
            return None
    except Exception as e:
        logger.error(f"Error Calling Agent: {e}")
        return None
    try:
        payload_eval = create_hle_score_payload(model_eval, row, response_content)
        response_eval = await response_generator_openrouter(openrouter_key, payload_eval, logger)
        if not response_eval["success"] or response_eval["content"] is None:
            return None
        else:
            try:
                correct = ast.literal_eval(response_eval["content"])['correct']
                if correct == 'yes':
                    return True
                else:
                    return False
            except:
                try:
                    if ('"correct":' in response_eval["content"] or "'correct':" in response_eval["content"]):
                        idx1 = response_eval["content"].find('"correct":')
                        idx2 = response_eval["content"].find("'correct':")
                        if idx1 != -1 and idx2 != -1:
                            return False
                        elif idx1 != -1:
                            if_yes = response_eval["content"][idx1:].find('yes')
                            if_no = response_eval["content"][idx1:].find('no')
                            if if_yes == -1:
                                return False
                            elif if_yes < if_no:
                                return True
                            else:
                                return False
                        elif idx2 != -1:
                            if_yes = response_eval["content"][idx2:].find('yes')
                            if_no = response_eval["content"][idx2:].find('no')
                            if if_yes == -1:
                                return False
                            elif if_yes < if_no:
                                return True
                            else:
                                return False
                        else:
                            return False
                    else:
                        return False
                except:
                    return False
    except Exception as e:
        logger.error(f"Error Evaluating Agent: {e}")
        return None


async def hle_scoring(openrouter_key: str, agent_url: str, agent_params: dict[Any, Any], logger: Logger, model_eval: str, hle_dataset: pd.DataFrame, hle_sys_prompt_mc: str, hle_sys_prompt_ex: str, prompt_param_name: Any, image_param_name: Any, images_enabled: bool) -> dict[str, float | tuple[float, float] | None] | None:
    
    try:
        time_start = time.time()
        llm_tasks: list[Coroutine[Any, Any, bool | None]] = [init_call_hle(openrouter_key, agent_url, agent_params, logger, model_eval, row, hle_sys_prompt_mc, hle_sys_prompt_ex, prompt_param_name, image_param_name, images_enabled) for _, row in hle_dataset.iterrows()]
        results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        results = [result for result in results if (result is not None and not isinstance(result, BaseException))]
        successful_results: int = 0
        total_results: int = len(results)
        for result in results:
            if result:
                successful_results += 1
    except Exception as e:
        logger.error(f"Error in hle_scoring: {e}")
        return

    try:
        accuracy = successful_results / total_results
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Successful Results: {successful_results}")
        logger.info(f"Total Results: {total_results}")
    except ZeroDivisionError:
        logger.info("No results found. Invalid LLM calls.")
        return
    
    hle_ci = Wald_CI("bernoulli", 2500, total_results, accuracy)
    
    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start} seconds")

    return {"hle_accuracy": round(accuracy, 4), "hle_ci": hle_ci}