import asyncio
import ast
import pandas as pd
from logging import Logger
from async_llm_call import response_generator_openrouter
from mmlu_pro.payloads_mmlu_pro import create_mmlu_pro_init_payload, create_mmlu_pro_score_payload
import time
from typing import Any, Coroutine
from math_evals.MLE import Wald_CI


async def init_call_mmlu_pro(openrouter_key: str, logger: Logger, model_init: str, model_eval: str, row: pd.Series, mmlu_pro_5shot: str) -> bool | None:
    payload_init = create_mmlu_pro_init_payload(model_init, row, {}, mmlu_pro_5shot)
    response = await response_generator_openrouter(openrouter_key, payload_init, logger)
    if not response["success"] or response["content"] is None:
        return None
    else:
        payload_eval = create_mmlu_pro_score_payload(model_eval, row, response["content"], {})
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


async def mmlu_pro_scoring(openrouter_key: str, logger: Logger, model_init: str, model_eval: str, mmlu_pro_dataset: pd.DataFrame, mmlu_pro_5shot: str) -> dict[str, float | tuple[float, float] | None] | None:

    try:
        time_start = time.time()
        llm_tasks: list[Coroutine[Any, Any, bool | None]] = [init_call_mmlu_pro(openrouter_key, logger, model_init, model_eval, row, mmlu_pro_5shot) for _, row in mmlu_pro_dataset.iterrows()]
        results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        results = [result for result in results if (result is not None and not isinstance(result, BaseException))]
        successful_results: int = 0
        total_results: int = len(results)
        for result in results:
            if result:
                successful_results += 1
    except Exception as e:
        logger.error(f"Error in mmlu_pro_scoring: {e}")
        return

    try:
        accuracy = successful_results / total_results
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Successful Results: {successful_results}")
        logger.info(f"Total Results: {total_results}")
    except ZeroDivisionError:
        logger.info("No results found. Invalid LLM calls.")
        return
    
    mmlu_pro_ci = Wald_CI("bernoulli", 2500, total_results, accuracy)
    
    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start} seconds")

    return {"mmlu_pro_accuracy": round(accuracy, 4), "mmlu_pro_ci": mmlu_pro_ci}