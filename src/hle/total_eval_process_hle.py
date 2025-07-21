import asyncio
import ast
import pandas as pd
from logging import Logger
from async_llm_call import response_generator_openrouter
from hle.payloads_hle import create_hle_init_payload, create_hle_score_payload
import time
from typing import Any, Coroutine


async def total_eval_process_hle(openrouter_key: str, logger: Logger, model_init: str, model_eval: str, row: pd.Series) -> bool | str | None:
    payload_init = create_hle_init_payload(model_init, row, {})
    response = await asyncio.to_thread(response_generator_openrouter, openrouter_key, payload_init, logger)
    if not response[1]:
        return False
    elif response[0] is None:
        return False
    else:
        payload_eval = create_hle_score_payload(model_eval, row, response[0], {})
        response_eval = await asyncio.to_thread(response_generator_openrouter, openrouter_key, payload_eval, logger)
        if not response_eval[1]:
            return False
        else:
            try:
                if response_eval[0] is None:
                    return False
                return ast.literal_eval(response_eval[0])['correct']
            except Exception:
                try:
                    if response_eval[0] is not None and ('"correct":' in response_eval[0] or "'correct':" in response_eval[0]):
                        idx1 = response_eval[0].find('"correct":')
                        idx2 = response_eval[0].find("'correct':")
                        if idx1 != -1 and idx2 != -1:
                            return False
                        elif idx1 != -1:
                            if_yes = response_eval[0][idx1:].find('yes')
                            if_no = response_eval[0][idx1:].find('no')
                            if if_yes == -1:
                                return False
                            elif if_yes < if_no:
                                return 'yes'
                            else:
                                return False
                        elif idx2 != -1:
                            if_yes = response_eval[0][idx2:].find('yes')
                            if_no = response_eval[0][idx2:].find('no')
                            if if_yes == -1:
                                return False
                            elif if_yes < if_no:
                                return 'yes'
                            else:
                                return False
                        else:
                            return False
                    else:
                        return False
                except Exception:
                    return False


async def hle_scoring(openrouter_key: str, logger: Logger, model_init: str, model_eval: str, hle_dataset: pd.DataFrame) -> float | None:
    
    time_start = time.time()
    llm_tasks: list[Coroutine[Any, Any, bool | str | None]] = [total_eval_process_hle(openrouter_key, logger, model_init, model_eval, row) for _, row in hle_dataset.iterrows()]
    results = await asyncio.gather(*llm_tasks)
    results = [result for result in results if (result is not None and result is not False)]

    successful_results: int = 0
    total_results: int = len(results)
    for result in results:
        if result == 'yes':
            successful_results += 1

    try:
        accuracy = successful_results / total_results
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Successful Results: {successful_results}")
        logger.info(f"Total Results: {total_results}")
    except ZeroDivisionError:
        logger.info("No results found. Invalid LLM calls.")
        return
    
    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start} seconds")
    return round(accuracy, 2)