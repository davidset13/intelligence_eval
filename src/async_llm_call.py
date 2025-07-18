import asyncio
import requests
from typing import Any
from logging import Logger
from payloads import create_general_eval_payload, create_general_score_payload
import pandas as pd
import ast

def response_generator_openrouter(openrouter_key: str, payload: dict[str, Any], logger: Logger) -> tuple[str | None, bool]:
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"], True
    except Exception as e:
        msg = f"Error in LLM Call: {e}"
        #logger.error(msg)
        return msg, False

async def total_eval_process(openrouter_key: str, logger: Logger, model_init: str, model_eval: str, row: pd.Series) -> bool | str | None:
    payload_init = create_general_eval_payload(model_init, row, {})
    response = await asyncio.to_thread(response_generator_openrouter, openrouter_key, payload_init, logger)
    if not response[1]:
        return False
    elif response[0] is None:
        return False
    else:
        payload_eval = create_general_score_payload(model_eval, row, response[0], {})
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

                        
