import asyncio
import requests
from typing import Any
from logging import Logger

async def response_generator_openrouter(openrouter_key: str, payload: dict[str, Any], logger: Logger) -> tuple[str | None, bool]:
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }
        response = await asyncio.to_thread(requests.post, url, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"], True
    except Exception as e:
        msg = f"Error in LLM Call: {e}"
        logger.error(msg)
        return msg, False