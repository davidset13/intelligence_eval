from typing import Any
from logging import Logger
import requests
import asyncio

async def response_generator_openrouter(openrouter_key: str, payload: dict[str, Any], logger: Logger) -> dict[str, Any]:
    
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }
        response = await asyncio.to_thread(requests.post, url, headers=headers, json=payload)
        return {"content": response.json()["choices"][0]["message"]["content"], "success": True}
    except Exception as e:
        msg = f"Error in LLM Call: {e}"
        logger.error(msg)
        return {"content": msg, "success": False}