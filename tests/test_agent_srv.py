from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from typing import Any, Optional
from dotenv import load_dotenv
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.async_llm_call import response_generator_openrouter
from cmn_pckgs.logger import get_logger

if not load_dotenv(os.path.join(os.getcwd(), ".private.env")):
    raise FileNotFoundError("Private Environment File Not Found")

openrouter_api_key: str= os.getenv("OPENROUTER_API_KEY") or ""

logger = get_logger("test_agent_srv")

class TestAgentPayload(BaseModel):
    model: str
    prompt: str
    image_enabled: bool = True
    image: Optional[str] = None
    output_type: Optional[str] = None

test_agent = FastAPI()


@test_agent.post("/test_agent")
async def test_agent_post(payload: TestAgentPayload):
    if not payload.image_enabled and payload.image:
        raise HTTPException(status_code=400, detail="Image is not enabled")

    payload_llm: dict[str, Any] = {
        "model": payload.model,
        "messages": [
            {
                "role": "user",
                "content": payload.prompt
            }
        ],
    }

    if payload.image_enabled and payload.image:
        payload_llm["messages"].append({
            "role": "image_url",
            "content": payload.image
        })

    if payload.output_type:
        payload_llm["response_format"] = {
            "type": payload.output_type
        }

    response = await response_generator_openrouter(openrouter_api_key, payload_llm, logger)
    return response["content"]


if __name__ == "__main__":
    try:
        logger.info("Test Agent Server Starting...")
        uvicorn.run(test_agent, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"Error in Test Agent Server: {e}")
        raise HTTPException(status_code=500)