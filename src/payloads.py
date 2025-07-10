from pydantic import BaseModel
from typing import Any, Optional
import pandas as pd

class GeneralHLEEval(BaseModel):
    model: str
    config: Optional[dict[str, Any]] = {}

async def create_general_eval_payload(model: str, row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row["question"])
    image = str(row["image"])
    if image != "nan":
        return {
            "model": model,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": image
                        }
                    ]
                }
            ],
            **config
        }
    else:
        return {
            "model": model,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        }
                    ]
                }
            ],
            **config
        }
