from typing import Any
import pandas as pd

# THIS METHOD IS DEPRECATED AND WILL BE REMOVED IN FUTURE RELEASE
def create_mmlu_pro_init_payload(model: str, row: pd.Series, mmlu_pro_5shot: str) -> dict[str, Any]:
    question = str(row["question"])
    options = str(row["options"])
    return {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": mmlu_pro_5shot
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Question: {question} \n Options: {options}"
                            },
                        ]
                    }
                ],
            }