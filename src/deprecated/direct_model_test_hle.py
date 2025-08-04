from typing import Any
import pandas as pd

# THIS METHOD IS DEPRECATED AND WILL BE REMOVED IN FUTURE RELEASE
def create_hle_init_payload(model: str, row: pd.Series, hle_sys_prompt_mc: str, hle_sys_prompt_ex: str) -> dict[str, Any]:
    question = str(row["question"])
    image = str(row["image"])
    answer_type = str(row["answer_type"])
    if image != "nan":
        if answer_type == "multiple_choice":
            return {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": hle_sys_prompt_mc
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": image
                            }
                        ]
                    }
                ],
            }
        else:
            return {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": hle_sys_prompt_ex
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": image
                            }
                        ]
                    }
                ],
            }
    else:
        if answer_type == "multiple_choice":
            return {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": hle_sys_prompt_mc
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                    }
                ],
            }
        else:
            return {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": hle_sys_prompt_ex
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                    }
                ],
            }