from typing import Any
import pandas as pd
import os

with open(os.path.join(os.getcwd(), "prompts", "mmlu_pro", "MMLU_PRO_5SHOT.md"), "r", encoding="utf-8") as f:
    mmlu_pro_5shot = f.read()


def create_mmlu_pro_init_payload(model: str, row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
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
                **config
            }
        

def create_mmlu_pro_score_payload(model: str, row: pd.Series, response: str, config: dict[str, Any]) -> dict[str, Any]:
    question = str(row["question"])
    correct_answer = str(row["answer"])
    options = str(row["options"])
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return your response as as JSON object, with keys 'extracted_final_answer', 'reasoning', 'correct', and 'confidence'."
            },
            {
                "role": "user",
                "content": f"""
                            Judge whether the following [response] to [question] is correct or not
                            based on the precise and unambiguous [correct_answer] below.

                            [question]: {question}

                            [options]: {options}
                            
                            [response]: {response}

                            Your judgement must be in the format and criteria specified below:

                            [correct_answer]: {correct_answer}

                            correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given
                            above, or is within a small margin of error for the numerical problems. Answer 'no'
                            otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the 
                            extracted answer is incorrect.
                            """
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "score_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "correct": {
                            "type": "string",
                            "enum": ["yes", "no"],
                            "description": "Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for the numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect."
                        },
                    },
                    "required": ["correct"],
                    "additionalProperties": False
                }
            },
        },
        **config
    }
    
