from typing import Any


def create_gpqa_score_payload(model: str, question: str, correct_answer: str, response: str) -> dict[str, Any]:
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

                            [question with options]: {question}

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
    }
    
