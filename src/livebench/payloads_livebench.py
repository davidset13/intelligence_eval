from typing import Any


def create_livebench_score_payload(model: str, category: str, response: str, question: str | None = None, correct_answer: str | None = None, input_text: str | None = None, task: str | None = None) -> dict[str, Any]:
    if category != "INS":
        if not question or not correct_answer:
            raise ValueError("question and correct_answer must be provided for reasoning and data analysis")
        return create_livebench_score_payload_gt(model, question, correct_answer, response)
    else:
        if not task or not input_text:
            raise ValueError("task and input_text must be provided for instruction following")
        return create_livebench_score_payload_if(model, task, input_text, response)


def create_livebench_score_payload_gt(model: str, question: str, correct_answer: str, response: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"""
                            Judge whether the following [response] to [question] is correct or not
                            based on the precise and unambiguous [correct_answer] below.

                            [question]: {question}

                            [response]: {response}

                            Your judgement must be in the format and criteria specified below:

                            [correct_answer]: {correct_answer}

                            You will output a JSON with this key:

                            "correct": Answer 'yes' if extracted_final_answer matches the [correct_answer] given
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


def create_livebench_score_payload_if(model: str, task: str, input_text: str, response: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"""
                            Your task is to judge how well the following [response] follows the [task] and [input_text]
                            below.

                            [task]: {task}

                            [input text]: {input_text}

                            Another LLM was asked to follow the [task] on the [input_text] and the response is:

                            [response]: {response}

                            Your task is to output a JSON that will be used to score the response. There will be two output keys
                            and here are the criteria for each one.

                            "instructions_followed": Answer 'yes' if the response successfully followed every instruction
                            in the [task] on the [input_text]. If it missed any instruction, answer 'no'.

                            "correctness": This component will look at each individual instruction in the [task] and check whether 
                            the response followed it correctly. If it did not follow it correctly, answer 'no'.

                            As such, the "instructions_followed" key checks if the response followed the [task] correctly. Meanwhile,
                            the "correctness" key checks if the response followed the [task] correctly.
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
                        "instructions_followed": {
                            "type": "string",
                            "enum": ["yes", "no"],
                            "description": "Answer 'yes' if the response followed the [task]. Answer 'no' otherwise."
                        },
                        "correctness": {
                            "type": "string",
                            "enum": ["yes", "no"],
                            "description": "Answer 'yes' if the response correctly executed the [task] on the [input_text]. Answer 'no' otherwise."
                        }
                    },
                    "required": ["instructions_followed", "correctness"],
                    "additionalProperties": False
                }
            },
        },
    }