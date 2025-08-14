from typing import Any


def code_extractor_payload(model: str, response: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"""
                            The prompt you are given is an output of any nature, but it is LLM generated output.
                            Specifically, this LLM generated output was asked to solve a particular coding question.
                            You do not have access to this question, but your job is to make the code possible to run. 
                            Since it is LLM generated code, it is neseted within a string or some other form of output.
                            As such, your task is to take the input (in form of a string) and produces the indices at which
                            a slice should occur. For instance, here is a sample output:

                            Hello, I am an LLM assistant, here is the answer to your coding question,
                            ```python
                            print('hello world')
                            ```

                            As we can see here, the code is print('hello world'), and we need to index the string
                            such that output[index1:index2] = "print('hello world')", allowing this code to be run.
                            
                            You will be given two distinct inputs to help with this task. The first input is the [raw output], this [raw output]
                            will help add context to what you need to extract. This will help you identify the starting and ending indices.
                            Remember: Regardless of the language that the code is provided in, it is being parsed in Python, which means
                            your first index should be inclusive and latter should be exclusive.

                            Alonsgide this raw output, you will be given an array (Python list) of tuples, called [char-idx array]. The list will be in this format
                            [(idx, char), (idx, char), ...]. Essentially, it gives the index of a character along with the character itself.
                            This should make it easier for you to properly select the indices.

                            Here are the inputs:

                            [raw input]: {response}

                            [char-idx array]: {[(i, idx) for i, idx in enumerate(response)]}

                            Now that you have identified the proper indices, you will output a JSON array with the following keys

                            open_idx: Integer index of the first character of the code. Remember it is inclusive

                            close_idx: Integer index of the last character of the code. Remember it is exclusive, so it should be the one after.
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
                        "open_idx": {
                            "type": "number",
                            "description": "Integer index of the first character of the code. Remember it is inclusive"
                        },
                        "close_idx": {
                            "type": "number",
                            "description": "Integer index of the last character of the code. Remember it is exclusive, so it should be the one after."
                        }
                    },
                    "required": ["open_idx", "close_idx"],
                    "additionalProperties": False
                }
            },
        },
    }
    
