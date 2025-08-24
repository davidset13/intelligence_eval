# Intelligence Server

* There are thousands of agents on the market today, and their intelligence ability is greatly determined by its prompting, tool usage, and RAG systems.
* For instance, community testers have determined that o4-mini scores twice as well on traditional intelligence benchmarks (HLE being most notable) with deep research pipelines enabled.
* This open-source project is my agentic intellect evaluation, feel free to plug your own agent into it and test yourself!
* More benchmarks will be added to the future, currently this server only supports intelligence benchmarks, but future versions will have more benchmarks as well as benchmarks that venture beyond just intellect (such as GAIA).

## Currently Supported Bechmarks:

[Humanity's Last Exam (HLE)](https://agi.safe.ai/)

[Massive Multitask Language Understanding Pro (MMLU-Pro)](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)

[Graduate-Level Google-Proof Q&A (GPQA)](https://huggingface.co/datasets/Idavidrein/gpqa)
    - Note: This server uses the diamond subset of the GPQA, or GPQA-Diamond

[LiveBench](https://livebench.ai/#/)

## IMPORTANT

* Make sure to follow proper protocol when using these datasets. Do not post them online or else they will become training data for future LLMs. Adhere to any HuggingFace protocols and terms of service when using.

* This project is designed to allow people to test agents that they have desinged (essentially, that they have an endpoint for). If you would like to use this repo to test a released proprietary agent (ChatGPT, Claude Desktop, etc.), make sure to follow proper guidelines.

* If you do want to test proprietary agents that you do not own, and intend to use something like Puppeteer to create an endpoint for the data, make sure to abide by any licensing.

## Setup
```bash
cd intelligence_eval
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows
curl -LsSf https://astral.sh/uv/install.sh | sh # MacOS/Linux
```

* Next, we configure the virtual environment and package locks

```bash
uv venv
.venv/Scripts/activate # Windows
source .venv/bin/activate # MacOS/Linux
uv sync

python src/get_datasets.py # Windows
python3 src/get_datasets.py # MacOS/Linux

python src/intelligence_srv.py # Windows
python3 src/intelligence_srv.py # MacOS/Linux
```

## Environment Variables (All Machines)
```bash
# Create .private.env file for environment variables
cp template.private.env
```

* Create a new file called .private.env and insert your API keys
* This project supports openrouter clients, but this can easily be modified to support other providers

## Testing

* A simple testing_agent is created under `test_agent_srv.py` and a sample call exists in `test_post.py`

```bash
python tests/test_agent_srv.py # Windows
python3 tests/test_agent_srv.py # MacOS/Linux
```

## Example Payload

**WARNING: THIS CALL IS SET TO A MAX DEFAULT ERROR OF 0.04, MEANING THAT HUNDREDS OF AGENT CALLS WILL BE MADE. THIS WILL BE CHARGED TO YOUR AGENT ON TOP OF THE EVALUATION COSTS. RAISE MARGIN OF ERROR (INSTRUCTIONS BELOW) IF COST IS A CONCERN**

```python
import requests # No need for asynchronous calls, that is handled internally.

response = requests.post(
    "http://127.0.0.1:3000/llm/general", 
    json={
        "agent_name": "test_agent",
        "agent_url": "http://127.0.0.1:8000/test_agent",
        "agent_params": {
            "model": "google/gemini-flash-1.5-8b",
            "prompt": "Unimportant", # The agent manually inserts prompts, whatever you include here is arbitrary
            "image_enabled": True,
            "image": None,
            "output_type": None
        }, 
        "prompt_param_name": "prompt",
        "image_param_name": "image",
        "hle": True,
        "hle_categories": ["math", "computer_science"],
        "mmlu_pro": True, 
        "mmlu_pro_categories": ["math", "physics"],
        "gpqa": True,
        "livebench": True,
        "livebench_categories": ["all"]
        "images_enabled": True
    }
)
```

## Payload Params Explanation

**Pydantic class available in `payloads.py`**

* URL:
    - `http://127.0.0.1:3000/llm/general` is the url of the intelligence server. This is the default for this file, feel free to change later

* JSON:
    - `agent_name`: Optional parameter for agent name.
        - Type: Any
    
    - `agent_url`: The url you call your agent at, see above for default value
        - Type: String

    - `agent_params`: The JSON that would be used to call your own agent. Make sure the prompt is inside of this payload. This is necessary to make sure the intelligence evaluation can properly run.
        - Type: Dict[Any, Any]

    - `prompt_param_name`: Set to "prompt" by default. Make sure to change this to whatever parameter name your agent uses to insert the prompt.
        - Type: Any
    
    - `images_enabled`: Some evaluations (HLE only currently), take JPG and PNG images as an input. If your agent supports agents, enable this feature. If your agent is text only, it will skip image questions.
        - Type: Boolean

    - `image_param_name`: Set to "image" by default. Same as prompt_param_name, make sure to change this to whatever parameter your agent uses for image insertion.

    - `hle`: If True, the intelligence server will create an evaluation of Humanity's Last Exam (HLE).
        - Type: Boolean

    - `hle_categories`: Optional parameter to select categories. If not selected, will evaluate all. Otherwise, choose any subset of valid categories, and only these will be evaluated.
        - Type: List[str]
            - Valid Categories: ("math", "biology", "computer_science", "other", "physics", "humanities", "chemistry", "engineering")
    
    - `mmlu_pro`: If True, the intelligence server will create an evaluation of Massive Multitask Language Understadning Pro (MMLU-Pro).
        - Type: Boolean

    - `mmlu_pro_categories`: Optional parameter to select categories. If not selected, will evaluate all. Otherwise, choose any subset of valid categories, and only these will be evaluated.
        - Type: List[str]
            - Valid Categories: ("math", "physics", "chemistry", "law", "engineering", "other", "economics", "health", "psychology", "business", "biology", "philosophy", "computer science", "history")
    
    - `gpqa`: If True, the intelligence server will create an evaluation of GPQA Diamond.
        - Type: Boolean
     
    - `livebench`: If True, the intelligence server will create an evaluation of LiveBench.
        - Type: Boolean

    - `livebench_categories`: Optional parameter to select categories. If not seleected, will evaluate all. Otherwise choose any subset of valid categories, and only these will be evaluated.
        - Type: List[str]
            - Valid Categories: ("reasoning", "data_analysis", "instruction_following", "math", "language")


## Changing Margin of Error

* Open `intelligence_server.py`
* Change the parameter `eps` to something other than 0.04 (default).
* The greater the margin of error, the less accurate the results, but fewer questions are used as input

## Interpretation of Results

* The result will come as a JSON that returns the name of your agent, if provided, as well as results.
* It gives the raw accuracy scores on whichever benchmarks you enabled.
* It also gives the 95% confidence intervals of each evaluation.
