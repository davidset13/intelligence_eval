import requests
import pandas as pd
import os

response = requests.post(
    "http://127.0.0.1:3000/llm/general", 
    json={
        "agent_name": "test_agent",
        "agent_url": "http://127.0.0.1:9000/",
        "agent_params": {
            "prompt": "Unimportant",
        }, 
        "prompt_param_name": "prompt",
        "image_param_name": "image",
        "hle": False,
        "hle_categories": ["all"],
        "mmlu_pro": True, 
        "mmlu_pro_categories": ["all"],
        "gpqa": False,
        "livebench": False,
        "livebench_categories": ["all"],
        "images_enabled": False
    }
)


def print_eval_results(response: requests.Response) -> None:
    print(f"Agent Name: {response.json()['agent_name']}")
    print("\n--------------------------------\n")
    print(f"HLE Accuracy: {response.json()['hle_accuracy']}")
    print(f"HLE CI: {response.json()['hle_ci']}")
    print("\n--------------------------------\n")
    for key, value in response.json()["hle_categories"].items():
        print(f"{key}: {value[0]}")
        print(f"CI: {value[1]}")
    print("\n--------------------------------\n")
    print(f"MMLU Pro Accuracy: {response.json()['mmlu_pro_accuracy']}")
    print(f"MMLU Pro CI: {response.json()['mmlu_pro_ci']}")
    print(f"MMLU Pro Categories: {response.json()['mmlu_pro_categories']}")


def print_total_results(model: str) -> None:
    if model == "GPT-4o":
        dataset = pd.read_csv(os.path.join(os.getcwd(),"intelligence_eval", "agent_ans", "chatgpt-gpt4o.csv"), encoding="utf-8")
    elif model == "GPT-5-FAST":
        dataset = pd.read_csv(os.path.join(os.getcwd(), "intelligence_eval", "agent_ans", "chatgpt-gpt5-fast.csv"), encoding="utf-8")
    elif model == "GPT-5-THINK":
        dataset = pd.read_csv(os.path.join(os.getcwd(), "intelligence_eval", "agent_ans", "chatgpt-gpt5-think.csv"), encoding="utf-8")
    else:
        raise ValueError("GPT Model Not Supported")
    
    accuracy = dataset['marked_correct'].sum() / len(dataset)
    print("total length", len(dataset))
    print("correct", dataset['marked_correct'].sum())
    print(f"{model} Accuracy: {accuracy}")

print_total_results("GPT-5-FAST")