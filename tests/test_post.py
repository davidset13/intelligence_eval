import requests

response = requests.post(
    "http://127.0.0.1:3000/llm/general", 
    json={
        "agent_name": "test_agent",
        "agent_url": "http://127.0.0.1:8000/test_agent",
        "agent_params": {
            "model": "google/gemini-2.5-pro",
            "prompt": "Unimportant",
            "image_enabled": False,
            "image": None,
            "output_type": None
        }, 
        "prompt_param_name": "prompt",
        "image_param_name": "image",
        "hle": True,
        "hle_categories": ["all"],
        "mmlu_pro": False, 
        "mmlu_pro_categories": ["all"],
        "gpqa": False,
        "livebench": False,
        "livebench_categories": ["all"],
        "images_enabled": False
    }
)

print(response.json())


