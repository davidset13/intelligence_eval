import requests

response = requests.post(
    "http://127.0.0.1:3000/llm/general", 
    json={
        "agent_name": "test_agent",
        "agent_url": "http://127.0.0.1:8000/test_agent",
        "agent_params": {
            "model": "google/gemini-flash-1.5-8b",
            "prompt": "Unimportant",
            "image_enabled": True,
            "image": None,
            "output_type": None
        }, 
        "prompt_param_name": "prompt",
        "image_param_name": "image",
        "hle": True, 
        "mmlu_pro": False, 
        "gpqa": False, 
        "images_enabled": True
    }
)

print(response.json())