import requests

response = requests.post("http://127.0.0.1:3000/llm/general", 
json={"agent_name": "test_agent", "agent_url": "http://127.0.0.1:8000/test_agent", "agent_params": 
    {"model": "meta-llama/llama-4-maverick", "prompt": "Unimportant", "image_enabled": True, "image": None, "output_type": None}, 
"prompt_param_name": "prompt", "image_param_name": "image", "hle": True, "mmlu_pro": True, "gpqa": True, "images_enabled": True})

print(response.json())

