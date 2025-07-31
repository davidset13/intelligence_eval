import requests

response = requests.post("http://127.0.0.1:3000/llm/general", json={"model": "anthropic/claude-sonnet-4", "hle": True, "mmlu_pro": True})

print(response.json())