import requests

response = requests.post("http://127.0.0.1:3000/llm/general", json={"model": "google/gemma-3n-e4b-it", "hle": True, "mmlu_pro": False})

print(response.json())
