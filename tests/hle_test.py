import requests

response = requests.post("http://127.0.0.1:3000/llm/general", json={"model": "bytedance/ui-tars-1.5-7b", "hle": True, "mmlu_pro": True})

print(response.json())