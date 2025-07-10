import requests

response = requests.post("http://127.0.0.1:3000/llm/general", json={"model": "openai/gpt-4o"})
print(response.json())

