import requests

response = requests.post("http://127.0.0.1:3000/llm/general", json={"model": "google/gemma-3-4b-it"})

