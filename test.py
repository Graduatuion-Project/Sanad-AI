import requests

url = ""
data = {"sentence": "انا وماما وبابا"}

response = requests.post(url, json=data)
print(response.json())