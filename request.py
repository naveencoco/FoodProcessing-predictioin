# request.py

import requests

url = "http://127.0.0.1:5000/predict"

# Example input data
data = {
    "carbohydrates": 13.33,
    "cholesterol": 0.02,
    "energy": 1117.0,
    "fat": 20.0,
    "fiber": 0.0,
    "proteins": 10.0,
    "salt": 3.6895,
    "saturated_fat": 8.33,
    "sodium": 1.4758,
    "sugars": 10.0
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("✅ Prediction:", response.json())
else:
    print("❌ Failed to get response:", response.status_code)
