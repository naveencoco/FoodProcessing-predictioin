import os
import requests
import numpy as np
from flask import Flask, request, render_template  # ✅ move this up
import joblib

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?id=1Ch_GV4I9HBKd67ujoznlExJfGRo0nYd7"
MODEL_PATH = "random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)
    print("Model downloaded.")

model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def predict():
    features = ["carbohydrates", "cholesterol", "energy", "fat", "fiber",
                "proteins", "salt", "saturated_fat", "sodium", "sugars"]
    prediction = None

    if request.method == "POST":
        input_data = [float(request.form[feat]) for feat in features]
        prediction = model.predict(np.array(input_data).reshape(1, -1))[0]

    return render_template("index.html", features=features, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

