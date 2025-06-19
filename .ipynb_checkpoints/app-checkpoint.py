import os
import numpy as np
import gdown
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

MODEL_ID = "1Ch_GV4I9HBKd67ujoznlExJfGRo0nYd7"
MODEL_PATH = "random_forest_model.pkl"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model using gdown...")
    gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
    print("Model downloaded.")

# Load the trained model
model = joblib.load(MODEL_PATH)

# 🆕 Updated feature names
features = [
    "carbohydrates_(g)",
    "cholesterol_(mg)",
    "energy_(kcal)",
    "fat_(g)",
    "fiber_(g)",
    "minerals_(mg)",
    "proteins_(g)",
    "salt/sodium_(mg)",
    "saturated-fat_(g)",
    "vitamins_(mg)"
]

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        input_data = [float(request.form[feat]) for feat in features]
        prediction = model.predict(np.array(input_data).reshape(1, -1))[0]

    return render_template("index.html", features=features, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



