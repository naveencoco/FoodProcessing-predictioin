import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

import numpy as np
import gdown
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

MODEL_ID = "1Ch_GV4I9HBKd67ujoznlExJfGRo0nYd7"
MODEL_PATH = "random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model using gdown...")
    gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
    print("Model downloaded.")

model = joblib.load(MODEL_PATH)

