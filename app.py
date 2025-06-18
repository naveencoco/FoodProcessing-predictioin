from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("random_forest_model.pkl")

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

