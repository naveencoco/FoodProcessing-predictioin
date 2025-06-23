from flask import Flask, render_template, request
import joblib
import numpy as np
import os  # Added to fetch environment port

# Create the Flask app and explicitly set static/template folders
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Nutritional features used in the model
features = ['carbohydrates', 'cholesterol', 'energy', 'fat', 'fiber', 'minerals',
            'proteins', 'salt_sodium', 'saturated_fat', 'vitamins']

feature_labels = {
    'carbohydrates': 'Carbohydrates (g)',
    'cholesterol': 'Cholesterol (mg)',
    'energy': 'Energy (kcal)',
    'fat': 'Fat (g)',
    'fiber': 'Fiber (g)',
    'minerals': 'Minerals (g)',
    'proteins': 'Proteins (g)',
    'salt_sodium': 'Salt/Sodium (mg)',
    'saturated_fat': 'Saturated Fat (g)',
    'vitamins': 'Vitamins (mg)'
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            input_data = [float(request.form[feature]) for feature in features]
            prediction = model.predict([input_data])[0]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", features=features, prediction=prediction, labels=feature_labels)


if __name__ == "__main__":
    # Use Render's dynamic port
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

