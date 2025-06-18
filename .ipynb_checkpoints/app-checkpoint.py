{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f42a8c6-c94e-4063-80fc-d5ab9ef1ab7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# app.py\n",
    "\n",
    "from flask import Flask, request, jsonify\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load the model\n",
    "model = joblib.load(\"random_forest_model.pkl\")\n",
    "\n",
    "# Initialize Flask app\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return jsonify({\"message\": \"Welcome to the Food Processing Prediction API using Flask!\"})\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    data = request.get_json()\n",
    "\n",
    "    # Extract features\n",
    "    features = [\n",
    "        data[\"carbohydrates\"],\n",
    "        data[\"cholesterol\"],\n",
    "        data[\"energy\"],\n",
    "        data[\"fat\"],\n",
    "        data[\"fiber\"],\n",
    "        data[\"proteins\"],\n",
    "        data[\"salt\"],\n",
    "        data[\"saturated_fat\"],\n",
    "        data[\"sodium\"],\n",
    "        data[\"sugars\"]\n",
    "    ]\n",
    "\n",
    "    prediction = model.predict(np.array(features).reshape(1, -1))\n",
    "\n",
    "    return jsonify({\"predicted_nova_group\": float(prediction[0])})\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
