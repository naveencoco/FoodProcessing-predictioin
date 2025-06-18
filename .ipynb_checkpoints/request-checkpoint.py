{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "beb9cf39-cb21-4376-bd62-ae3d5bf3afcb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# rest.py\n",
    "\n",
    "from flask import Flask, request, render_template_string\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "model = joblib.load(\"random_forest_model.pkl\")\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "HTML_TEMPLATE = '''\n",
    "<!doctype html>\n",
    "<title>Food Processing Prediction</title>\n",
    "<h2>Enter Nutritional Information</h2>\n",
    "<form method=\"POST\" action=\"/predict\">\n",
    "  {% for feature in features %}\n",
    "    <label>{{ feature }}:</label>\n",
    "    <input name=\"{{ feature }}\" required type=\"number\" step=\"any\"><br><br>\n",
    "  {% endfor %}\n",
    "  <input type=\"submit\" value=\"Predict NOVA Group\">\n",
    "</form>\n",
    "\n",
    "{% if prediction is not none %}\n",
    "  <h3>Predicted NOVA Group: {{ prediction }}</h3>\n",
    "{% endif %}\n",
    "'''\n",
    "\n",
    "@app.route(\"/\", methods=[\"GET\", \"POST\"])\n",
    "def predict():\n",
    "    features = [\"carbohydrates\", \"cholesterol\", \"energy\", \"fat\", \"fiber\",\n",
    "                \"proteins\", \"salt\", \"saturated_fat\", \"sodium\", \"sugars\"]\n",
    "    prediction = None\n",
    "\n",
    "    if request.method == \"POST\":\n",
    "        try:\n",
    "            input_data = [float(request.form[feat]) for feat in features]\n",
    "            prediction = model.predict(np.array(input_data).reshape(1, -1))[0]\n",
    "        except Exception as e:\n",
    "            prediction = f\"Error: {e}\"\n",
    "\n",
    "    return render_template_string(HTML_TEMPLATE, features=features, prediction=prediction)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
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
