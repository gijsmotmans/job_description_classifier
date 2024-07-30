from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load your machine learning model
model = joblib.load("your_model.pkl")


# Define a route for your API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Assuming your input data is in JSON format and contains features for prediction
    features = data["features"]

    # Convert features to numpy array
    features_array = np.array(features).reshape(1, -1)  # Assuming it's a single sample

    # Make prediction using your model
    prediction = model.predict(features_array)

    # Return the prediction as JSON response
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)