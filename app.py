from flask import Flask, request, jsonify
from classifier import JobDescriptionClassifier

app = Flask(__name__)
classifier = JobDescriptionClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    description = data['description']
    prediction = classifier.predict(description)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
