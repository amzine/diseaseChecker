from flask import Flask,request, jsonify
import pickle
import joblib
import os

app = Flask(__name__)
models = {}
models_dir = "models"
for file_name in os.listdir(models_dir):
    model_path = os.path.join(models_dir, file_name)
    model_name = os.path.splitext(file_name)[0]

    if file_name.endswith(".pkl"):
        with open(model_path, "rb") as f:
            models[model_name] = pickle.load(f)
    elif file_name.endswith(".joblib"):
        models[model_name] = joblib.load(model_path)
condition_labels = {
    "asthma": {0: "no asthma", 1: "asthma"},
    "hypertension": {0: "no hypertension", 1: "hypertension"},
    "diabetes": {0: "no diabetes", 1: "diabetes"},
    "parkinson" :{0: "no parkinson", 1 :"parkinson"}
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data.get("model_name")

    if model_name not in models:
        return jsonify({"error": "invalide model name , choose from " + ', '.join(models.keys())}), 400
    model = models[model_name]
    labels = condition_labels.get(model_name, {0: "Class 0", 1: "Class 1"})  # Default labels if not specified

    features = data.get("input")
    if features is None:
        return jsonify({'error': 'No input data provided'}), 400

    # Ensure the input is in the correct 2D shape
    if isinstance(features[0], list):
        input_data = features
    else:
        input_data = [features]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None
    
    prediction_label = labels.get(prediction, "Unknown")

    # Format probabilities as strings with percentages if available
    if probability is not None:
        probability_str = {
            f"{labels[0]}": f"{probability[0] * 100:.2f}%",
            f"{labels[1]}": f"{probability[1] * 100:.2f}%"
        }
    else:
        probability_str = None
    result = {
        "model_used" : model_name,
        "model_prediction" : prediction_label,
        "model_probalilty" : probability_str
    }
    return jsonify(result)

    


# prinxt(names)
# with open("asthma_prediction_model.pkl", "rb") as f:
#     model = pickle.load(f)

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     input_data = np.array(data['input'])
#     if input_data.ndim == 1:
#         input_data = input_data.reshape(1, -1)
#     prediction = model.predict(input_data)
#     prediction = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
#     return jsonify({'predicion': prediction})
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)