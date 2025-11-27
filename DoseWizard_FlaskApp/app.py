import sys
import os
# Add the parent directory to the Python path to allow for package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from pathlib import Path
from CoreLogic.simulation_core import SimulationConfig, EnvironmentManager
from CoreLogic.lime_explainer import Predictor
from stable_baselines3 import A2C, PPO, TD3

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, 'WorkingModels')
MODEL_CACHE = {}

def load_models(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    model_path = Path(MODELS_DIR) / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    config = SimulationConfig(model_type="PPO", patient_name="child#002") # patient_name is not used for prediction, but required by SimulationConfig
    env_mgr = EnvironmentManager(config, None)  # Pass None for meal_scenario
    env_mgr.register_environments()
    env, lowenv, innerenv, highenv = env_mgr.create_environments()

    models = {}
    for model_short_name, env_instance in [("lowmodel", lowenv), ("innermodel", innerenv), ("highmodel", highenv)]:
        # Construct the full path to the model file
        full_model_path = model_path / f"{model_short_name}.zip"
        if not full_model_path.exists():
            raise FileNotFoundError(f"Model file not found: {full_model_path}")
        
        # Determine the model class based on the model type in the config
        if config.model_type == "A2C":
            model_class = A2C
        elif config.model_type == "PPO":
            model_class = PPO
        elif config.model_type == "TD3":
            model_class = TD3
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        models[model_short_name] = model_class.load(str(full_model_path), env=env_instance)

    predictor = Predictor(models["lowmodel"], models["innermodel"], models["highmodel"])
    MODEL_CACHE[model_name] = predictor
    return predictor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def get_models():
    if not os.path.exists(MODELS_DIR):
        return jsonify([])
    models = [name for name in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, name))]
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    blood_glucose = float(data.get('blood_glucose') or 0)
    meal = float(data.get('meal') or 0)
    model_name = data.get('model_name')

    if not model_name:
        return jsonify({'error': 'Please select a model.'}), 400

    try:
        predictor = load_models(model_name)
        # The model expects a 2D array of shape (1, 2)
        obs = np.array([[blood_glucose, meal]])
        prediction = predictor.predict(obs)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': f"An error occurred on the server: {e}"}), 500

if __name__ == '__main__':
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created '{MODELS_DIR}' directory. Please place your models in this directory.")
    app.run(debug=True)
