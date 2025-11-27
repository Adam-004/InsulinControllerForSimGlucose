import lime
import lime.lime_tabular
import numpy as np

class Predictor:
    def __init__(self, low_model, inner_model, high_model):
        self.low_model = low_model
        self.inner_model = inner_model
        self.high_model = high_model

    def predict(self, x):
        """
        Predicts the action for a given set of observations.
        x is a numpy array of shape (n_samples, n_features)
        """
        predictions = []
        for obs in x:
            value = obs[0]
            # The model expects a 2D array of shape (1, 1)
            obs_array = np.array([[value]])
            if value > 130:
                action, _ = self.high_model.predict(obs_array, deterministic=True)
            elif 70 < value <= 130:
                action, _ = self.inner_model.predict(obs_array, deterministic=True)
            else:
                action, _ = self.low_model.predict(obs_array, deterministic=True)
            predictions.append(action[0])
        return np.array(predictions)

class Explainer:
    def __init__(self, predictor, training_data, feature_names):
        self.predictor = predictor
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=['action'],
            verbose=True,
            mode='regression'
        )

    def explain_instance(self, data_row, num_features=1):
        """
        Explains a single instance.
        """
        return self.explainer.explain_instance(
            data_row,
            self.predictor.predict,
            num_features=num_features
        )
