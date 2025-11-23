import os
import pandas as pd
from pathlib import Path
from stable_baselines3 import A2C, TD3

from simulation_core import (
    SimulationConfig, MealGenerator, EnvironmentManager,
    ModelTrainer, SimulationRunner, DataSaver, MetricsCalculator, 
    clear_console
)
from lime_explainer import Predictor, Explainer

def main():
    clear_console()
    print("=" * 60)
    print("     Interactive Model Loader & Simulation Runner")
    print("=" * 60)

    model_type = input("Choose model type (A2C / TD3): ").strip().upper()
    if model_type not in {"A2C", "TD3"}:
        model_type = "A2C"

    use_existing_models = input("Do you want to load existing trained models? (y/n): ").strip().lower() == "y"

    config = SimulationConfig(model_type=model_type)
    patient_params = config.get_patient_params()
    print(f"Patient {config.patient_name} | BW: {patient_params['bw']} kg")

    meal_gen = MealGenerator(config)
    scenario, meals = meal_gen.create_meal_scenario(patient_params["bw"])
    meal_gen.print_meals(meals)

    env_mgr = EnvironmentManager(config, scenario)
    env_mgr.register_environments()
    env, lowenv, innerenv, highenv = env_mgr.create_environments()

    trainer = ModelTrainer(lowenv, innerenv, highenv, config)
    lowmodel, innermodel, highmodel = trainer.train_or_load_models(use_existing_models)

    runner = SimulationRunner(env, lowmodel, innermodel, highmodel, config)
    frames, log_data = runner.run()

    # Create a DataFrame from the log data
    log_df = pd.DataFrame(log_data)

    # LIME Explanation
    if not log_df.empty:
        print("=" * 60)
        print("     Generating LIME Explanation")
        print("=" * 60)
        
        # 1. Create a predictor
        predictor = Predictor(lowmodel, innermodel, highmodel)

        # 2. Create an explainer
        feature_names = ['blood glucose', 'meal']
        training_data = log_df[feature_names].values
        
        explainer = Explainer(predictor, training_data, feature_names)

        # 3. Explain an instance (e.g., the first row)
        instance_to_explain = training_data[0]
        explanation = explainer.explain_instance(instance_to_explain, num_features=len(feature_names))

        # 4. Save the explanation to an HTML file
        explanation_path = env_mgr.path_to_results / "lime_explanation.html"
        explanation.save_to_html(explanation_path)
        print(f"LIME explanation saved to: {explanation_path}")


    saver = DataSaver(env_mgr.path_to_results, config)
    saver.save_csv(log_data)
    saver.save_video(frames)
    saver.save_plot(log_data)

    metrics_calc = MetricsCalculator(env_mgr.path_to_results)
    metrics = metrics_calc.calculate(log_data)
    metrics_calc.save(metrics)

    env.close()


if __name__ == "__main__":
    main()
