from simulation_core import (
    SimulationConfig, MealGenerator, EnvironmentManager,
    ModelTrainer, SimulationRunner, DataSaver, MetricsCalculator
)
import pandas as pd
from lime_explainer import Predictor, Explainer


# === Main Entry Point ===
def main():
    # Setup Config 
    config = SimulationConfig(model_type="PPO", patient_name="child#002")
    patient_params = config.get_patient_params()
    print(f"Patient {config.patient_name} | BW: {patient_params['bw']} kg")

    # Generate Meals
    meal_gen = MealGenerator(config)
    scenario, meals = meal_gen.create_meal_scenario(patient_params["bw"])
    meal_gen.print_meals(meals)

    # Manage Enviroments
    env_mgr = EnvironmentManager(config, scenario)
    env_mgr.register_environments()
    env, lowenv, innerenv, highenv = env_mgr.create_environments()

    # Train Models
    trainer = ModelTrainer(lowenv, innerenv, highenv, config)
    lowmodel, innermodel, highmodel = trainer.train_or_load_models(use_existing_models=False)

    # Run the simulation
    runner = SimulationRunner(env, lowmodel, innermodel, highmodel, config)
    frames, log_data = runner.run()

    # Create a DataFrame from the log data
    log_df = pd.DataFrame(log_data)

    # LIME Explanation
    if not log_df.empty:
        print("=" * 60)
        print("     Generating LIME Explanations for Actions")
        print("=" * 60)
        
        # 1. Create a predictor
        predictor = Predictor(lowmodel, innermodel, highmodel)

        # 2. Create an explainer
        feature_names = ['blood glucose', 'meal']
        training_data = log_df[feature_names].values
        
        explainer = Explainer(predictor, training_data, feature_names)

        # 3. Explain instances where an action was taken
        explanations_log = []
        action_indices = log_df[log_df['action'] > 0].index

        for i in action_indices:
            instance_to_explain = training_data[i]
            explanation = explainer.explain_instance(instance_to_explain, num_features=len(feature_names))
            
            log_entry = f"--- Explanation for Step at Time: {log_df.loc[i, 'time']} (Action: {log_df.loc[i, 'action']:.2f}) ---\n"
            log_entry += f"Model Prediction: {explanation.predicted_value:.2f}\n"
            log_entry += "Feature Contributions:\n"
            for feature, weight in explanation.as_list():
                log_entry += f"  - {feature}: {weight:.4f}\n"
            explanations_log.append(log_entry)

        # 4. Save the explanations to a log file
        if explanations_log:
            explanation_path = env_mgr.path_to_results / "lime_explanations_log.txt"
            with open(explanation_path, "w") as f:
                f.write("\n\n".join(explanations_log))
            print(f"LIME explanations log saved to: {explanation_path}")
        else:
            print("No actions were taken during the simulation, so no LIME explanations were generated.")

    # Save Result and metrics
    saver = DataSaver(env_mgr.path_to_results, config)
    saver.save_csv(log_data)
    saver.save_meals_to_csv(meals)
    saver.save_video(frames)
    saver.save_plot(log_data)

    metrics_calc = MetricsCalculator(env_mgr.path_to_results)
    metrics = metrics_calc.calculate(log_data)
    metrics_calc.save(metrics)

    env.close()


if __name__ == "__main__":
    main()
