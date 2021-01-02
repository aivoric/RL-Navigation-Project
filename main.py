from helper import train_dqn
import pickle
import time
from os import path
import json
from unityagents import UnityEnvironment
import torch
    
# Create the environment
env = UnityEnvironment(file_name="Banana.app", no_graphics=True)

# Retrieve the experiment configurations:
with open('experiments.json') as f:
    experiments_data = json.load(f)["experiments"]

# Iterate through all the experiments:
for experiment in experiments_data:
    if experiment["skip"] == False:
        # Get experiment variables:
        experiment_name = experiment["experiment_name"]
        experiment_config = experiment["experiment_config"]
        num_of_episodes = 20 # experiment_config["num_of_episodes"]
        max_steps = 1000 # experiment_config["max_steps"]
        
        # Main variables:
        learning_rate = experiment_config["learning_rate"]
        gamma = experiment_config["gamma"]
        epsilon_decay = experiment_config["epsilon_decay"]
        epsilon_min = experiment_config["epsilon_min"]
        tau = experiment_config["tau"]
        
        # Model variables:
        model_fc1_units = experiment_config["model_fc1_units"]
        model_fc2_units = experiment_config["model_fc2_units"]
        model_fc3_units = experiment_config["model_fc3_units"]
        model_dropout = experiment_config["model_dropout"]
        model_starting_weights = experiment_config["model_starting_weights"]
        model_batch_norm = experiment_config["model_batch_norm"]
        
        # Inform user about the experiment:
        print("\n#############################\nSTARTING NEW EXPERIMENT: {} \n#############################\n".format(experiment_name))
        print("EXPERIMENT CONFIG:\n")
        print("Number of episodes: {}".format(num_of_episodes))
        print("Max Steps: {}".format(max_steps))
        print("Learning Rate: {}".format(learning_rate))
        print("Gamma: {}".format(gamma))
        print("Epsilon Decay: {}".format(epsilon_decay))
        print("Epsilon Minimum: {}".format(epsilon_min))
        print("TAU: {}".format(tau))
        print("Model Fully Connected Layer 1 Size: {}".format(model_fc1_units))
        print("Model Fully Connected Layer 2 Size: {}".format(model_fc2_units))
        print("Model Fully Connected Layer 3 Size: {} (0 = Layer not used)".format(model_fc3_units))
        print("Model Uses a 30% Dropout Probability: {}".format(model_dropout))
        print("")

        # Train the model:
        best_score, all_scores, model_state_dict = train_dqn(env, num_of_episodes, max_steps, learning_rate, gamma, epsilon_decay, epsilon_min,
                                                   model_fc1_units, model_fc2_units, model_fc3_units, model_starting_weights, model_dropout, model_batch_norm,
                                                   tau)
        print("\nFinal Score: {}".format(best_score))

        # Save the results and final model state dictionary:
        timestamp = int(time.time())
        
        results_dict = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'best_score': best_score,
            'all_scores': all_scores
        }
        
        results_file_name = path.join('results', '{}_{}_results'.format(timestamp, experiment_name))
        model_file_name = path.join('models', '{}_{}_model'.format(timestamp, experiment_name))
        
        with open(results_file_name, 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("\nResults saved to: {}.".format(results_file_name))
            
        torch.save(model_state_dict, model_file_name)
        print("\nModel saved to: {}.".format(model_file_name))

# Close the unity environment
env.close()

