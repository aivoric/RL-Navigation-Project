from helper import train_dqn
import pickle
import time
from os import path
import json
from unityagents import UnityEnvironment
    
# Create the environment
env = UnityEnvironment(file_name="Banana.app", no_graphics=True)

# Retrieve the experiment configurations:
with open('experiments.json') as f:
    experiments_data = json.load(f)["experiments"]

# Iterate through all the experiments:
for experiment in experiments_data:
    if experiment["skip"] == False:
        # Get expriment variables:
        experiment_name = experiment["experiment_name"]
        experiment_config = experiment["experiment_config"]
        
        num_of_episodes = experiment_config["num_of_episodes"]
        max_steps = experiment_config["max_steps"]
        learning_rate = experiment_config["learning_rate"]
        gamma = experiment_config["gamma"]
        epsilon_decay = experiment_config["epsilon_decay"]
        epsilon_min = experiment_config["epsilon_min"]
        
        # Inform user about the experiment:
        print("\n#############################\nSTARTING NEW EXPERIMENT: {} \n#############################\n".format(experiment_name))
        print("EXPERIMENT CONFIG:\n")
        print("Number of episodes: {}".format(num_of_episodes))
        print("Max Steps: {}".format(max_steps))
        print("Learning Rate: {}".format(learning_rate))
        print("Gamma: {}".format(gamma))
        print("Epsilon Decay: {}".format(epsilon_decay))
        print("Epsilon Minimum Value: {}".format(epsilon_min))
        print("")

        # Train the model:
        final_score, all_scores, model = train_dqn(env, num_of_episodes, max_steps, learning_rate, gamma, epsilon_decay, epsilon_min)
        print("\nFinal Score: {}".format(final_score))

        # Save the results and final model state dictionary:
        timestamp = int(time.time())
        results_dict = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'final_score': final_score,
            'all_scores': all_scores,
            'model': model
        }
        file_name = path.join('results', '{}_{}.pickle'.format(timestamp, experiment_name))
        with open(file_name, 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("\nResults saved to: {}.".format(file_name))

# Close the unity environment
env.close()

