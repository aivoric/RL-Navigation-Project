from helper import train_dqn
import pickle
import time
from os import path
import json
from unityagents import UnityEnvironment

with open('experiments.json') as f:
    experiments_data = json.load(f)["experiments"]
    
# Create the environment
env = UnityEnvironment(file_name="Banana.app", no_graphics=True)

for experiment in experiments_data:
    if experiment["skip"] == False:
        experiment_name = experiment["experiment_name"]
        experiment_config = experiment["experiment_config"]
        
        print("\n#############################\nSTARTING NEW EXPERIMENT: {} \n#############################\n".format(experiment_name))
        
        num_of_episodes = experiment_config["num_of_episodes"]
        max_steps = experiment_config["max_steps"]
        learning_rate = experiment_config["learning_rate"]
        gamma = experiment_config["gamma"]
        epsilon_decay = experiment_config["epsilon_decay"]
        epsilon_min = experiment_config["epsilon_min"]

        # train the DQN
        final_score, all_scores, model = train_dqn(env, num_of_episodes, max_steps, learning_rate, gamma, epsilon_decay, epsilon_min)

        timestamp = int(time.time())

        results_dict = {
            'timestamp': timestamp,
            'final_score': final_score,
            'all_scores': all_scores,
            'model': model
        }

        file_name = path.join('results', '{}_{}.pickle'.format(timestamp, experiment_name))

        with open(file_name, 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("\nResults saved to: {}.".format(file_name))

        # Print final score
        print("\nFinal Score: {}".format(final_score))

# Close the unity environment
env.close()

