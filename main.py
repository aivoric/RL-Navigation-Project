from helper import train_dqn
import pickle
import time
from os import path

num_of_episodes = 1000
max_steps = 1000
learning_rate = 0.0005 # Original value: 5e-4 or 0.0005
gamma = 0.99
epsilon_decay = 0.995
epsilon_min = 0.01

# train the DQN
final_score, all_scores, model = train_dqn(num_of_episodes, max_steps, learning_rate, gamma, epsilon_decay, epsilon_min)

timestamp = int(time.time())

results_dict = {
    'timestamp': timestamp,
    'final_score': final_score,
    'all_scores': all_scores,
    'model': model
}

file_name = path.join('results', '{}.pickle'.format(timestamp))

with open(file_name, 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Results saved to: {}.".format(file_name))

# Print final score
print("Final Score: {}".format(final_score))

# Test retrieving pickled data
with open(file_name, 'rb') as handle:
    saved_results = pickle.load(handle)
    # print("Best model retrieved: {}".format(saved_results['model']))
    
