from helper import test_dqn
import pickle

# Set the 
file_with_results = 'results/1609345659.pickle'

# Retrieving pickled data
with open(file_with_results, 'rb') as handle:
    saved_results = pickle.load(handle)
    print("\n\n###### SAVED RESULTS: {}\n\n".format(saved_results['model']))
    print("Retrieved model which achieved this score: {}".format(saved_results['final_score']))

# Test the model
test_dqn(saved_results['model'])