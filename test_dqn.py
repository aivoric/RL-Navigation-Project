import os
import pickle

result_files = os.listdir("results_archive")
print(result_files)


for file in result_files:
    print("\n#################\nANALYSING RESULTS FOR: {}\n#################".format(file))
    
    file_location = os.path.join('results', file)
        
    with open(file_location, 'rb') as f:
        saved_results = pickle.load(f)
        
    print("\nFINAL SCORE: {}\n".format(saved_results['final_score']))
    
    scores = saved_results['all_scores']
    print(scores)
    