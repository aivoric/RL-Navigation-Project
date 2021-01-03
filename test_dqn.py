from helper import run_trained_dqn_agent
from unityagents import UnityEnvironment
import os

# Create the environment
env = UnityEnvironment(file_name="Banana.app", seed=51)

# Get the location of the tained model
trained_model_location = os.path.join('models', '1609615254_Various Changes_model')

# Pass the environment and trained model into a function which handles everything
run_trained_dqn_agent(env, trained_model_location)