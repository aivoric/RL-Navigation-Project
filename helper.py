import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

from agent import Agent, TrainedAgent

def train_dqn(num_of_episodes, max_steps, learning_rate, gamma, epsilon_decay, epsilon_min):
    """
    Training function.
    """
    
    scores = []
    scores_window = deque(maxlen=100)
    best_score = -100
    
    # Create the environment
    env = UnityEnvironment(file_name="Banana.app", no_graphics=True)

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    # Get number of actions and state size
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    # Create the agent
    agent = Agent(state_size, action_size, learning_rate, gamma, epsilon_decay, epsilon_min)

    for episode in range(1, num_of_episodes+1):
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]
        episode_score = 0
        for step in range(max_steps):
            action = agent.act(state)
            
            # Move the environment to the next state with the appropriate action
            env_info = env.step(action)[brain_name]             
            
            # Get the next state, reward, and whether the state is done
            next_state = env_info.vector_observations[0]        
            reward = env_info.rewards[0]                        
            done = env_info.local_done[0]
            
            # Update the agent with the information:
            agent.update(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Update the score:
            episode_score += reward
            if done:
                break
                
        if episode % 10 == 0:
            print("Episode: {} of {}. Score: {:.2f}. Epsilon: {:.2f}".format(episode, num_of_episodes, np.mean(scores_window), agent.epsilon))
            if np.mean(scores_window) > best_score:
                best_score = np.mean(scores_window)
                file_name = agent.save_agent(episode)
                print("Saved a new agent: {}".format(file_name))
                
        # Update epsilon
        agent.update_epsilon()
            
        # Update the scores for this episode
        scores_window.append(episode_score)   
        scores.append(episode_score)    
    
    # Get the latest model
    best_model = agent.get_best_model()
    
    return np.mean(scores_window), scores, best_model

def test_dqn(state_dict):
    # Create the environment
    env = UnityEnvironment(file_name="Banana.app")

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset the environment
    env_info = env.reset()[brain_name]
    
    # Get number of actions and state size
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    score = 0
    
    # Create the agent
    trained_agent = TrainedAgent(state_size, action_size, state_dict)
    
    while True:
        action = trained_agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            print("Agent Finished. Total score: {}".format(score))
            break
        
    
def draw_chart(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()