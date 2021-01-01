import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from network import QNetwork
from memory import MemoryBuffer

class Agent():
    """
    The agent is responsible for interacting and learning from the environment.
    """
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon_decay, epsilon_min, 
                 model_fc1_units, model_fc2_units, model_fc3_units, model_starting_weights, model_dropout, model_batch_norm,
                 tau=1e-3, buffer_size=int(1e5), batch_size=64, update_every=4):
        
        # Set the device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set the action and space sizes
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialise the Q Networks:
        self.qnetwork_main = QNetwork(state_size, action_size, model_fc1_units, model_fc2_units, 
                                      model_fc3_units, model_starting_weights, model_dropout, model_batch_norm).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, model_fc1_units, model_fc2_units, 
                                      model_fc3_units, model_starting_weights, model_dropout, model_batch_norm).to(self.device)

        # Initialise the Optimizer:
        self.optimizer = optim.Adam(self.qnetwork_main.parameters(), lr = learning_rate)
        
        # Initialise the memory
        self.memory = MemoryBuffer(action_size, buffer_size, batch_size, self.device)
        
        # Initialise other variables:
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        
        # Set gamma and learning rate
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Set epsilon
        self.epsilon = 1
        self.epsilon_decay =  epsilon_decay
        self.epsilon_min = epsilon_min
                
        # Initialise the time step (this will be used when learning every UPDATE_EVERY step)
        self.time_step = 0
    
    def update_epsilon(self):
        """
        Decay the epsilon value, or chose the minimum one.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent with the newly acquired environment information.
        Check whether it is time to learn, if so, then learn.
        """
        self.memory.save_experience(state, action, reward, next_state, done)
        self.time_step += 1
        
        if((self.time_step % self.update_every == 0) and (len(self.memory) > self.batch_size)):
            self.learn()
    
    def learn(self):
        """
        Acquire a sample from the memory buffer.
        Calculate loss and backpropagate.
        """
        # Get an experience sample:
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_main(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update the target model
        self.update_target_model(self.qnetwork_main, self.qnetwork_target) 
    
    def update_target_model(self, main_model, target_model):
        """
        Soft update model parameters.
        
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            main_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, main_param in zip(target_model.parameters(), main_model.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

    def act(self, state):
        """
        Return actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Set evaluation mode
        self.qnetwork_main.eval()
        
        # Get the action values:
        with torch.no_grad():
            action_values = self.qnetwork_main(state)
            
        # Set training mode
        self.qnetwork_main.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def save_agent(self, episode):
        file_name = 'agents/agent-ep{}.pth'.format(episode)
        torch.save(self.qnetwork_main.state_dict(), file_name)
        return file_name
    
    def get_last_model(self):
        return self.qnetwork_main.state_dict()


class TrainedAgent():
    def __init__(self, state_size, action_size, state_dict):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        
        #state_dict = torch.load(state_dict)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def act(self, state):
        """
        Return actions for given state as per current policy.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
    
        # Get the action values:
        with torch.no_grad():
            action_values = self.model(state)
            
        return np.argmax(action_values.cpu().data.numpy())
        
        
        
        
    