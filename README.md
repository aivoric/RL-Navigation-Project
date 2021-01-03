## Introduction

This is a reinforcement learning project which teaches an agent to navigate on its own a room full of bananas and collect only yellow bananas while avoiding the blue.

For a quick preview of the final trained agent:
https://youtu.be/Og5-pX0pGjM

Enjoy the read below for more details!

## Environment Introduction

The environment is a 3D space where an agent walks around a square world and collects bananas.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

The state space consists of 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction

The action space consist of 4 discrete actions:
- 0 - move forward
- 1 - move backward
- 2 - turn left
- 3 - turn right 

## How to Install the Project

To setup the environment on your machine:
1. Install Python 3.6+
2. Clone this repository:
        git clone https://github.com/aivoric/RL-Navigation-Project.git
3. Create a virtual python environment:
        python3 -m venv venv
4. Activate it with:
        source venv/bin/activate
5. Install all the dependecies:
        pip install -r requirements.txt
6. Download and install Anaconda so that you can run a Jupyter notebook from:
        http://anaconda.com/

## Overview of Project Files

The project is broken down into the following core python files:
- main.py
- helper.py
- agent.py
- network.py
- memory.py
- test_dqn.py
- experiments.json

The following 2 folders:
- /models
- /results

And the following jupyter notebook:
- Navigation Results.ipynb

All the other files are used by the unity environment which allow you to run the environment. Most of the files are based on version 0.4 of ml-agents which is from July 2018 so it is quite outdated. For reference, a more modern ml-agents can be downloaded from: 
https://github.com/Unity-Technologies/ml-agents 

## Navigating the Project

The project contains a custom setup experiments framework which allows to launch a long training job so that many agents can be trained across a long time (hours / days), and then results can be compared.

This framework relies on the **experiments.json** file which contains all the hyperparameters for training the agents as well as some custom experiments, e.g. changing network architecture, or using batch normalisation.

The experiments.json file is consumed by the main.py file which reads all the experiments you want to run, and their respective hyperparameters.

Main.py also instatiates the Unity environment.

Main.py file imports the **train_dqn()** function from helper.py which receives the environment and all the hyperparameters for the training job. **train_dqn()** is the main training function which then proceeds to instantiate the agent from agent.py, which in return then instantiates the network architecture from network.py as well as the memory buffer from memory.py for replay learning.

The training function outputs model results into the /results folder, and also the trained model into /models.

The Navigation Results.ipynb Jupyter notebook consumes all the data from the /results folder and visualises it.

The test_dqn.py allows you to replay any trained agent by loading a model from /models. You need to specify in the test_dqn.py file which model you want to load.

## How to Train a New Agent

IMPORTANT: Please ensure you read the instructions above on Navigating the Project.

To train a new agent you need to setup an experiment in experiments.json. Feel free to remove all the experiments from there and start from scratch.

Adjusting all the core hyperparameters, e.g. learning rate, gamma, tau, epsilon decay - is straight forward. However, training custom agents, e.g. with different network architecture, will require modifying other files.

## Model Performance Summary

### Summary

The best agent was trained for 2000 episodes using the hyper parameters listed below.

The environment was solved after 930 episodes, breaking a score of 13 averaged across 100 episodes.

The agent also had a breakthrough in learning between episodes 1200 and 1300, achieving an average score of just above 15.

A video of the live agent can be found here:
https://youtu.be/Og5-pX0pGjM

The agent model is stored here:
/models/1609615254_Various Changes_model

And its results are stored here:
/results/1609615254_Various Changes_results

A graphic summarising the performance:
!["Model Results"](https://github.com/aivoric/RL-Navigation-Project/model_results.png)

### Hyperparameters used

- learning_rate: 0.0001
- gamma: 0.99
- epsilon_decay: 0.997
- epsilon_min: 0.01
- tau: 0.01
- model_fc1_units: 64
- model_fc2_units: 128
- model_fc3_units: 0   (0 = no 3rd layer was used, however an experiment was attempted)
- model_starting_weights: false   (different starting weights)
- model_dropout: false     (dropout of 30%)
- model_batch_norm: false
- double_dqn: false
- prioritised_replay: false
- dueling_dqn: false

## Future Improvements

As part of this project a testing architecture has been setup. This testing architecture should be expanded to include:
1. Further testing of different hyperparameters
2. Changes to the network architecture
3. Introducing prioritised memory replay (I've started research into this: https://knowledge.udacity.com/questions/433081)
4. Introducing Dueling DQN