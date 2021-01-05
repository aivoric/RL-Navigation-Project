## Performance Report

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
!["Model Results"](https://github.com/aivoric/RL-Navigation-Project/blob/master/model_results.png?raw=true)

### Learning Algorithm

The algorithm is based on a Double DQN implementation with a memory replay mechanism.

There are also 2 identical Q value networks which are instantiated right at the start based on a simple Linear architecture (see section below for Model Architecture).

For every step the agent makes in the environment using one of the 4 above discrete actions, the following happens:
1. The data from the step is added to a memory buffer (state, action, reward, next_state, done)
2. Every 4 steps, and if there are sufficient samples in the buffer, the agent attempts to learn.
3. Learning is done by randomly sampling experiences from the memory buffer.
4. For every sampled experience calculate 2 Q values: your current model, and your new model with the new experience + obtained reward
5. Calculate the loss based on the 2 Q values from the network outputs
6. Backpropagate your main model to update the weights
7. Gently update the target model weights based on your main model weights. The TAU hyperparameter controls this update.


### Model Architecture

The architecture is a simple Linear model which:
- Consumes an input vector size 37 (37 state dimensions)
- Has 2 inner layers: 64, and 128 neurons in size
- Has 4 outputs (one for each action)

Some attempts were made at testing dropouts, batch normalisation, and different weight initialisation. However, more exploration is required in that direction.

### Hyperparameters

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