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

### Model Architecture

The architecture was a simple Linear model which:
- Consumed an input vector size 37 (37 state dimensions)
- 2 inner layers: 64, and 128 neurons in size
- 4 outputs (one for each action)

Some attempts were made at testing dropouts, batch normalisation, and different weight initialisation. However, more exploration is required in that direction.

### Hyperparameters used

To understand some of the parameters below, please read the README in regards to the experiments framework used in this project.

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