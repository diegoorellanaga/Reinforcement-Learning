# This folder contains one code file until now: 
## actor-critic.py

## Code general explanation:

This code implements the following algoritm:

![title](images/algorithm.png)

This algorithm was taken from the book "Reinforcement Learning An Introduction, second edition by Richard S. Sutton and Andrew G. Barto"

The code is divided into two sections. The agent class declaration and agent training.

You can modify the following agent parameters:

**Agent instantiation**

actor neural network parameters:

 - number of layers
 - number of neurons on each layer
 - type of decay for the learning rate
 - the activation function for the layers
 
critic neural network parameters:

 - number of layers
 - number of neurons on each layer
 - type of decay for the learning rate
 - the activation function for the layers

**Training**

 - number of episodes
 - the number of samples to consider before updating the agent weights.
 - batch size


### TODO

 - Ability to work under continuous actions
 - the option of natural gradients



 
