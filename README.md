# Deep Reinforcement Learning Agents

This repository contains a collection of reinforcement learning algorithms written in Tensorflow. I have used some of the code from the awjuliani repository to build the code of the reinforce algorithm. You can check his tutorial in the following link: [Medium](https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724)

I modified the awjuliani REINFORCE algorithm code and [explained](https://github.com/diegoorellanaga/Reinforcement-Learning/blob/master/reinforce/code-explanation/Vanilla-policy-gradient-explanation.md) it line by line. The following modifications were made:

* **Vanilla-Policy** - An implementation of a neural network vanilla-policy-gradient agent that solves full RL problems with states, delayed rewards, and an arbitrary number of actions. We added a model saver, a model recover and the ability to visually see the policy performance in the cart-pole environment.

![Alt Text](/reinforce/code-explanation/Pictures/cartpole.gif)



I have also implemented an actor-critic algorithm for the cartpole, it currently works for any limited discrete environment. I'm currently working to expand it to the continuous action cases.
