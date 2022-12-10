# synchronousRL

## Introduction
Modeling synchronous swarms of fireflies is an important challenge in computational ecology that is typically calculated through the use of a synchronization model, like the Koromoto model. However, this process is not always effective given the substantial variability and noise in the datasets of synchronous fireflies. There has never been an attempt at trying to train a reinforcement learning algorithm to describe the synchrony. Moreover, synchronized phenomena have rarely been modeled with reinforcement learning. This project aims to understand the ability for reinforcement learning to operate in a synchronous environment, then compare the produced model to standard synchronization models in terms of accuracy and convergence rate. 

## Structure
Validation Data and Model Accuracy: see Expected_values.ipynb and Model_Evaluation.ipynb

Kuramoto Model: see kuramoto.ipynb

RL Model: see DQN_RL.py