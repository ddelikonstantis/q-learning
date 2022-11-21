# Q-Learning Epsilon-Greedy algorithm
Reinforcement Learning constitutes one of the three basic Machine Learning paradigms, alongside Supervised Learning and Unsupervised Learning. Reinforcement Learning is concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. 

## Objective
In this application, we are studying the domain of Q-Learning; a model-free Reinforcement Learning algorithm, combined with Epsilon-Greedy; a simple method for balancing an agent’s exploration versus exploitation dilemma, on a Multi-Agent environment. Specifically, we consider a stochastic game approach framework, which consists of two identical agents (clones), which interact with their environment through two possible actions (alpha, delta) and receive one of the following rewards (a=1, b=1, d=-1, z=-1). Based on the study, we experiment on a number of episodes (150), rounds of each episode (20) and epsilon settings. The results suggest that, as time goes by, each agent chooses the best strategy (actions) for himself, which gives him the best payoff (reward), meaning that our multi-agent Q-Learning algorithm converges to a Nash equilibrium.

## Dependencies
```sh
pip install -r requirements.txt
```