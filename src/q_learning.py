import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('ggplot')


# state rewards
a, b, d, z = 1, 1, -1, -1
# list of actions
actions = ['Alpha', 'Delta']
# list of all possible states
states = [0, 1, 2, 3]
# rewards for each agent per state
state_rewards = {0:[a,b], 1:[d,z], 2:[z,d], 3:[b,a]}
# total episodes
episodes = 150
# 20 rounds per episode
rounds = 20
# discount factor
gamma = 0.9
# exploration vs exploitation factor
# starts with value 1 and decrements by 0.01 in each episode (in 100th episode epsilon = 0)
epsilon = 1
# learning rate
alpha = 0.6


def decrease_epsilon(epsilon):
    # decrease epsilon by 0.01 until 0
    if epsilon <= 0:
        return 0
    else:
        epsilon -= 0.01

        return epsilon

def initialize_q_table(agent):
    q_table = np.zeros((len(states), len(actions)))
    print("Initializing q_table for", agent, '\n', q_table)

    return q_table

def initialize_state(states):
    # initialize random state
    initial_state = np.random.choice(states)
    print("Randomly choosing initial state:", initial_state)

    return initial_state

def get_rewards(state):
    # rewards per state
    agent_reward = 0
    if state == 0:
        agent_reward = state_rewards[0][0]
    elif state == 1:
        agent_reward = state_rewards[1][0]
    elif state == 2:
        agent_reward = state_rewards[2][0]
    elif state == 3:
        agent_reward = state_rewards[3][0]

    print("Agent's reward for state", state, "is:", agent_reward)

    return agent_reward

def increment_time(timing):
    # time factor
    timing = timing + 1
    print("Time:", timing)

    return timing

def get_next_state(Agent1_action, Agent2_action):
    # next state depending on action
    next_state = 0
    if Agent1_action == 'Alpha' and Agent2_action == 'Alpha':
        next_state = 0
    elif Agent1_action == 'Alpha' and Agent2_action == 'Delta':
        next_state = 1
    elif Agent1_action == 'Delta' and Agent2_action == 'Alpha':
        next_state = 2
    elif Agent1_action == 'Delta' and Agent2_action == 'Delta':
        next_state = 3

    print("Next state:", next_state)

    return next_state

def q_table_update(q_table, state, action, reward, alpha, gamma, new_state):
    # q-table update method
    if action == 'Alpha':
        action = 0
    else:
        action = 1
    reward = get_rewards(state)

    q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state][action])
    
    return q_table

def explore(actions, agent):
    # choose random action
    ac = np.random.choice(actions)
    print(agent, 'chooses action', ac)

    return ac

def exploit(q_table, agent):
    # choose best action from q-table
    print(agent, 'chose to exploit best action from q_table')

    return np.argmax(q_table)

def explore_or_exploit(actions, q_table, epsilon, agent):
    # compare random number with epsilon and choose explore or exploit accordingly
    if np.random.random() > epsilon:
        print(agent, 'chooses to exploit')
        return exploit(q_table, agent)
    else:
        print(agent, 'chooses to explore')
        return explore(actions, agent)

def qLearning(states, actions, episodes=150, rounds=20, gamma=0.9, alpha=0.6, epsilon=1): 
    # initialize q-tables with zero values
    agent1_q_table = initialize_q_table('Agent 1')
    agent2_q_table = initialize_q_table('Agent 2')
   
    # log
    episode_lengths = np.zeros(episodes)
    eps_length = np.zeros(episodes)
    episode_rewards = np.zeros(episodes)
    
    # initialize episode time
    timing_eps = 0

    # initialize Agent 1 q-tables for plotting
    Agent1_qtable_0 = np.zeros(episodes)
    Agent1_qtable_1 = np.zeros(episodes)
    Agent1_qtable_2 = np.zeros(episodes)
    Agent1_qtable_3 = np.zeros(episodes)
    Agent1_qtable_4 = np.zeros(episodes)
    Agent1_qtable_5 = np.zeros(episodes)
    Agent1_qtable_6 = np.zeros(episodes)
    Agent1_qtable_7 = np.zeros(episodes)

    # initialize Agent 2 q-tables for plotting
    Agent2_qtable_0 = np.zeros(episodes)
    Agent2_qtable_1 = np.zeros(episodes)
    Agent2_qtable_2 = np.zeros(episodes)
    Agent2_qtable_3 = np.zeros(episodes)
    Agent2_qtable_4 = np.zeros(episodes)
    Agent2_qtable_5 = np.zeros(episodes)
    Agent2_qtable_6 = np.zeros(episodes)
    Agent2_qtable_7 = np.zeros(episodes)

    # loop for every episode
    for episode in range(episodes):
        print("Episode:", episode)

        # reset the environment and pick the first action
        state = initialize_state(states)

        # initialize round time
        timing = 0

        # loop for every round
        for roundd in range(rounds):
            print("Round:", roundd)
            
            # choose an action (explore or exploit according to epsilon)
            agent1_action = explore_or_exploit(actions, agent1_q_table, epsilon, 'Agent 1')
            agent2_action = explore_or_exploit(actions, agent2_q_table, epsilon, 'Agent 2')
            next_state = get_next_state(agent1_action, agent2_action)

            # get reward, transit to next state
            reward = get_rewards(next_state)

            # update log
            episode_rewards[episode] += reward
            episode_lengths[episode] = timing

            # update q-table according to action
            agent1_q_table = q_table_update(agent1_q_table, state, agent1_action, reward, alpha, gamma, next_state)
            print('Updating q-table for Agent 1', '\n', agent1_q_table)
            agent2_q_table = q_table_update(agent2_q_table, state, agent2_action, reward, alpha, gamma, next_state)
            print('Updating q-table for Agent 2', '\n', agent2_q_table)

            # update state
            state = next_state

            # update round time
            timing = increment_time(timing)
        
        # decrease epsilon by 0.01 until 0
        epsilon = decrease_epsilon(epsilon)
        print("epsilon value is:", epsilon)

        # update episode time
        timing_eps = increment_time(timing_eps)

        # update log
        eps_length[episode] = timing_eps

        # Agent 1, Q-value, State 1, action A
        Agent1_qtable_0[episode] = agent1_q_table[0, 0]
        # Agent 1, Q-value, State 1, action D
        Agent1_qtable_1[episode] = agent1_q_table[0, 1]
        # Agent 1, Q-value, State 2, action A
        Agent1_qtable_2[episode] = agent1_q_table[1, 0]
        # Agent 1, Q-value, State 2, action D
        Agent1_qtable_3[episode] = agent1_q_table[1, 1]
        # Agent 1, Q-value, State 3, action A
        Agent1_qtable_4[episode] = agent1_q_table[2, 0]
        # Agent 1, Q-value, State 3, action D
        Agent1_qtable_5[episode] = agent1_q_table[2, 1]
        # Agent 1, Q-value, State 4, action A
        Agent1_qtable_6[episode] = agent1_q_table[3, 0]
        # Agent 1, Q-value, State 4, action D
        Agent1_qtable_7[episode] = agent1_q_table[3, 1]

        # Agent 2, Q-value, State 1, action A
        Agent2_qtable_0[episode] = agent2_q_table[0, 0]
        # Agent 2, Q-value, State 1, action D
        Agent2_qtable_1[episode] = agent2_q_table[0, 1]
        # Agent 2, Q-value, State 2, action A
        Agent2_qtable_2[episode] = agent2_q_table[1, 0]
        # Agent 2, Q-value, State 2, action D
        Agent2_qtable_3[episode] = agent2_q_table[1, 1]
        # Agent 2, Q-value, State 3, action A
        Agent2_qtable_4[episode] = agent2_q_table[2, 0]
        # Agent 2, Q-value, State 3, action D
        Agent2_qtable_5[episode] = agent2_q_table[2, 1]
        # Agent 2, Q-value, State 4, action A
        Agent2_qtable_6[episode] = agent2_q_table[3, 0]
        # Agent 2, Q-value, State 4, action D
        Agent2_qtable_7[episode] = agent2_q_table[3, 1]

    # Agent 1 cumulative reward per episode
    plt.plot(eps_length, episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Agent 1 cumulative reward per episode')
    plt.show()

    # Agent 1 q-values for each state per episode plot
    agent1_df = pd.DataFrame({"Episodes": eps_length,
                              "State 1 - action A" : Agent1_qtable_0,
                              "State 1 - action D" : Agent1_qtable_1,
                              "State 2 - action A" : Agent1_qtable_2,
                              "State 2 - action D" : Agent1_qtable_3,
                              "State 3 - action A" : Agent1_qtable_4,
                              "State 3 - action D" : Agent1_qtable_5,
                              "State 4 - action A" : Agent1_qtable_6,
                              "State 4 - action D" : Agent1_qtable_7})

    agent1_df.plot(x="Episodes", y=["State 1 - action A",
                                    "State 1 - action D",
                                    "State 2 - action A",
                                    "State 2 - action D",
                                    "State 3 - action A",
                                    "State 3 - action D",
                                    "State 4 - action A",
                                    "State 4 - action D"])
    plt.ylabel('Q-values')
    plt.title('Agent 1 Q-values for each state per episode')
    plt.show()

    # Agent 2 cumulative reward per episode
    plt.plot(eps_length, episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Agent 2 cumulative reward per episode')
    plt.show()

    # Agent 2 q-values for each state per episode
    agent2_df = pd.DataFrame({"Episodes": eps_length,
                              "State 1 - action A" : Agent2_qtable_0,
                              "State 1 - action D" : Agent2_qtable_1,
                              "State 2 - action A" : Agent2_qtable_2,
                              "State 2 - action D" : Agent2_qtable_3,
                              "State 3 - action A" : Agent2_qtable_4,
                              "State 3 - action D" : Agent2_qtable_5,
                              "State 4 - action A" : Agent2_qtable_6,
                              "State 4 - action D" : Agent2_qtable_7})

    agent2_df.plot(x="Episodes", y=["State 1 - action A",
                                    "State 1 - action D",
                                    "State 2 - action A",
                                    "State 2 - action D",
                                    "State 3 - action A",
                                    "State 3 - action D",
                                    "State 4 - action A",
                                    "State 4 - action D"])
    plt.ylabel('Q-values')
    plt.title('Agent 2 Q-values for each state per episode')
    plt.show()


    return agent1_q_table


q_table = qLearning(states=states, actions=actions, episodes=150, rounds=20, gamma = gamma, alpha = 0.6, epsilon = 1)